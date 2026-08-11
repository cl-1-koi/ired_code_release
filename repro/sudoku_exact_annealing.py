"""Constraint-oracle simulated annealing sanity check for Sudoku search."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal environments
    njit = None

from repro.sudoku_energy_calibration import exact_conflict_energy, load_dataset
from repro.sudoku_metrics import clue_cells, decode_digits, strict_validity


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def row_column_energy(grid: torch.Tensor) -> int:
    onehot = torch.nn.functional.one_hot(grid.long(), num_classes=9)
    rows = onehot.sum(dim=1)
    columns = onehot.sum(dim=0)
    return int((rows - 1).clamp_min(0).sum() + (columns - 1).clamp_min(0).sum())


def _duplicate_energy(counts: np.ndarray) -> int:
    return int(np.maximum(counts - 1, 0).sum())


def _apply_count_change(
    counts: np.ndarray, group: int, digit: int, change: int
) -> int:
    before = max(int(counts[group, digit]) - 1, 0)
    counts[group, digit] += change
    after = max(int(counts[group, digit]) - 1, 0)
    return after - before


def _apply_swap_counts(
    row_counts: np.ndarray,
    column_counts: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    first_digit: int,
    second_digit: int,
) -> int:
    """Apply a proposed swap to count tables and return its exact energy delta."""
    if first_digit == second_digit:
        return 0
    first_row, first_col = first
    second_row, second_col = second
    delta = 0
    if first_row != second_row:
        delta += _apply_count_change(row_counts, first_row, first_digit, -1)
        delta += _apply_count_change(row_counts, first_row, second_digit, 1)
        delta += _apply_count_change(row_counts, second_row, second_digit, -1)
        delta += _apply_count_change(row_counts, second_row, first_digit, 1)
    if first_col != second_col:
        delta += _apply_count_change(column_counts, first_col, first_digit, -1)
        delta += _apply_count_change(column_counts, first_col, second_digit, 1)
        delta += _apply_count_change(column_counts, second_col, second_digit, -1)
        delta += _apply_count_change(column_counts, second_col, first_digit, 1)
    return delta


if njit is not None:
    @njit(cache=True)
    def _numba_penalty(count: int) -> int:
        return count - 1 if count > 1 else 0


    @njit(cache=True)
    def _numba_change(counts, group: int, digit: int, change: int) -> int:
        before = _numba_penalty(counts[group, digit])
        counts[group, digit] += change
        return _numba_penalty(counts[group, digit]) - before


    @njit(cache=True)
    def _numba_swap_delta(
        row_counts, column_counts, first_row, first_col, second_row, second_col,
        first_digit, second_digit,
    ) -> int:
        if first_digit == second_digit:
            return 0
        delta = 0
        if first_row != second_row:
            delta += _numba_change(row_counts, first_row, first_digit, -1)
            delta += _numba_change(row_counts, first_row, second_digit, 1)
            delta += _numba_change(row_counts, second_row, second_digit, -1)
            delta += _numba_change(row_counts, second_row, first_digit, 1)
        if first_col != second_col:
            delta += _numba_change(column_counts, first_col, first_digit, -1)
            delta += _numba_change(column_counts, first_col, second_digit, 1)
            delta += _numba_change(column_counts, second_col, second_digit, -1)
            delta += _numba_change(column_counts, second_col, first_digit, 1)
        return delta


    @njit(cache=True)
    def _numba_anneal(
        initial_grid, mutable_positions, mutable_lengths, steps,
        start_temperature, cooling, seed,
    ):
        np.random.seed(seed)
        grid = initial_grid.copy()
        row_counts = np.zeros((9, 9), dtype=np.int16)
        column_counts = np.zeros((9, 9), dtype=np.int16)
        for row in range(9):
            for col in range(9):
                digit = grid[row, col]
                row_counts[row, digit] += 1
                column_counts[col, digit] += 1
        energy = 0
        for group in range(9):
            for digit in range(9):
                energy += _numba_penalty(row_counts[group, digit])
                energy += _numba_penalty(column_counts[group, digit])
        best_energy = energy
        best_grid = grid.copy()
        accepted = 0
        temperature = start_temperature
        no_improvement = 0
        for proposal in range(steps):
            box = np.random.randint(len(mutable_lengths))
            length = mutable_lengths[box]
            first_offset = np.random.randint(length)
            second_offset = np.random.randint(length - 1)
            if second_offset >= first_offset:
                second_offset += 1
            first_row = mutable_positions[box, first_offset, 0]
            first_col = mutable_positions[box, first_offset, 1]
            second_row = mutable_positions[box, second_offset, 0]
            second_col = mutable_positions[box, second_offset, 1]
            first_digit = grid[first_row, first_col]
            second_digit = grid[second_row, second_col]
            delta = _numba_swap_delta(
                row_counts, column_counts, first_row, first_col, second_row,
                second_col, first_digit, second_digit,
            )
            accept = delta <= 0 or np.random.random() < math.exp(
                -delta / max(temperature, 1e-9)
            )
            if accept:
                grid[first_row, first_col] = second_digit
                grid[second_row, second_col] = first_digit
                energy += delta
                accepted += 1
                if energy < best_energy:
                    best_energy = energy
                    best_grid = grid.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                _numba_swap_delta(
                    row_counts, column_counts, first_row, first_col, second_row,
                    second_col, second_digit, first_digit,
                )
                no_improvement += 1
            if energy == 0:
                return grid, 0, proposal + 1, accepted
            temperature *= cooling
            if no_improvement >= 5_000:
                temperature = max(0.5 * start_temperature, temperature)
                no_improvement = 0
        return best_grid, best_energy, steps, accepted


def initialize_boxes(
    puzzle: torch.Tensor, clues: torch.Tensor, generator: random.Random
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    grid = puzzle.clone()
    mutable_boxes = []
    for box_row in range(3):
        for box_col in range(3):
            positions = []
            present = set()
            for row in range(box_row * 3, box_row * 3 + 3):
                for col in range(box_col * 3, box_col * 3 + 3):
                    if bool(clues[row, col]):
                        present.add(int(puzzle[row, col]))
                    else:
                        positions.append((row, col))
            missing = [digit for digit in range(9) if digit not in present]
            if len(missing) != len(positions):
                raise ValueError("invalid puzzle clues within a box")
            generator.shuffle(missing)
            for (row, col), digit in zip(positions, missing):
                grid[row, col] = digit
            if len(positions) >= 2:
                mutable_boxes.append(positions)
    return grid, mutable_boxes


def anneal_board(
    puzzle: torch.Tensor,
    clues: torch.Tensor,
    *,
    seed: int,
    restarts: int,
    steps_per_restart: int,
    start_temperature: float,
    cooling: float,
    engine: str = "auto",
) -> dict:
    if engine not in ("auto", "python", "numba"):
        raise ValueError(engine)
    use_numba = engine == "numba" or (engine == "auto" and njit is not None)
    if engine == "numba" and njit is None:
        raise RuntimeError("numba engine requested but numba is not installed")
    generator = random.Random(seed)
    best_grid = None
    best_energy = 10**9
    total_proposals = 0
    total_accepted = 0
    completed_restarts = 0
    for restart in range(restarts):
        grid, mutable_boxes = initialize_boxes(puzzle, clues, generator)
        if not mutable_boxes:
            energy = row_column_energy(grid)
            return {"grid": grid, "energy": energy, "solved": energy == 0,
                    "proposals": total_proposals, "accepted": total_accepted,
                    "restarts": restart + 1}
        if use_numba:
            positions = np.zeros((len(mutable_boxes), 9, 2), dtype=np.int64)
            lengths = np.zeros(len(mutable_boxes), dtype=np.int64)
            for box, mutable in enumerate(mutable_boxes):
                lengths[box] = len(mutable)
                for position, (row, col) in enumerate(mutable):
                    positions[box, position] = (row, col)
            candidate, energy, proposals, accepted = _numba_anneal(
                grid.numpy().astype(np.int64),
                positions,
                lengths,
                steps_per_restart,
                start_temperature,
                cooling,
                seed + restart * 10_007,
            )
            total_proposals += int(proposals)
            total_accepted += int(accepted)
            candidate_grid = torch.from_numpy(candidate.copy())
            if energy < best_energy:
                best_energy, best_grid = int(energy), candidate_grid
            if energy == 0:
                return {"grid": candidate_grid, "energy": 0, "solved": True,
                        "proposals": total_proposals, "accepted": total_accepted,
                        "restarts": restart + 1}
            completed_restarts = restart + 1
            continue
        grid_array = grid.numpy().copy()
        row_counts = np.zeros((9, 9), dtype=np.int16)
        column_counts = np.zeros((9, 9), dtype=np.int16)
        for row in range(9):
            for col in range(9):
                digit = int(grid_array[row, col])
                row_counts[row, digit] += 1
                column_counts[col, digit] += 1
        energy = _duplicate_energy(row_counts) + _duplicate_energy(column_counts)
        if energy < best_energy:
            best_energy, best_grid = energy, grid.clone()
        temperature = start_temperature
        no_improvement = 0
        for _ in range(steps_per_restart):
            total_proposals += 1
            positions = generator.choice(mutable_boxes)
            first, second = generator.sample(positions, 2)
            first_digit = int(grid_array[first])
            second_digit = int(grid_array[second])
            delta = _apply_swap_counts(
                row_counts, column_counts, first, second, first_digit, second_digit
            )
            proposed = energy + delta
            accept = delta <= 0 or generator.random() < math.exp(-delta / max(temperature, 1e-9))
            if accept:
                grid_array[first], grid_array[second] = second_digit, first_digit
                energy = proposed
                total_accepted += 1
                if energy < best_energy:
                    best_energy = energy
                    best_grid = torch.from_numpy(grid_array.copy())
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                reverse_delta = _apply_swap_counts(
                    row_counts, column_counts, first, second, second_digit, first_digit
                )
                if reverse_delta != -delta:
                    raise RuntimeError("incremental energy rollback mismatch")
                no_improvement += 1
            if energy == 0:
                solved_grid = torch.from_numpy(grid_array.copy())
                return {"grid": solved_grid, "energy": 0, "solved": True,
                        "proposals": total_proposals, "accepted": total_accepted,
                        "restarts": restart + 1}
            temperature *= cooling
            if no_improvement >= 5_000:
                temperature = max(0.5 * start_temperature, temperature)
                no_improvement = 0
        completed_restarts = restart + 1
    assert best_grid is not None
    return {"grid": best_grid, "energy": best_energy, "solved": best_energy == 0,
            "proposals": total_proposals, "accepted": total_accepted,
            "restarts": completed_restarts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("standard-val", "hard-test"), default="standard-val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--steps-per-restart", type=int, default=100_000)
    parser.add_argument("--start-temperature", type=float, default=2.0)
    parser.add_argument("--cooling", type=float, default=0.9995)
    parser.add_argument("--engine", choices=("auto", "python", "numba"), default="auto")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if min(args.limit, args.restarts, args.steps_per_restart) < 1:
        raise SystemExit("limit, restarts, and steps must be positive")
    dataset = load_dataset(args.dataset)
    if args.start_index < 0 or args.start_index + args.limit > len(dataset):
        raise SystemExit("requested dataset slice is out of range")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for index in range(args.start_index, args.start_index + args.limit):
        inp, label, mask = dataset[index]
        clues = clue_cells(mask.unsqueeze(0))[0]
        puzzle = decode_digits(inp.unsqueeze(0))[0]
        result = anneal_board(
            puzzle,
            clues,
            seed=args.seed + index * 1_000_003,
            restarts=args.restarts,
            steps_per_restart=args.steps_per_restart,
            start_temperature=args.start_temperature,
            cooling=args.cooling,
            engine=args.engine,
        )
        reference = decode_digits(label.unsqueeze(0))[0]
        strict = bool(strict_validity(result["grid"].unsqueeze(0))[0])
        clue_consistent = bool((result["grid"][clues] == puzzle[clues]).all())
        rows.append({
            "dataset_index": index,
            "solved": bool(result["solved"] and strict and clue_consistent),
            "strict_valid": strict,
            "clue_consistent": clue_consistent,
            "exact_reference": bool((result["grid"] == reference).all()),
            "final_energy": int(result["energy"]),
            "full_conflict_energy": int(exact_conflict_energy(result["grid"].unsqueeze(0))[0]),
            "proposals": int(result["proposals"]),
            "accepted": int(result["accepted"]),
            "restarts": int(result["restarts"]),
        })
    summary = {
        "schema": "ired/sudoku-exact-annealing-v1",
        "created_utc": utc_now(),
        "dataset": args.dataset,
        "slice": {"start": args.start_index,
                  "end_exclusive": args.start_index + args.limit, "n": args.limit},
        "search": {"seed": args.seed, "restarts": args.restarts,
                   "steps_per_restart": args.steps_per_restart,
                   "start_temperature": args.start_temperature, "cooling": args.cooling,
                   "engine": ("numba" if args.engine == "auto" and njit is not None else args.engine),
                   "box_legality_preserved": True,
                   "energy": "row_and_column_duplicate_count"},
        "metrics": {
            "solved": sum(row["solved"] for row in rows),
            "solve_rate": sum(row["solved"] for row in rows) / len(rows),
            "exact_reference_rate": sum(row["exact_reference"] for row in rows) / len(rows),
            "mean_proposals": sum(row["proposals"] for row in rows) / len(rows),
            "mean_acceptance_rate": sum(row["accepted"] for row in rows) /
                                    max(1, sum(row["proposals"] for row in rows)),
        },
        "rows": rows,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
