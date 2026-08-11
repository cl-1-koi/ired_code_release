"""Instrument the released IRED Sudoku search without changing its updates."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from repro.sudoku_energy_calibration import (
    Accumulator,
    exact_conflict_energy,
    load_dataset,
    load_weight_source,
)
from repro.sudoku_metrics import decode_digits, sudoku_batch_metrics
from repro.train_sudoku import sha256_file, verify_manifest_seal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def capture_code_provenance() -> dict:
    """Freeze provenance before a long evaluation can outlive source edits."""
    repo = Path(__file__).resolve().parents[1]
    dependency_paths = [
        Path(__file__).resolve(),
        repo / "repro/sudoku_energy_calibration.py",
        repo / "repro/sudoku_metrics.py",
        repo / "repro/train_sudoku.py",
        repo / "diffusion_lib/denoising_diffusion_pytorch_1d.py",
        repo / "models.py",
    ]
    status = git("status", "--porcelain")
    return {
        "captured_utc": utc_now(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in dependency_paths
        },
        "cuda_determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
    }


def _mean_metric(metric: torch.Tensor) -> torch.Tensor:
    return metric.detach().float().reshape(-1)


def trace_restart(
    model, inp, label, mask, *, inner_steps: int, seed: int, reverse_noise: bool
):
    """Run one exact released sampler trajectory and return per-board telemetry."""
    device = inp.device
    torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    img = torch.randn((len(inp), *model.out_shape), generator=generator, device=device)
    landscapes = []
    final_prediction = None
    for timestep in reversed(range(model.num_timesteps)):
        t = torch.full((len(inp),), timestep, device=device, dtype=torch.long)
        clue_value = model.q_sample(inp, t, noise=torch.zeros_like(inp))
        img = img * (1 - mask) + clue_value * mask
        img, _ = model.p_sample(
            inp, img, timestep, None, scale=False, with_noise=reverse_noise
        )
        img = img * (1 - mask) + clue_value * mask

        accepted_steps = torch.zeros(len(inp), device=device)
        energy_changes = torch.zeros(len(inp), device=device)
        gradient_norms = torch.zeros(len(inp), device=device)
        for _ in range(inner_steps):
            current = img.detach().requires_grad_(True)
            energy = model.model(inp, current, t, return_energy=True).reshape(-1)
            gradient = torch.autograd.grad(energy.sum(), current)[0]
            proposal = current.detach() - model.opt_step_size[timestep] * gradient.detach()
            proposal = proposal * (1 - mask) + clue_value * mask
            maximum = float(model.sqrt_alphas_cumprod[timestep])
            proposal = proposal.clamp(-maximum, maximum)
            with torch.no_grad():
                proposed_energy = model.model(
                    inp, proposal, t, return_energy=True
                ).reshape(-1)
            accepted = proposed_energy <= energy.detach()
            img = torch.where(accepted[:, None], proposal, current.detach())
            accepted_steps += accepted.float()
            energy_changes += torch.where(
                accepted, proposed_energy - energy.detach(), torch.zeros_like(energy)
            )
            gradient_norms += (gradient.detach() * (1 - mask)).norm(dim=1)

        img = img.detach().clamp(-float(model.sqrt_alphas_cumprod[timestep]),
                                 float(model.sqrt_alphas_cumprod[timestep]))
        prediction = model.predict_start_from_noise(img, t, torch.zeros_like(img))
        metrics = sudoku_batch_metrics(prediction, label, mask)
        with torch.no_grad():
            learned_energy = model.model(inp, img, t, return_energy=True).reshape(-1)
        conflicts = exact_conflict_energy(metrics.predicted_digits).to(device)
        unknown_errors = (
            (1 - metrics.unknown_cell_accuracy) *
            (~mask.reshape(-1, 9, 9, 9)[..., 0].bool()).sum(dim=(1, 2))
        )
        landscapes.append({
            "timestep": timestep,
            "learned_energy": learned_energy.detach(),
            "accept_fraction": accepted_steps / inner_steps,
            "accepted_energy_change": energy_changes,
            "gradient_norm": gradient_norms / inner_steps,
            "strict_valid": metrics.strict_valid.to(device),
            "clue_consistent": metrics.clue_consistent.to(device),
            "exact_reference": metrics.exact_solution.to(device),
            "unknown_accuracy": metrics.unknown_cell_accuracy.to(device),
            "exact_conflict_energy": conflicts,
            "unknown_errors": unknown_errors.to(device),
        })
        final_prediction = prediction.detach()
        if timestep != 0:
            previous_t = t - 1
            img = model.sqrt_alphas_cumprod[previous_t].reshape(-1, 1) * prediction
    assert final_prediction is not None
    return landscapes, final_prediction, landscapes[-1]["learned_energy"]


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if float(left_rank.std()) == 0 or float(right_rank.std()) == 0:
        return None
    return float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1])


def evaluate_search(
    model,
    dataset,
    *,
    start_index: int,
    limit: int,
    batch_size: int,
    restarts: int,
    inner_steps: int,
    seed: int,
    reverse_noise: bool,
) -> tuple[list[dict], list[dict]]:
    device = next(model.parameters()).device
    trajectory = defaultdict(Accumulator)
    restart_metrics = defaultdict(Accumulator)
    within_board_correlations = defaultdict(lambda: {"conflict": [], "errors": []})
    end = start_index + limit
    for offset in range(start_index, end, batch_size):
        batch_end = min(end, offset + batch_size)
        rows = [dataset[index] for index in range(offset, batch_end)]
        inp = torch.stack([row[0] for row in rows]).to(device)
        label = torch.stack([row[1] for row in rows]).to(device)
        mask = torch.stack([row[2] for row in rows]).float().to(device)
        predictions = []
        energies = []
        per_restart_solved = []
        for restart in range(restarts):
            landscapes, prediction, final_energy = trace_restart(
                model,
                inp,
                label,
                mask,
                inner_steps=inner_steps,
                seed=seed + offset * 1_000_003 + restart * 10_007,
                reverse_noise=reverse_noise,
            )
            predictions.append(prediction)
            energies.append(final_energy)
            final_metrics = sudoku_batch_metrics(prediction, label, mask)
            per_restart_solved.append(final_metrics.strict_valid & final_metrics.clue_consistent)
            for landscape in landscapes:
                key = landscape["timestep"]
                trajectory[key].add(**{
                    name: _mean_metric(value)
                    for name, value in landscape.items()
                    if name != "timestep"
                })

        stacked_predictions = torch.stack(predictions)
        stacked_energies = torch.stack(energies)
        stacked_solved = torch.stack(per_restart_solved).to(device)
        stacked_digits = torch.stack([
            decode_digits(prediction) for prediction in stacked_predictions
        ])
        for prefix in (1, 2, 4, 8, 16):
            if prefix > restarts:
                continue
            prefix_energy = stacked_energies[:prefix]
            best_index = prefix_energy.argmin(dim=0)
            board_index = torch.arange(len(inp), device=device)
            selected = stacked_predictions[:prefix][best_index, board_index]
            selected_metrics = sudoku_batch_metrics(selected, label, mask)
            selected_conflicts = exact_conflict_energy(selected_metrics.predicted_digits).to(device)
            blank_count = (~mask.reshape(-1, 9, 9, 9)[..., 0].bool()).sum(dim=(1, 2))
            selected_errors = (1 - selected_metrics.unknown_cell_accuracy.to(device)) * blank_count
            hamming_from_first = (
                stacked_digits[:prefix] != stacked_digits[0:1]
            ).sum(dim=(2, 3))
            diverse = (hamming_from_first > 0).any(dim=0)
            unique_counts = []
            for board in range(len(inp)):
                flattened = stacked_digits[:prefix, board].reshape(prefix, -1)
                unique_counts.append(len(torch.unique(flattened, dim=0)))
            restart_metrics[prefix].add(
                first_solved=stacked_solved[0],
                any_solved=stacked_solved[:prefix].any(dim=0),
                selected_solved=(selected_metrics.strict_valid & selected_metrics.clue_consistent),
                selected_unknown_accuracy=selected_metrics.unknown_cell_accuracy,
                selected_conflict_energy=selected_conflicts,
                selected_unknown_errors=selected_errors,
                restart_diverse=diverse,
                unique_decoded_boards=torch.tensor(unique_counts, device=device),
                hamming_from_first=hamming_from_first.float().mean(dim=0),
            )
            all_metrics = [
                sudoku_batch_metrics(candidate, label, mask)
                for candidate in stacked_predictions[:prefix]
            ]
            all_conflicts = torch.stack([
                exact_conflict_energy(metric.predicted_digits) for metric in all_metrics
            ]).to(device)
            all_errors = torch.stack([
                (1 - metric.unknown_cell_accuracy.to(device)) * blank_count
                for metric in all_metrics
            ])
            if prefix >= 2:
                for board in range(len(inp)):
                    conflict_correlation = spearman(
                        prefix_energy[:, board], all_conflicts[:, board]
                    )
                    error_correlation = spearman(
                        prefix_energy[:, board], all_errors[:, board]
                    )
                    if conflict_correlation is not None:
                        within_board_correlations[prefix]["conflict"].append(
                            conflict_correlation
                        )
                    if error_correlation is not None:
                        within_board_correlations[prefix]["errors"].append(error_correlation)

    trajectory_records = []
    for timestep in sorted(trajectory, reverse=True):
        values = trajectory[timestep]
        trajectory_records.append({
            "timestep": timestep,
            "n_restart_boards": len(values.tensor("learned_energy")),
            "learned_energy_mean": values.mean("learned_energy"),
            "proposal_accept_fraction": values.mean("accept_fraction"),
            "accepted_energy_change_mean": values.mean("accepted_energy_change"),
            "gradient_norm_mean": values.mean("gradient_norm"),
            "strict_valid_rate": values.mean("strict_valid"),
            "clue_consistency_rate": values.mean("clue_consistent"),
            "exact_reference_rate": values.mean("exact_reference"),
            "unknown_accuracy": values.mean("unknown_accuracy"),
            "exact_conflict_energy_mean": values.mean("exact_conflict_energy"),
            "unknown_errors_mean": values.mean("unknown_errors"),
        })
    restart_records = []
    for prefix in sorted(restart_metrics):
        values = restart_metrics[prefix]
        correlations = within_board_correlations[prefix]
        restart_records.append({
            "restart_prefix": prefix,
            "n_boards": len(values.tensor("first_solved")),
            "first_restart_solve_rate": values.mean("first_solved"),
            "oracle_any_restart_solve_rate": values.mean("any_solved"),
            "minimum_learned_energy_solve_rate": values.mean("selected_solved"),
            "minimum_energy_unknown_accuracy": values.mean("selected_unknown_accuracy"),
            "minimum_energy_exact_conflict_mean": values.mean("selected_conflict_energy"),
            "minimum_energy_unknown_errors_mean": values.mean("selected_unknown_errors"),
            "fraction_boards_with_restart_diversity": values.mean("restart_diverse"),
            "mean_unique_decoded_boards": values.mean("unique_decoded_boards"),
            "mean_hamming_cells_from_first_restart": values.mean("hamming_from_first"),
            "within_board_spearman_n_exact_conflict": len(correlations["conflict"]),
            "within_board_spearman_energy_vs_exact_conflict": (
                sum(correlations["conflict"]) / len(correlations["conflict"])
                if correlations["conflict"] else None
            ),
            "within_board_spearman_n_unknown_errors": len(correlations["errors"]),
            "within_board_spearman_energy_vs_unknown_errors": (
                sum(correlations["errors"]) / len(correlations["errors"])
                if correlations["errors"] else None
            ),
        })
    return trajectory_records, restart_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("standard-val", "hard-test"), default="standard-val")
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--innerloop-steps", type=int, default=20)
    parser.add_argument(
        "--reverse-noise", action="store_true",
        help="diagnostic arm: add posterior noise during reverse diffusion",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if min(args.limit, args.batch_size, args.restarts, args.innerloop_steps) < 1:
        raise SystemExit("limit, batch size, restarts, and inner steps must be positive")

    code_provenance = capture_code_provenance()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    seal = verify_manifest_seal(manifest)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != seal:
        raise RuntimeError("checkpoint and sealed manifest do not match")
    dataset = load_dataset(args.dataset)
    if args.start_index < 0 or args.start_index + args.limit > len(dataset):
        raise SystemExit("requested dataset slice is out of range")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    model = load_weight_source(payload, manifest, args.weights, torch.device("cuda", 0))
    trajectory, restarts = evaluate_search(
        model,
        dataset,
        start_index=args.start_index,
        limit=args.limit,
        batch_size=args.batch_size,
        restarts=args.restarts,
        inner_steps=args.innerloop_steps,
        seed=args.seed,
        reverse_noise=args.reverse_noise,
    )
    summary = {
        "schema": "ired/sudoku-search-calibration-v1",
        "created_utc": utc_now(),
        "code": code_provenance,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                       "step": int(payload["step"]), "manifest_seal": seal},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "dataset": args.dataset,
        "slice": {"start": args.start_index,
                  "end_exclusive": args.start_index + args.limit, "n": args.limit},
        "sampling": {"weight_source": args.weights, "seed": args.seed,
                     "restarts": args.restarts, "innerloop_steps": args.innerloop_steps,
                     "diffusion_landscapes": 10,
                     "reverse_noise": args.reverse_noise,
                     "update_rule": (
                         "released_code_with_reverse_noise_diagnostic"
                         if args.reverse_noise else "released_code_exact"
                     )},
        "trajectory": trajectory,
        "restart_selection": restarts,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
