"""Frozen learned-energy diagnostics for the released IRED Sudoku model."""

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
import torch.nn.functional as F

from repro.sudoku_metrics import clue_cells, decode_digits
from repro.train_sudoku import (
    build_model,
    sha256_file,
    verify_manifest_seal,
)
from sat_dataset import SudokuDataset, SudokuRRNDataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    probability = successes / total
    denominator = 1 + z * z / total
    centre = (probability + z * z / (2 * total)) / denominator
    radius = z * (
        (probability * (1 - probability) / total + z * z / (4 * total * total)) ** 0.5
    ) / denominator
    return [centre - radius, centre + radius]


def load_dataset(name: str):
    if name == "standard-train":
        return SudokuDataset("sudoku", split="train")
    if name == "standard-val":
        return SudokuDataset("sudoku", split="val")
    if name == "hard-test":
        return SudokuRRNDataset("sudoku-rrn", split="test")
    if name == "rrn-valid":
        return SudokuRRNDataset("sudoku-rrn", split="val")
    raise ValueError(name)


def exact_conflict_energy(digits: torch.Tensor) -> torch.Tensor:
    """Count duplicate digits across rows, columns, and boxes per board."""
    if digits.ndim != 3 or digits.shape[1:] != (9, 9):
        raise ValueError(f"expected (B,9,9), got {tuple(digits.shape)}")
    onehot = F.one_hot(digits.long(), num_classes=9)
    rows = onehot.sum(dim=2)
    cols = onehot.sum(dim=1)
    boxes = onehot.reshape(-1, 3, 3, 3, 3, 9).sum(dim=(2, 4))
    penalties = [
        (counts - 1).clamp_min(0).flatten(start_dim=1).sum(dim=1)
        for counts in (rows, cols, boxes)
    ]
    return penalties[0] + penalties[1] + penalties[2]


def corrupt_solutions(
    labels: torch.Tensor,
    masks: torch.Tensor,
    distance: int,
    dataset_indices: list[int],
    seed: int,
) -> torch.Tensor:
    """Make deterministic exact-distance corruptions restricted to blank cells."""
    if distance < 1:
        raise ValueError("distance must be positive")
    digits = decode_digits(labels).cpu()
    clues = clue_cells(masks).cpu()
    if len(dataset_indices) != len(digits):
        raise ValueError("dataset index count does not match batch")
    corrupt = digits.clone()
    for row, dataset_index in enumerate(dataset_indices):
        unknown = (~clues[row]).nonzero(as_tuple=False)
        if len(unknown) < distance:
            raise ValueError(
                f"dataset index {dataset_index} has {len(unknown)} blanks, "
                f"fewer than requested distance {distance}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1_000_003 * dataset_index + 9_176 * distance)
        selected = unknown[torch.randperm(len(unknown), generator=generator)[:distance]]
        for ordinal, (board_row, board_col) in enumerate(selected.tolist()):
            original = int(digits[row, board_row, board_col])
            corrupt[row, board_row, board_col] = (original + 1 + ordinal % 8) % 9
    onehot = F.one_hot(corrupt.long(), num_classes=9).to(labels.dtype)
    return ((onehot - 0.5) * 2).reshape_as(labels)


def _derived_seed(seed: int, offset: int, timestep: int, distance: int) -> int:
    return int(seed + offset * 1_000_003 + timestep * 10_007 + distance * 101)


def _masked_mse(values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    unknown = 1.0 - mask
    return ((values - target).square() * unknown).sum(dim=1) / unknown.sum(dim=1).clamp_min(1)


def _unknown_errors(values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    predicted = decode_digits(values)
    target_digits = decode_digits(target)
    unknown = ~clue_cells(mask)
    return ((predicted != target_digits) & unknown).sum(dim=(1, 2))


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=1)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    return numerator / denominator.clamp_min(1e-12)


class Accumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, **values: torch.Tensor) -> None:
        for key, value in values.items():
            self.values[key].append(value.detach().float().cpu().reshape(-1))

    def tensor(self, key: str) -> torch.Tensor:
        return torch.cat(self.values[key])

    def mean(self, key: str) -> float:
        return float(self.tensor(key).mean())

    def median(self, key: str) -> float:
        return float(self.tensor(key).median())


def load_weight_source(
    payload: dict,
    manifest: dict,
    weight_source: str,
    device: torch.device,
):
    kernel = int(manifest.get("model", {}).get("final_conv_kernel", 1))
    diffusion = build_model(729, 729, innerloop_steps=20, final_conv_kernel=kernel).to(device)
    if weight_source == "raw":
        diffusion.load_state_dict(payload["model"])
        model = diffusion
    elif weight_source == "ema":
        from ema_pytorch import EMA

        ema = EMA(diffusion, beta=0.995, update_every=10).to(device)
        ema.load_state_dict(payload["ema"])
        model = ema.ema_model
    else:
        raise ValueError(weight_source)
    model.out_dim = 729
    model.out_shape = (729,)
    model.eval()
    return model


def evaluate_pairs(
    model,
    dataset,
    *,
    start_index: int,
    limit: int,
    batch_size: int,
    distances: list[int],
    timesteps: list[int],
    seed: int,
) -> list[dict]:
    device = next(model.parameters()).device
    accumulators = {
        (timestep, distance): Accumulator()
        for timestep in timesteps
        for distance in distances
    }
    end = min(len(dataset), start_index + limit)
    for offset in range(start_index, end, batch_size):
        batch_end = min(end, offset + batch_size)
        indices = list(range(offset, batch_end))
        rows = [dataset[index] for index in indices]
        inp_cpu = torch.stack([row[0] for row in rows])
        label_cpu = torch.stack([row[1] for row in rows])
        mask_cpu = torch.stack([row[2] for row in rows]).float()
        inp = inp_cpu.to(device)
        label = label_cpu.to(device)
        mask = mask_cpu.to(device)
        target_digits = decode_digits(label_cpu)
        target_exact_energy = exact_conflict_energy(target_digits)
        for distance in distances:
            corrupt_cpu = corrupt_solutions(label_cpu, mask_cpu, distance, indices, seed)
            corrupt = corrupt_cpu.to(device)
            corrupt_exact_energy = exact_conflict_energy(decode_digits(corrupt_cpu))
            exact_margin = corrupt_exact_energy - target_exact_energy
            for timestep in timesteps:
                t = torch.full((len(rows),), timestep, device=device, dtype=torch.long)
                generator = torch.Generator(device=device)
                generator.manual_seed(_derived_seed(seed, offset, timestep, distance))
                noise = torch.randn(label.shape, generator=generator, device=device)
                valid_noisy = model.q_sample(label, t, noise=noise)
                corrupt_noisy = model.q_sample(corrupt, t, noise=noise)
                clue_value = model.q_sample(inp, t, noise=torch.zeros_like(inp))
                valid_noisy = valid_noisy * (1 - mask) + clue_value * mask
                corrupt_noisy = corrupt_noisy * (1 - mask) + clue_value * mask

                with torch.no_grad():
                    paired_inp = torch.cat((inp, inp), dim=0)
                    paired_state = torch.cat((valid_noisy, corrupt_noisy), dim=0)
                    paired_t = torch.cat((t, t), dim=0)
                    paired_energy = model.model(
                        paired_inp, paired_state, paired_t, return_energy=True
                    ).reshape(-1)
                    energy_valid, energy_corrupt = paired_energy.chunk(2)

                current = corrupt_noisy.detach().requires_grad_(True)
                energy = model.model(inp, current, t, return_energy=True).reshape(-1)
                gradient = torch.autograd.grad(energy.sum(), current)[0]
                effective_descent = -gradient * (1 - mask)
                correction = (valid_noisy - current.detach()) * (1 - mask)
                cosine = _safe_cosine(effective_descent, correction)
                gradient_norm = effective_descent.norm(dim=1)

                step_size = model.opt_step_size[timestep]
                proposal = current.detach() - step_size * gradient.detach()
                proposal = proposal * (1 - mask) + clue_value * mask
                max_value = float(model.sqrt_alphas_cumprod[timestep])
                proposal = proposal.clamp(-max_value, max_value)
                with torch.no_grad():
                    proposal_energy = model.model(
                        inp, proposal, t, return_energy=True
                    ).reshape(-1)
                accepted = proposal_energy <= energy.detach()
                accepted_state = torch.where(
                    accepted[:, None], proposal, current.detach()
                )
                mse_before = _masked_mse(current.detach(), valid_noisy, mask)
                mse_after = _masked_mse(accepted_state, valid_noisy, mask)
                errors_before = _unknown_errors(current.detach(), label, mask)
                errors_after = _unknown_errors(accepted_state, label, mask)

                accumulators[(timestep, distance)].add(
                    energy_valid=energy_valid,
                    energy_corrupt=energy_corrupt,
                    energy_margin=energy_corrupt - energy_valid,
                    rank_correct=(energy_valid < energy_corrupt),
                    exact_energy_margin=exact_margin,
                    gradient_cosine=cosine,
                    gradient_positive=(cosine > 0),
                    gradient_norm=gradient_norm,
                    proposal_accepted=accepted,
                    proposal_energy_change=proposal_energy - energy.detach(),
                    accepted_mse_improvement=mse_before - mse_after,
                    accepted_hamming_improvement=(errors_before - errors_after),
                )

    records = []
    for timestep in timesteps:
        for distance in distances:
            values = accumulators[(timestep, distance)]
            correct = int(values.tensor("rank_correct").sum())
            n = len(values.tensor("rank_correct"))
            records.append({
                "timestep": timestep,
                "edit_distance": distance,
                "n": n,
                "energy_valid_mean": values.mean("energy_valid"),
                "energy_corrupt_mean": values.mean("energy_corrupt"),
                "energy_margin_mean": values.mean("energy_margin"),
                "energy_margin_median": values.median("energy_margin"),
                "ranking_correct": correct,
                "ranking_accuracy": correct / n,
                "ranking_wilson_95": wilson_interval(correct, n),
                "exact_conflict_margin_mean": values.mean("exact_energy_margin"),
                "exact_conflict_ranking_accuracy": float(
                    (values.tensor("exact_energy_margin") > 0).float().mean()
                ),
                "gradient_corrective_cosine_mean": values.mean("gradient_cosine"),
                "gradient_corrective_positive_rate": values.mean("gradient_positive"),
                "gradient_norm_mean": values.mean("gradient_norm"),
                "proposal_accept_rate": values.mean("proposal_accepted"),
                "proposal_energy_change_mean": values.mean("proposal_energy_change"),
                "accepted_mse_improvement_mean": values.mean("accepted_mse_improvement"),
                "accepted_hamming_improvement_mean": values.mean(
                    "accepted_hamming_improvement"
                ),
            })
    return records


def parse_int_list(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset", choices=("standard-train", "standard-val", "hard-test", "rrn-valid"),
        default="standard-val",
    )
    parser.add_argument("--weights", choices=("raw", "ema", "both"), default="both")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distances", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--timesteps", type=parse_int_list, default=[0, 3, 6, 9])
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.limit < 1 or args.batch_size < 1:
        raise SystemExit("limit and batch size must be positive")
    if any(not 0 <= timestep < 10 for timestep in args.timesteps):
        raise SystemExit("timesteps must be in [0,9]")

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    manifest_seal = verify_manifest_seal(manifest)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != manifest_seal:
        raise RuntimeError("checkpoint and sealed manifest do not match")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = load_dataset(args.dataset)
    if args.start_index < 0 or args.start_index + args.limit > len(dataset):
        raise SystemExit("requested dataset slice is out of range")

    sources = ["raw", "ema"] if args.weights == "both" else [args.weights]
    started = time.time()
    results = []
    device = torch.device("cuda", 0)
    for source in sources:
        model = load_weight_source(payload, manifest, source, device)
        records = evaluate_pairs(
            model,
            dataset,
            start_index=args.start_index,
            limit=args.limit,
            batch_size=args.batch_size,
            distances=args.distances,
            timesteps=args.timesteps,
            seed=args.seed,
        )
        results.append({"weight_source": source, "records": records})
        del model
        torch.cuda.empty_cache()

    summary = {
        "schema": "ired/sudoku-energy-calibration-v1",
        "created_utc": utc_now(),
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "calibration_source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": int(payload["step"]),
            "manifest_seal": manifest_seal,
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "model": {
            "name": manifest.get("model", {}).get("name"),
            "architecture_reference": manifest.get("model", {}).get(
                "architecture_reference", "released_code"
            ),
            "final_conv_kernel": int(manifest.get("model", {}).get("final_conv_kernel", 1)),
        },
        "dataset": args.dataset,
        "slice": {
            "start": args.start_index,
            "end_exclusive": args.start_index + args.limit,
            "n": args.limit,
        },
        "controls": {
            "seed": args.seed,
            "distances": args.distances,
            "timesteps": args.timesteps,
            "shared_noise_within_valid_corrupt_pair": True,
            "corruptions_restricted_to_non_clue_cells": True,
            "official_step_size_for_single_proposal": True,
        },
        "results": results,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
