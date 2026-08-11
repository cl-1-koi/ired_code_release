"""Evaluate an IRED Sudoku checkpoint with published and strict metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA

from repro.sudoku_metrics import sudoku_batch_metrics
from repro.train_sudoku import build_model, sha256_file
from sat_dataset import SudokuDataset, SudokuRRNDataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("standard-train", "standard-val", "hard-test",
                                               "rrn-valid"),
                        default="standard-val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--innerloop-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != manifest["seal"]["sha256"]:
        raise RuntimeError("checkpoint and sealed manifest do not match")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = load_dataset(args.dataset)
    begin = args.start_index
    end = len(dataset) if args.limit is None else min(len(dataset), begin + args.limit)
    if not 0 <= begin < end <= len(dataset):
        raise SystemExit("requested evaluation slice is empty or out of range")

    device = torch.device("cuda", 0)
    final_conv_kernel = int(manifest.get("model", {}).get("final_conv_kernel", 1))
    diffusion = build_model(
        dataset.inp_dim, dataset.out_dim, args.innerloop_steps,
        final_conv_kernel=final_conv_kernel,
    ).to(device)
    diffusion.out_dim = dataset.out_dim
    diffusion.out_shape = (dataset.out_dim,)
    ema = EMA(diffusion, beta=0.995, update_every=10).to(device)
    ema.load_state_dict(payload["ema"])
    model = ema.ema_model
    model.out_dim = dataset.out_dim
    model.out_shape = (dataset.out_dim,)
    model.eval()

    forward_calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal forward_calls
        forward_calls += 1

    hook = model.model.ebm.register_forward_hook(count_forward)
    board_file = output / "boards.jsonl"
    sums = {"cell_accuracy": 0.0, "unknown_cell_accuracy": 0.0,
            "published_satnet_constraint_fraction": 0.0}
    strict_count = clue_count = exact_count = solved_count = 0
    started = time.time()
    with board_file.open("w", encoding="utf-8") as handle:
        for offset in range(begin, end, args.batch_size):
            batch_end = min(end, offset + args.batch_size)
            rows = [dataset[index] for index in range(offset, batch_end)]
            inp = torch.stack([row[0] for row in rows]).to(device)
            label = torch.stack([row[1] for row in rows]).to(device)
            mask = torch.stack([row[2] for row in rows]).float().to(device)
            prediction = model.sample(inp, label, mask, batch_size=len(rows))
            metrics = sudoku_batch_metrics(prediction, label, mask)
            solved = metrics.strict_valid & metrics.clue_consistent
            strict_count += int(metrics.strict_valid.sum())
            clue_count += int(metrics.clue_consistent.sum())
            exact_count += int(metrics.exact_solution.sum())
            solved_count += int(solved.sum())
            sums["cell_accuracy"] += float(metrics.cell_accuracy.sum())
            sums["unknown_cell_accuracy"] += float(metrics.unknown_cell_accuracy.sum())
            sums["published_satnet_constraint_fraction"] += float(
                metrics.satnet_constraint_fraction.sum())
            for row_index, dataset_index in enumerate(range(offset, batch_end)):
                record = {
                    "dataset_index": dataset_index,
                    "strict_valid": bool(metrics.strict_valid[row_index]),
                    "clue_consistent": bool(metrics.clue_consistent[row_index]),
                    "valid_solution": bool(solved[row_index]),
                    "exact_reference_solution": bool(metrics.exact_solution[row_index]),
                    "cell_accuracy": float(metrics.cell_accuracy[row_index]),
                    "unknown_cell_accuracy": float(metrics.unknown_cell_accuracy[row_index]),
                    "published_satnet_constraint_fraction":
                        float(metrics.satnet_constraint_fraction[row_index]),
                    "predicted_digits": metrics.predicted_digits[row_index].cpu().tolist(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    hook.remove()

    n = end - begin
    summary = {
        "schema": "ired/sudoku-evaluation-v1",
        "created_utc": utc_now(),
        "dataset": args.dataset,
        "slice": {"start": begin, "end_exclusive": end, "n": n},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                       "step": int(payload["step"]),
                       "manifest_sha256": payload["manifest_sha256"]},
        "model": {"final_conv_kernel": final_conv_kernel,
                  "architecture_reference": manifest.get("model", {}).get(
                      "architecture_reference", "released_code")},
        "sampling": {"seed": args.seed, "batch_size": args.batch_size,
                     "diffusion_landscapes": 10, "innerloop_steps": args.innerloop_steps,
                     "observed_ebm_forward_calls": forward_calls,
                     "expected_ebm_forward_calls_per_batch": 10 * (1 + 2 * args.innerloop_steps)},
        "metrics": {
            "valid_solution_count": solved_count,
            "valid_solution_rate": solved_count / n,
            "valid_solution_wilson_95": wilson_interval(solved_count, n),
            "strict_valid_count": strict_count,
            "strict_valid_rate": strict_count / n,
            "clue_consistent_count": clue_count,
            "clue_consistency_rate": clue_count / n,
            "exact_reference_count": exact_count,
            "exact_reference_rate": exact_count / n,
            **{key: value / n for key, value in sums.items()},
        },
        "wall_seconds": time.time() - started,
        "boards_sha256": sha256_file(board_file),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
