"""Sealed, resumable IRED Sudoku training using the released loss/model path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.optim import Adam

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import DiffusionWrapper, SudokuEBM
from sat_dataset import SudokuDataset


UPSTREAM_COMMIT = "3d74b85fab7fcf5e28aaf15e9ed3bf51c1a1d545"
PAPER_SHA256 = "f8348abcc56307cd6cd502c4786cb0dab9dc11feaa5bc2d012661bca2051373c"
EXPECTED_DATA = {
    "sudoku/features.pt": "8349ff2a210f6a5bffc052dbddbde6b3461ef893122d19b375fc5dd2f444fbef",
    "sudoku/labels.pt": "596fa31210b143fed37252d2694caeb297e18998a8127d2c9a58897552da5ec8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StatefulPermutationBatcher:
    """Match shuffled epoch batching while preserving exact resume state."""

    def __init__(self, n: int, batch_size: int, seed: int):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.permutation: torch.Tensor | None = None
        self.offset = 0

    def next(self) -> torch.Tensor:
        if self.permutation is None or self.offset >= self.n:
            self.permutation = torch.randperm(self.n, generator=self.generator)
            self.offset = 0
        end = min(self.offset + self.batch_size, self.n)
        batch = self.permutation[self.offset:end]
        self.offset = end
        return batch

    def state_dict(self) -> dict:
        return {
            "n": self.n,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "offset": self.offset,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["n"] != self.n or state["batch_size"] != self.batch_size:
            raise RuntimeError("batcher configuration mismatch")
        self.generator.set_state(state["generator_state"].cpu())
        self.permutation = (None if state["permutation"] is None
                            else state["permutation"].cpu())
        self.offset = int(state["offset"])


def build_model(
    inp_dim: int, out_dim: int, innerloop_steps: int = 20,
    final_conv_kernel: int = 1,
    sudoku_negative_opt_steps: int = 0,
):
    if final_conv_kernel not in (1, 3):
        raise ValueError("final_conv_kernel must be 1 or 3")
    energy = SudokuEBM(inp_dim=inp_dim, out_dim=out_dim)
    if final_conv_kernel == 3:
        # The released SudokuEBM uses a 1x1 output convolution, while Table 10
        # of the paper specifies 3x3.  Keep released-code behavior as the
        # default and expose the paper-declared architecture as a sealed arm.
        energy.conv5 = torch.nn.Conv2d(384, 9, 3, padding=1)
    diffusion = GaussianDiffusion1D(
        DiffusionWrapper(energy),
        seq_length=32,
        objective="pred_noise",
        timesteps=10,
        sampling_timesteps=10,
        supervise_energy_landscape=True,
        use_innerloop_opt=True,
        show_inference_tqdm=False,
        sudoku=True,
        innerloop_steps=innerloop_steps,
        sudoku_negative_opt_steps=sudoku_negative_opt_steps,
    )
    return diffusion


def dataset_hashes(data_root: Path) -> dict:
    values = {}
    for relative, expected in EXPECTED_DATA.items():
        path = data_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"dataset hash mismatch for {relative}: {actual}")
        values[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    return values


def source_hashes(repo: Path) -> dict:
    names = [
        "models.py",
        "sat_dataset.py",
        "diffusion_lib/denoising_diffusion_pytorch_1d.py",
        "repro/train_sudoku.py",
        "repro/evaluate_sudoku.py",
        "repro/sudoku_metrics.py",
        "ops/run_ired_sudoku_ladder.sh",
        "ops/run_ired_sudoku_extension.sh",
        "ops/run_ired_sudoku_eval.sh",
        "ops/run_ired_sudoku_eval_queue.sh",
        "ops/run_ired_sudoku_fidelity_arm.sh",
        "requirements-repro.txt",
        "IRED_SUDOKU_REPRO_SPEC_20260809.md",
    ]
    return {name: sha256_file(repo / name) for name in names if (repo / name).is_file()}


def make_manifest(args, dataset, model, repo: Path, data_root: Path) -> dict:
    status = git("status", "--porcelain")
    content = {
        "schema": "ired/sudoku-training-manifest-v1",
        "created_utc": utc_now(),
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "upstream_commit": UPSTREAM_COMMIT,
            "dirty": bool(status),
            "dirty_paths": status.splitlines(),
            "source_hashes": source_hashes(repo),
        },
        "paper": {"title": "Learning Iterative Reasoning through Energy Diffusion",
                  "sha256": PAPER_SHA256, "declared_sudoku_steps": 50000},
        "data": {"root": str(data_root), "files": dataset_hashes(data_root),
                 "train_examples": len(dataset), "standard_validation_examples": 1000,
                 "hard_test_examples": 18000},
        "model": {
            "name": "released SudokuEBM CNN",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "final_conv_kernel": args.final_conv_kernel,
            "architecture_reference": (
                "released_code" if args.final_conv_kernel == 1
                else "paper_table_10"
            ),
            "diffusion_landscapes": 10,
            "inner_gradient_steps_per_landscape": 20,
            "contrastive_cell_corruption_probability": 0.05,
            "contrastive_negative_opt_steps": args.sudoku_negative_opt_steps,
            "contrastive_negative_variant": (
                "released_random_digit_corruption"
                if args.sudoku_negative_opt_steps == 0
                else "random_digit_corruption_then_energy_refinement"
            ),
        },
        "training": {
            "seed": args.seed, "batch_size": args.batch_size,
            "learning_rate": 1e-4, "adam_betas": [0.9, 0.99],
            "gradient_clip": 1.0, "ema_beta": 0.995, "ema_update_every": 10,
            "target_steps": args.target_steps, "precision": "float32",
            "stateful_shuffled_epoch_batches": True,
        },
        "execution": {
            "health_gate_step": 200,
            "planned_handoff": "resume the same sealed trajectory to target_steps after health gate",
            "output_dir": str(Path(args.output_dir).resolve()),
        },
        "environment": {
            "python": os.sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    return {**content, "seal": {"sha256": sha256_json(content)}}


def verify_manifest(manifest: dict, args) -> str:
    seal = verify_manifest_seal(manifest)
    if manifest["code"]["commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError("source commit changed")
    if manifest["code"]["dirty"] or git("status", "--porcelain"):
        raise RuntimeError("source worktree is dirty")
    training = manifest["training"]
    if (training["seed"], training["batch_size"], training["target_steps"]) != (
        args.seed, args.batch_size, args.target_steps
    ):
        raise RuntimeError("training contract changed")
    manifest_negative_steps = int(
        manifest.get("model", {}).get("contrastive_negative_opt_steps", 0)
    )
    if manifest_negative_steps != args.sudoku_negative_opt_steps:
        raise RuntimeError("contrastive negative refinement changed")
    return seal


def verify_manifest_seal(manifest: dict) -> str:
    seal = manifest["seal"]["sha256"]
    if sha256_json({key: value for key, value in manifest.items() if key != "seal"}) != seal:
        raise RuntimeError("manifest seal mismatch")
    return seal


def make_extension_manifest(
    args, dataset, model, repo: Path, data_root: Path,
    parent_manifest: dict, parent_manifest_path: Path,
    parent_checkpoint_path: Path, parent_checkpoint: dict,
) -> dict:
    parent_seal = verify_manifest_seal(parent_manifest)
    parent_training = parent_manifest["training"]
    if (parent_training["seed"], parent_training["batch_size"]) != (
        args.seed, args.batch_size
    ):
        raise RuntimeError("parent seed or batch size changed")
    parent_step = int(parent_checkpoint["step"])
    if parent_checkpoint["manifest_sha256"] != parent_seal:
        raise RuntimeError("parent checkpoint does not belong to parent manifest")
    if args.target_steps <= parent_step:
        raise RuntimeError("extension target must exceed the parent step")

    manifest = make_manifest(args, dataset, model, repo, data_root)
    if manifest["code"]["dirty"]:
        raise RuntimeError("source worktree is dirty")
    content = {key: value for key, value in manifest.items() if key != "seal"}
    content["lineage"] = {
        "kind": "exact_resume_target_extension",
        "reason": (
            "continue the exact 50k trajectory to the released code's 1.3M "
            "default after the stronger strict whole-board gate missed"
        ),
        "parent_source_commit": parent_manifest["code"]["commit"],
        "parent_manifest": str(parent_manifest_path.resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_manifest_seal": parent_seal,
        "parent_checkpoint": str(parent_checkpoint_path.resolve()),
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint_path),
        "parent_step": parent_step,
        "parent_target_steps": parent_training["target_steps"],
    }
    content["execution"]["planned_handoff"] = (
        "evaluate standard and hard sets over inference steps "
        "{1,5,10,15,20,40,80} at 100k, 300k, 1M, and 1.3M"
    )
    if args.sudoku_negative_opt_steps > 0:
        content["lineage"]["kind"] = "warm_start_search_state_negative_intervention"
        content["lineage"]["reason"] = (
            "apply the released continuous-task hard-negative refinement to "
            "Sudoku contrastive corruptions; frozen diagnostics found good "
            "pairwise energy ordering but weak sampler-state geometry"
        )
        content["execution"]["planned_handoff"] = (
            "evaluate the held-out energy/gradient calibration and fixed "
            "standard/hard strict metrics before any further continuation"
        )
    return {**content, "seal": {"sha256": sha256_json(content)}}


def save_checkpoint(path: Path, *, step: int, model, optimizer, ema, batcher,
                    manifest_sha256: str) -> None:
    payload = {
        "schema": "ired/sudoku-checkpoint-v1", "step": step,
        "manifest_sha256": manifest_sha256, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "ema": ema.state_dict(),
        "batcher": batcher.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-steps", type=int, default=50000)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-conv-kernel", type=int, choices=(1, 3), default=1)
    parser.add_argument("--sudoku-negative-opt-steps", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--parent-manifest", default=None,
        help="signed parent manifest used only to start a new target-extension lineage",
    )
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    args = parser.parse_args()
    if not 0 < args.stop_after_step <= args.target_steps:
        raise SystemExit("stop-after-step must be in 1..target-steps")
    if args.sudoku_negative_opt_steps < 0:
        raise SystemExit("sudoku-negative-opt-steps must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    if repo in output.parents or output == repo:
        raise SystemExit("output directory must be outside the source tree")
    data_root = Path(os.environ.get("IRED_DATA_ROOT", repo / "data")).resolve()
    dataset = SudokuDataset("sudoku", split="train")
    device = torch.device("cuda", 0)
    model = build_model(
        dataset.inp_dim, dataset.out_dim,
        final_conv_kernel=args.final_conv_kernel,
        sudoku_negative_opt_steps=args.sudoku_negative_opt_steps,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    ema = EMA(model, beta=0.995, update_every=10).to(device)
    batcher = StatefulPermutationBatcher(len(dataset), args.batch_size, args.seed + 1)

    manifest_path = output / "training_manifest.json"
    if args.resume:
        # Keep process-global and batching RNG tensors on CPU. Optimizer/model
        # loaders copy their own tensors onto the CUDA parameters as needed.
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected_manifest_seal = verify_manifest(manifest, args)
        elif args.parent_manifest:
            parent_manifest_path = Path(args.parent_manifest).resolve()
            parent_manifest = json.loads(parent_manifest_path.read_text())
            manifest = make_extension_manifest(
                args, dataset, model, repo, data_root,
                parent_manifest, parent_manifest_path,
                Path(args.resume).resolve(), payload,
            )
            atomic_json(manifest_path, manifest)
            expected_manifest_seal = parent_manifest["seal"]["sha256"]
        else:
            raise RuntimeError(
                "resume into a new output directory requires --parent-manifest"
            )
        if payload["manifest_sha256"] != expected_manifest_seal:
            raise RuntimeError("checkpoint manifest mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        ema.load_state_dict(payload["ema"]); batcher.load_state_dict(payload["batcher"])
        random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
        step = int(payload["step"])
    else:
        if manifest_path.exists():
            raise RuntimeError("refusing to overwrite an existing run")
        manifest = make_manifest(args, dataset, model, repo, data_root)
        if manifest["code"]["dirty"]:
            raise RuntimeError("source worktree is dirty")
        atomic_json(manifest_path, manifest)
        step = 0

    telemetry = output / "telemetry.jsonl"
    started = time.time(); window_started = started; window_examples = 0; window_steps = 0
    while step < args.stop_after_step:
        indices = batcher.next()
        inp = ((dataset.features[indices].reshape(-1, 729) - 0.5) * 2).to(device)
        label = ((dataset.labels[indices].reshape(-1, 729) - 0.5) * 2).to(device)
        mask = dataset.cond_entry[indices].reshape(-1, 729).float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = model(inp, label, mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); ema.update(); step += 1; window_examples += len(indices); window_steps += 1
        if step % args.log_every == 0 or step == args.stop_after_step:
            now = time.time(); elapsed = max(now - window_started, 1e-9)
            free, total = torch.cuda.mem_get_info(device)
            record = {
                "utc": utc_now(), "kind": "train", "step": step,
                "loss": float(loss.detach()), "loss_denoise": float(components[0]),
                "loss_energy": float(components[1]), "loss_opt": float(components[2]),
                "grad_norm": float(grad_norm),
                "steps_per_second": window_steps / elapsed,
                "examples_per_second": window_examples / elapsed,
                "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_free_bytes": free, "gpu_total_bytes": total,
            }
            with telemetry.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
            window_started = now; window_examples = 0; window_steps = 0
        if step % args.checkpoint_every == 0 or step == args.stop_after_step:
            path = output / f"checkpoint_step_{step:09d}.pt"
            save_checkpoint(path, step=step, model=model, optimizer=optimizer, ema=ema,
                            batcher=batcher, manifest_sha256=manifest["seal"]["sha256"])
            shutil.copyfile(path, output / "latest.pt")
        if args.time_budget_seconds and time.time() - started >= args.time_budget_seconds:
            break

    # A time-budget exit may not coincide with the checkpoint cadence. Always
    # preserve the exact state reached by this invocation.
    latest = output / "latest.pt"
    latest_step = None
    if latest.exists():
        latest_step = int(torch.load(latest, map_location="cpu", weights_only=False)["step"])
    if latest_step != step:
        path = output / f"checkpoint_step_{step:09d}.pt"
        save_checkpoint(path, step=step, model=model, optimizer=optimizer, ema=ema,
                        batcher=batcher, manifest_sha256=manifest["seal"]["sha256"])
        shutil.copyfile(path, latest)

    summary = {
        "schema": "ired/sudoku-training-invocation-v1", "created_utc": utc_now(),
        "step": step, "requested_stop_after_step": args.stop_after_step,
        "target_steps": args.target_steps, "reached_requested_stop": step >= args.stop_after_step,
        "reached_target": step >= args.target_steps, "wall_seconds": time.time() - started,
        "latest_checkpoint": str(latest),
        "latest_checkpoint_sha256": sha256_file(latest),
        "training_manifest_sha256": sha256_file(manifest_path),
        "training_manifest_seal": manifest["seal"]["sha256"],
    }
    atomic_json(output / f"training_summary_step_{step:09d}.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
