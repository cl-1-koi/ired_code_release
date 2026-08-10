#!/usr/bin/env python3
"""Token-free checkpoint handoff and artifact closure for the IRED ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runpod
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_seal(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def command(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args, returncode=124, stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )


def ssh_base(pod: dict[str, Any]) -> list[str]:
    return [
        "ssh", "-i", pod["ssh_key"], "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-p", str(pod["port"]),
        f"root@{pod['host']}",
    ]


def ssh(pod: dict[str, Any], shell: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return command([*ssh_base(pod), shell], timeout=timeout)


def rsync_from(pod: dict[str, Any], remote: str, local: Path) -> subprocess.CompletedProcess[str]:
    local.parent.mkdir(parents=True, exist_ok=True)
    return command(
        [
            "rsync", "-a", "--partial", "-e",
            f"ssh -i {pod['ssh_key']} -p {pod['port']} -o BatchMode=yes",
            f"root@{pod['host']}:{remote}", str(local),
        ],
        timeout=1800,
    )


def rsync_to(pod: dict[str, Any], local: Path, remote: str) -> subprocess.CompletedProcess[str]:
    return command(
        [
            "rsync", "-a", "--partial", "-e",
            f"ssh -i {pod['ssh_key']} -p {pod['port']} -o BatchMode=yes",
            str(local), f"root@{pod['host']}:{remote}",
        ],
        timeout=1800,
    )


def validate_landmark(checkpoint: Path, manifest_path: Path, expected_step: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    content = {key: value for key, value in manifest.items() if key != "seal"}
    seal = manifest["seal"]["sha256"]
    if json_seal(content) != seal:
        raise RuntimeError("training manifest seal mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["step"]) != expected_step:
        raise RuntimeError(f"checkpoint step {payload['step']} != {expected_step}")
    if payload["manifest_sha256"] != seal:
        raise RuntimeError("checkpoint does not belong to extension manifest")
    return {
        "schema": "ired/sudoku-landmark-closure-v1",
        "verified_utc": utc_now(),
        "step": expected_step,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "training_manifest_sha256": sha256(manifest_path),
        "training_manifest_seal": seal,
        "source_commit": manifest["code"]["commit"],
    }


def acquire_landmark(config: dict[str, Any], step: int) -> dict[str, Any]:
    trainer = config["trainer"]
    step9 = f"{step:09d}"
    remote_checkpoint = f"{trainer['remote_train_dir']}/checkpoint_step_{step9}.pt"
    remote_manifest = f"{trainer['remote_train_dir']}/training_manifest.json"
    local_dir = Path(config["local_checkpoint_root"]) / f"step_{step9}"
    checkpoint = local_dir / "checkpoint.pt"
    manifest = local_dir / "training_manifest.json"
    closure_path = local_dir / "checkpoint_closure.json"
    if closure_path.is_file():
        return json.loads(closure_path.read_text())

    probe = ssh(
        trainer,
        f"test -s {shlex.quote(remote_checkpoint)} && "
        f"test -s {shlex.quote(remote_manifest)}",
    )
    if probe.returncode:
        return {"step": step, "state": "awaiting_checkpoint"}

    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_part = local_dir / "checkpoint.pt.part"
    manifest_part = local_dir / "training_manifest.json.part"
    for remote, local in (
        (remote_checkpoint, checkpoint_part), (remote_manifest, manifest_part)
    ):
        result = rsync_from(trainer, remote, local)
        if result.returncode:
            return {"step": step, "state": "sync_failed", "error": result.stderr[-500:]}
    os.replace(checkpoint_part, checkpoint)
    os.replace(manifest_part, manifest)
    closure = validate_landmark(checkpoint, manifest, step)
    atomic_json(closure_path, closure)
    return closure


def deploy_landmark(
    evaluator: dict[str, Any], local_root: Path, step: int, closure: dict[str, Any]
) -> dict[str, Any]:
    step9 = f"{step:09d}"
    local_dir = local_root / f"step_{step9}"
    remote_dir = f"{evaluator['remote_input_root']}/step_{step9}"
    checkpoint_sha = closure["checkpoint_sha256"]
    manifest_sha = closure["training_manifest_sha256"]
    check = ssh(
        evaluator,
        f"test -s {shlex.quote(remote_dir + '/checkpoint.pt')} && "
        f"test -s {shlex.quote(remote_dir + '/training_manifest.json')} && "
        f"test \"$(sha256sum {shlex.quote(remote_dir + '/checkpoint.pt')} | cut -d' ' -f1)\" = {checkpoint_sha} && "
        f"test \"$(sha256sum {shlex.quote(remote_dir + '/training_manifest.json')} | cut -d' ' -f1)\" = {manifest_sha}",
    )
    if check.returncode == 0:
        return {"state": "deployed_verified", "step": step}

    made = ssh(evaluator, f"mkdir -p {shlex.quote(remote_dir)}")
    if made.returncode:
        return {"state": "deploy_failed", "error": made.stderr[-500:]}
    pairs = (
        (local_dir / "checkpoint.pt", remote_dir + "/checkpoint.pt.part"),
        (local_dir / "training_manifest.json", remote_dir + "/training_manifest.json.part"),
    )
    for local, remote in pairs:
        result = rsync_to(evaluator, local, remote)
        if result.returncode:
            return {"state": "deploy_failed", "error": result.stderr[-500:]}
    publish = ssh(
        evaluator,
        f"test \"$(sha256sum {shlex.quote(remote_dir + '/checkpoint.pt.part')} | cut -d' ' -f1)\" = {checkpoint_sha} && "
        f"test \"$(sha256sum {shlex.quote(remote_dir + '/training_manifest.json.part')} | cut -d' ' -f1)\" = {manifest_sha} && "
        f"mv {shlex.quote(remote_dir + '/checkpoint.pt.part')} {shlex.quote(remote_dir + '/checkpoint.pt')} && "
        f"mv {shlex.quote(remote_dir + '/training_manifest.json.part')} {shlex.quote(remote_dir + '/training_manifest.json')}",
    )
    if publish.returncode:
        return {"state": "deploy_failed", "error": publish.stderr[-500:]}
    return {"state": "deployed_verified", "step": step}


def trainer_probe(config: dict[str, Any]) -> dict[str, Any]:
    pod = config["trainer"]
    root = pod["remote_train_dir"]
    shell = (
        f"if tmux has-session -t {shlex.quote(pod['session'])} 2>/dev/null; "
        "then echo session=running; else echo session=absent; fi; "
        f"tail -n 1 {shlex.quote(root + '/telemetry.jsonl')} 2>/dev/null || true; "
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader 2>/dev/null || true"
    )
    result = ssh(pod, shell)
    return {
        "returncode": result.returncode,
        "lines": result.stdout.splitlines(),
        "error": result.stderr[-500:] if result.returncode else "",
    }


def safe_relative(remote_file: str, remote_root: str) -> Path | None:
    prefix = remote_root.rstrip("/") + "/"
    if not remote_file.startswith(prefix):
        return None
    relative = Path(remote_file[len(prefix):])
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return relative


def verify_result(local_dir: Path, marker_name: str) -> dict[str, Any]:
    marker = local_dir / marker_name
    payload = json.loads(marker.read_text())
    if marker_name == "failure.json":
        closure = {
            "verified": True, "state": "failed", "verified_utc": utc_now(),
            "failure_sha256": sha256(marker),
        }
        atomic_json(local_dir / "failure_closure.json", closure)
        return closure
    checked = 0
    for item in payload.get("files", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe artifact path in completion manifest")
        path = local_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing result artifact {relative}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"result artifact mismatch {relative}")
        checked += 1
    closure = {
        "verified": True, "state": "complete", "verified_utc": utc_now(),
        "files": checked, "completion_sha256": sha256(marker),
    }
    atomic_json(local_dir / "artifact_closure.json", closure)
    return closure


def collect_results(config: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    remote_root = evaluator["remote_result_root"].rstrip("/")
    local_root = Path(config["local_result_root"]) / evaluator["pod_id"]
    find = ssh(
        evaluator,
        f"find {shlex.quote(remote_root)} -type f "
        "\\( -name completion.json -o -name failure.json \\) -print 2>/dev/null | sort",
    )
    report: dict[str, Any] = {
        "returncode": find.returncode, "discovered_markers": 0,
        "newly_verified": [], "error": find.stderr[-500:] if find.returncode else "",
    }
    if find.returncode:
        return report
    for remote_marker in find.stdout.splitlines():
        relative = safe_relative(remote_marker, remote_root)
        if relative is None:
            continue
        report["discovered_markers"] += 1
        local_dir = local_root / relative.parent
        closure_name = (
            "artifact_closure.json" if relative.name == "completion.json"
            else "failure_closure.json"
        )
        if (local_dir / closure_name).is_file():
            continue
        result = rsync_from(evaluator, str(Path(remote_marker).parent) + "/", local_dir)
        if result.returncode:
            report["newly_verified"].append(
                {"path": str(relative.parent), "state": "sync_failed", "error": result.stderr[-500:]}
            )
            continue
        try:
            closure = verify_result(local_dir, relative.name)
        except Exception as exc:  # preserve the bad copy for diagnosis
            closure = {"state": "verification_failed", "error": str(exc)}
        report["newly_verified"].append({"path": str(relative.parent), **closure})
    return report


def supervise_fidelity_arm(config: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    remote = arm["remote_output"].rstrip("/")
    shell = (
        f"if test -s {shlex.quote(remote + '/completion.json')}; then echo state=complete; "
        f"elif test -s {shlex.quote(remote + '/failure.json')}; then echo state=failed; "
        f"elif tmux has-session -t {shlex.quote(arm['session'])} 2>/dev/null; "
        "then echo state=running; else echo state=idle_or_missing; fi; "
        f"tail -n 1 {shlex.quote(remote + '/train/telemetry.jsonl')} 2>/dev/null || true; "
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader 2>/dev/null || true"
    )
    probe = ssh(arm, shell)
    report: dict[str, Any] = {
        "returncode": probe.returncode,
        "lines": probe.stdout.splitlines(),
        "error": probe.stderr[-500:] if probe.returncode else "",
    }
    state = "unreachable"
    if probe.returncode == 0 and report["lines"]:
        state = report["lines"][0].removeprefix("state=")
    report["state"] = state
    if state not in ("complete", "failed"):
        return report
    local = Path(config["local_fidelity_root"]) / arm["pod_id"] / arm["name"]
    closure_name = "artifact_closure.json" if state == "complete" else "failure_closure.json"
    if (local / closure_name).is_file():
        report["artifact_closure"] = json.loads((local / closure_name).read_text())
        return report
    result = rsync_from(arm, remote + "/", local)
    if result.returncode:
        report["artifact_closure"] = {
            "state": "sync_failed", "error": result.stderr[-500:]
        }
        return report
    try:
        report["artifact_closure"] = verify_result(
            local, "completion.json" if state == "complete" else "failure.json"
        )
    except Exception as exc:
        report["artifact_closure"] = {
            "state": "verification_failed", "error": str(exc)
        }
    return report


def cycle(config: dict[str, Any]) -> dict[str, Any]:
    account = {pod["id"]: pod for pod in runpod.get_pods()}
    local_checkpoint_root = Path(config["local_checkpoint_root"])
    landmarks = []
    for step in config["landmarks"]:
        try:
            closure = acquire_landmark(config, int(step))
            deployed = {}
            if "checkpoint_sha256" in closure:
                for evaluator in config["evaluators"]:
                    deployed[evaluator["pod_id"]] = deploy_landmark(
                        evaluator, local_checkpoint_root, int(step), closure
                    )
            landmarks.append({"step": step, "closure": closure, "deployed": deployed})
        except Exception as exc:
            landmarks.append({"step": step, "state": "watcher_error", "error": str(exc)})
    evaluations = {
        evaluator["pod_id"]: collect_results(config, evaluator)
        for evaluator in config["evaluators"]
    }
    fidelity_arms = {
        arm["pod_id"]: supervise_fidelity_arm(config, arm)
        for arm in config.get("fidelity_arms", [])
    }
    return {
        "schema": "ired/sudoku-program-supervisor-v1",
        "updated_utc": utc_now(),
        "active_pod_ids": sorted(
            pod_id for pod_id, pod in account.items()
            if pod.get("desiredStatus") == "RUNNING"
        ),
        "trainer": trainer_probe(config),
        "landmarks": landmarks,
        "evaluations": evaluations,
        "fidelity_arms": fidelity_arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        config = json.loads(args.config.read_text())
        atomic_json(args.status, cycle(config))
        if args.once:
            break
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
