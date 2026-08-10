#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 COMMIT SOURCE_DIR DATA_DIR CHECKPOINT MANIFEST OUTPUT_DIR DATASET INNER_STEPS SEED GPU" >&2
  exit 2
fi

commit="$1"
source_dir="$2"
data_dir="$3"
checkpoint="$4"
manifest="$5"
output_dir="$6"
dataset="$7"
inner_steps="$8"
seed="$9"
gpu="${10}"

cd "$source_dir"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
test -s "$checkpoint"
test -s "$manifest"
test ! -e "$output_dir"
export IRED_DATA_ROOT="$data_dir"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$output_dir"

python - "$output_dir/evaluation_manifest.json" "$commit" "$checkpoint" "$manifest" "$dataset" "$inner_steps" "$seed" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone

path, commit, checkpoint, manifest, dataset, inner_steps, seed = sys.argv[1:]
def sha256(name):
    digest = hashlib.sha256()
    with open(name, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
payload = {
    "schema": "ired/sudoku-evaluation-manifest-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "evaluator_source_commit": commit,
    "checkpoint": os.path.abspath(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "training_manifest": os.path.abspath(manifest),
    "training_manifest_sha256": sha256(manifest),
    "dataset": dataset,
    "innerloop_steps": int(inner_steps),
    "seed": int(seed),
    "batch_size": 16,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

on_error() {
  code=$?
  printf '{"schema":"ired/sudoku-evaluation-failure-v1","exit_code":%d,"source_commit":"%s","dataset":"%s","innerloop_steps":%d,"seed":%d}\n' \
    "$code" "$commit" "$dataset" "$inner_steps" "$seed" > "$output_dir/failure.json"
  exit "$code"
}
trap on_error ERR

python -m repro.evaluate_sudoku \
  --checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --output-dir "$output_dir" \
  --dataset "$dataset" \
  --batch-size 16 \
  --innerloop-steps "$inner_steps" \
  --seed "$seed" \
  2>&1 | tee "$output_dir/console.log"

python - "$output_dir" "$commit" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
root, commit = sys.argv[1:]
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
files = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if os.path.isfile(path) and name not in ("completion.json", "failure.json"):
        files.append({"path": name, "bytes": os.path.getsize(path), "sha256": sha256(path)})
summary = json.load(open(os.path.join(root, "summary.json"), encoding="utf-8"))
payload = {
    "schema": "ired/sudoku-evaluation-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "evaluator_source_commit": commit,
    "checkpoint_step": summary["checkpoint"]["step"],
    "dataset": summary["dataset"],
    "innerloop_steps": summary["sampling"]["innerloop_steps"],
    "files": files,
}
with open(os.path.join(root, "completion.json"), "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
