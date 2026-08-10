#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 COMMIT SOURCE_DIR DATA_DIR PARENT_TRAIN_DIR OUTPUT_DIR TARGET_STEPS GPU" >&2
  exit 2
fi

commit="$1"
source_dir="$2"
data_dir="$3"
parent_train_dir="$4"
output_dir="$5"
target_steps="$6"
gpu="$7"

cd "$source_dir"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
test -s "$parent_train_dir/latest.pt"
test -s "$parent_train_dir/training_manifest.json"
test ! -e "$output_dir"
export IRED_DATA_ROOT="$data_dir"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"

mkdir -p "$(dirname "$output_dir")" "$output_dir/logs"
failure="$output_dir/failure.json"
state="$output_dir/run_state.json"

write_state() {
  local run_state="$1"
  local latest_step="$2"
  python - "$state" "$run_state" "$latest_step" "$target_steps" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, state, latest, target = sys.argv[1:]
payload = {
    "schema": "ired/sudoku-extension-state-v1",
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "latest_step": int(latest),
    "target_step": int(target),
    "queued_next_experiment": (
        "standard/hard inference-step sweep {1,5,10,15,20,40,80} at the next "
        "landmark checkpoint"
    ),
    "retain_or_terminate_rationale": (
        "retain the high-CPU L40S through the exact 1.3M continuation and "
        "declared checkpoint evaluations"
    ),
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
}

on_error() {
  code=$?
  write_state failed 50000
  printf '{"schema":"ired/sudoku-extension-failure-v1","exit_code":%d,"source_commit":"%s","parent_step":50000,"target_step":%d}\n' \
    "$code" "$commit" "$target_steps" > "$failure"
  exit "$code"
}
trap on_error ERR

write_state running 50000
python -m repro.train_sudoku \
  --output-dir "$output_dir/train" \
  --target-steps "$target_steps" \
  --stop-after-step "$target_steps" \
  --batch-size 64 \
  --checkpoint-every 50000 \
  --log-every 100 \
  --seed 42 \
  --resume "$parent_train_dir/latest.pt" \
  --parent-manifest "$parent_train_dir/training_manifest.json" \
  2>&1 | tee "$output_dir/logs/train_extension.log"

test -s "$output_dir/train/training_summary_step_$(printf '%09d' "$target_steps").json"
write_state complete "$target_steps"
sha256sum \
  "$output_dir/train/training_manifest.json" \
  "$output_dir/train/latest.pt" \
  "$output_dir/train/training_summary_step_$(printf '%09d' "$target_steps").json" \
  > "$output_dir/completion.sha256"
