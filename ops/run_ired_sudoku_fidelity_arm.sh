#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 COMMIT SOURCE_DIR DATA_DIR OUTPUT_DIR SEED FINAL_CONV_KERNEL GPU LABEL" >&2
  exit 2
fi

commit="$1"
source_dir="$2"
data_dir="$3"
output_dir="$4"
seed="$5"
final_conv_kernel="$6"
gpu="$7"
label="$8"

cd "$source_dir"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
test "$final_conv_kernel" = 1 || test "$final_conv_kernel" = 3
test ! -e "$output_dir"
export IRED_DATA_ROOT="$data_dir"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$output_dir/logs"

on_error() {
  code=$?
  printf '{"schema":"ired/sudoku-fidelity-arm-failure-v1","exit_code":%d,"source_commit":"%s","label":"%s","seed":%d,"final_conv_kernel":%d}\n' \
    "$code" "$commit" "$label" "$seed" "$final_conv_kernel" > "$output_dir/failure.json"
  exit "$code"
}
trap on_error ERR

python -m repro.train_sudoku \
  --output-dir "$output_dir/train" \
  --target-steps 50000 \
  --stop-after-step 50000 \
  --batch-size 64 \
  --checkpoint-every 50000 \
  --log-every 100 \
  --seed "$seed" \
  --final-conv-kernel "$final_conv_kernel" \
  2>&1 | tee "$output_dir/logs/train.log"

for dataset_seed in standard-val:314159 hard-test:271828; do
  dataset=${dataset_seed%%:*}
  eval_seed=${dataset_seed##*:}
  eval_dir="$output_dir/eval_${dataset}"
  python -m repro.evaluate_sudoku \
    --checkpoint "$output_dir/train/latest.pt" \
    --manifest "$output_dir/train/training_manifest.json" \
    --output-dir "$eval_dir" \
    --dataset "$dataset" \
    --batch-size 16 \
    --innerloop-steps 20 \
    --seed "$eval_seed" \
    2>&1 | tee "$output_dir/logs/eval_${dataset}.log"
done

python - "$output_dir" "$commit" "$label" "$seed" "$final_conv_kernel" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone

root, commit, label, seed, kernel = sys.argv[1:]
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
names = [
    "train/training_manifest.json", "train/latest.pt",
    "train/training_summary_step_000050000.json",
    "eval_standard-val/summary.json", "eval_standard-val/boards.jsonl",
    "eval_hard-test/summary.json", "eval_hard-test/boards.jsonl",
]
files = []
for name in names:
    path = os.path.join(root, name)
    files.append({"path": name, "bytes": os.path.getsize(path), "sha256": digest(path)})
payload = {
    "schema": "ired/sudoku-fidelity-arm-completion-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": commit, "label": label, "seed": int(seed),
    "final_conv_kernel": int(kernel), "files": files,
}
with open(os.path.join(root, "completion.json"), "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
