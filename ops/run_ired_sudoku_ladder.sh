#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory required}
DATA_DIR=${2:?data directory required}
OUTPUT_DIR=${3:?output directory required}

cd "$SOURCE_DIR"
export IRED_DATA_ROOT="$DATA_DIR"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/eval_health" \
  "$OUTPUT_DIR/eval_standard" "$OUTPUT_DIR/eval_hard"

write_state() {
  local phase=$1
  local state=$2
  local queued=$3
  local rationale=$4
  python - "$OUTPUT_DIR/run_state.json" "$phase" "$state" "$queued" "$rationale" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, phase, state, queued, rationale = sys.argv[1:]
payload = {
    "schema": "ired/run-state-v1",
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "state": state,
    "queued_next_experiment": queued,
    "retain_or_terminate_rationale": rationale,
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
}

on_error() {
  local code=$?
  write_state "${CURRENT_PHASE:-unknown}" "failed" "diagnose declared arm; do not mutate it" \
    "retain while failure artifacts are synchronized and a compatible follow-up is prepared"
  exit "$code"
}
trap on_error ERR

CURRENT_PHASE=health_train_200
write_state "$CURRENT_PHASE" running health_eval_16 \
  "healthy pod has a declared same-arm handoff through 50k"
python -m repro.train_sudoku \
  --output-dir "$OUTPUT_DIR/train" \
  --target-steps 50000 \
  --stop-after-step 200 \
  --batch-size 64 \
  --checkpoint-every 200 \
  --log-every 20 \
  --seed 42 \
  2>&1 | tee "$OUTPUT_DIR/logs/health_train.log"

python - "$OUTPUT_DIR/train/telemetry.jsonl" <<'PY'
import json, math, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert rows and rows[-1]["step"] == 200
for row in rows:
    for key in ("loss", "loss_denoise", "loss_energy", "grad_norm", "steps_per_second"):
        assert math.isfinite(row[key]), (key, row)
assert rows[-1]["steps_per_second"] > 0
PY

CURRENT_PHASE=health_eval_16
write_state "$CURRENT_PHASE" running resume_same_trajectory_to_50000 \
  "R0 accuracy is diagnostic; finite training and a functioning sampler trigger R1"
python -m repro.evaluate_sudoku \
  --checkpoint "$OUTPUT_DIR/train/latest.pt" \
  --manifest "$OUTPUT_DIR/train/training_manifest.json" \
  --output-dir "$OUTPUT_DIR/eval_health" \
  --dataset standard-val \
  --limit 16 \
  --batch-size 16 \
  --innerloop-steps 20 \
  --seed 314159 \
  2>&1 | tee "$OUTPUT_DIR/logs/health_eval.log"
test -s "$OUTPUT_DIR/eval_health/summary.json"

CURRENT_PHASE=train_50000
write_state "$CURRENT_PHASE" running standard_eval_1000 \
  "continuing the sealed R0 state avoids restart delay and preserves the declared arm"
python -m repro.train_sudoku \
  --output-dir "$OUTPUT_DIR/train" \
  --target-steps 50000 \
  --stop-after-step 50000 \
  --batch-size 64 \
  --checkpoint-every 5000 \
  --log-every 100 \
  --seed 42 \
  --resume "$OUTPUT_DIR/train/latest.pt" \
  2>&1 | tee "$OUTPUT_DIR/logs/train_50000.log"

CURRENT_PHASE=standard_eval_1000
write_state "$CURRENT_PHASE" running hard_eval_18000 \
  "same checkpoint and sampler are already resident; immediate evaluation is fastest"
python -m repro.evaluate_sudoku \
  --checkpoint "$OUTPUT_DIR/train/latest.pt" \
  --manifest "$OUTPUT_DIR/train/training_manifest.json" \
  --output-dir "$OUTPUT_DIR/eval_standard" \
  --dataset standard-val \
  --batch-size 16 \
  --innerloop-steps 20 \
  --seed 314159 \
  2>&1 | tee "$OUTPUT_DIR/logs/eval_standard.log"

CURRENT_PHASE=hard_eval_18000
write_state "$CURRENT_PHASE" running artifact_sync_and_next_arm_decision \
  "hard evaluation is the final declared benchmark phase"
python -m repro.evaluate_sudoku \
  --checkpoint "$OUTPUT_DIR/train/latest.pt" \
  --manifest "$OUTPUT_DIR/train/training_manifest.json" \
  --output-dir "$OUTPUT_DIR/eval_hard" \
  --dataset hard-test \
  --batch-size 16 \
  --innerloop-steps 20 \
  --seed 271828 \
  2>&1 | tee "$OUTPUT_DIR/logs/eval_hard.log"

CURRENT_PHASE=complete
write_state "$CURRENT_PHASE" complete artifact_sync_and_next_arm_decision \
  "retain until all signed artifacts are synchronized and the next compatible arm is chosen"
sha256sum \
  "$OUTPUT_DIR/train/training_manifest.json" \
  "$OUTPUT_DIR/train/latest.pt" \
  "$OUTPUT_DIR/eval_standard/summary.json" \
  "$OUTPUT_DIR/eval_hard/summary.json" > "$OUTPUT_DIR/final_artifact_hashes.sha256"
