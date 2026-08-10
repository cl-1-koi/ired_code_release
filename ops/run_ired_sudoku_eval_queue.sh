#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 COMMIT SOURCE_DIR DATA_DIR INPUT_ROOT OUTPUT_ROOT GPU INNER_STEPS_CSV WAIT_SESSION" >&2
  exit 2
fi

commit="$1"
source_dir="$2"
data_dir="$3"
input_root="$4"
output_root="$5"
gpu="$6"
inner_steps_csv="$7"
wait_session="$8"

cd "$source_dir"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"

# Let the already-running 50k curve finish on this GPU before consuming the
# first landmark.  The checkpoint may arrive earlier than the long 80-step arm.
while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep 60
done

IFS=',' read -r -a inner_steps <<< "$inner_steps_csv"
for step in 100000 300000 1000000 1300000; do
  step9=$(printf '%09d' "$step")
  input_dir="$input_root/step_${step9}"
  checkpoint="$input_dir/checkpoint.pt"
  manifest="$input_dir/training_manifest.json"
  while [[ ! -s "$checkpoint" || ! -s "$manifest" ]]; do
    sleep 300
  done
  for inner in "${inner_steps[@]}"; do
    inner3=$(printf '%03d' "$inner")
    for dataset_seed in standard-val:314159 hard-test:271828; do
      dataset=${dataset_seed%%:*}
      seed=${dataset_seed##*:}
      output_dir="$output_root/step_${step9}/${dataset}-inner${inner3}-s${seed}"
      if [[ -s "$output_dir/completion.json" ]]; then
        continue
      fi
      if [[ -s "$output_dir/failure.json" ]]; then
        echo "refusing to overwrite failed evaluation $output_dir" >&2
        exit 1
      fi
      bash ops/run_ired_sudoku_eval.sh \
        "$commit" "$source_dir" "$data_dir" \
        "$checkpoint" "$manifest" "$output_dir" \
        "$dataset" "$inner" "$seed" "$gpu"
    done
  done
done
