#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:?usage: submit.sh <config.yaml> <bestweights.pt> [run_dir]}"
WEIGHTS="${2:?usage: submit.sh <config.yaml> <bestweights.pt> [run_dir]}"
RUN="${3:-results/submission}"
RAW="${RAW:-segthor_full}"

echo ">> staging $RUN from $CONFIG + $WEIGHTS"
mkdir -p "$RUN"
cp "$CONFIG" "$RUN/config.yaml"
cp "$WEIGHTS" "$RUN/bestweights.pt"

EVAL_ARGS=(--config "$RUN/config.yaml" --paths.data_path data
  --paths.raw_dataset "$RAW" --paths.results_dir "$RUN"
  --eval.weights "$RUN/bestweights.pt" --runtime.amp_dtype bf16)

echo ">> wiping $RUN/eval"
rm -rf "$RUN/eval"

echo ">> val eval"
python src/eval_3D.py "${EVAL_ARGS[@]}" --eval.submission

echo ">> test eval"
python src/eval_3D.py "${EVAL_ARGS[@]}" --eval.split test

echo ">> assemble"
python scripts/make_submission.py --run "$RUN"
