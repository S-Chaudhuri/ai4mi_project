#!/usr/bin/env bash
# Submission pipeline for a given config + weights: stages the run dir,
# wipes its eval/, runs the val and test evals, then assembles group-12.
# Run from the repo root:
#   bash scripts/submit.sh <config.yaml> <bestweights.pt> [run_dir]
# e.g.
#   bash scripts/submit.sh configs/best_run.yaml best_weights/MedNeXt3D_3ch/bestweights.pt
# The test set must already be in the dataset (src/preprocessing/preprocess_3d.py
# --dest_dir data/<dataset> --test_source_dir data/test).
# RAW overrides the raw NIfTI folder name (default segthor_full).
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
uv run python src/eval_3D.py "${EVAL_ARGS[@]}" --eval.submission

echo ">> test eval"
uv run python src/eval_3D.py "${EVAL_ARGS[@]}" --eval.split test

echo ">> assemble"
uv run python scripts/make_submission.py --run "$RUN"
