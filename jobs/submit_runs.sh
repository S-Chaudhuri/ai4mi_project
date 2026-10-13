#!/bin/bash
# ============================================================================
# Submit experiments safely (run from the project root, ~/project).
#
#   bash jobs/submit_runs.sh list                  show the experiments
#   bash jobs/submit_runs.sh check <name>|all      check settings only, submit nothing
#   bash jobs/submit_runs.sh test  <name>          1-epoch check job (30 min limit)
#   bash jobs/submit_runs.sh run   <name> [...]    full runs ("all" = every experiment)
#   DRY_RUN=1 bash jobs/submit_runs.sh run all     check, then only print the sbatch lines
#
# Every selected experiment is checked first (same options as the real job);
# if any check fails, NOTHING is submitted.
#
# To add an experiment, add a line to EXPERIMENTS:
#   "name | job file | settings (CHANNELS=...,SKIP_AUGMENTED=...) | model and extra options"
# ============================================================================

set -euo pipefail

EXPERIMENTS=(
  "ResUNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|ResUNet3D"
  "MedNeXt3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|MedNeXt3D --training.batch_size 4"
  "UNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|UNet3D"
  "VNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|VNet3D"
  "ResUNet3D_3ch_aug|3d_npy.job|CHANNELS=3|ResUNet3D"
  "MedNeXt3D_3ch_aug|3d_npy.job|CHANNELS=3|MedNeXt3D --training.batch_size 4"
  "ResUNet3D_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1|ResUNet3D"
  "MedNeXt3D_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1|MedNeXt3D --training.batch_size 4"
  "UNet3D_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1|UNet3D"
  "VNet3D_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1|VNet3D"
  "ResUNet3D_aug_skel1|3d_npy.job|CHANNELS=1,SKELETON_WEIGHT=1|ResUNet3D --augmentation.enabled"
  "MedNeXt3D_aug_skel1|3d_npy.job|CHANNELS=1,SKELETON_WEIGHT=1|MedNeXt3D --training.batch_size 4 --augmentation.enabled"
  "UNet3D_aug_skel1|3d_npy.job|CHANNELS=1,SKELETON_WEIGHT=1|UNet3D --augmentation.enabled"
  "VNet3D_aug_skel1|3d_npy.job|CHANNELS=1,SKELETON_WEIGHT=1|VNet3D --augmentation.enabled"
  "ResUNet3D_tversky_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1,LOSS=tversky|ResUNet3D"
  "ENet_2d_base|2d_png.job|SKIP_AUGMENTED=True,ONLINE_AUG=False|ENet"
  "ENet_2d_partial|2d_png.job|SKIP_AUGMENTED=True,ONLINE_AUG=False,MODE=partial|ENet"
)

[[ -d jobs && -f src/train.py ]] || { echo "Run this from the project root (cd ~/project)" >&2; exit 1; }

find_exp() {
  for e in "${EXPERIMENTS[@]}"; do [[ "${e%%|*}" == "$1" ]] && { echo "$e"; return; }; done
  echo "Unknown experiment '$1' (see: bash jobs/submit_runs.sh list)" >&2; exit 1
}

job_args() {  # job_args <experiment line> <check|test|run>: the model and options for the job file
  IFS='|' read -r _ _ _ args <<< "$1"
  if [[ "$2" == "test" ]]; then echo "$args --training.epochs 1"; else echo "$args"; fi
}

check() {  # check <experiment line> <mode>: run the job file in CHECK_ONLY mode on this node
  IFS='|' read -r name job env _ <<< "$1"
  IFS=',' read -ra envs <<< "$env"
  echo "--- checking $name"
  # shellcheck disable=SC2046
  env "${envs[@]}" CHECK_ONLY=1 bash "jobs/$job" $(job_args "$1" "$2")
}

submit() {  # submit <experiment line> <test|run>
  IFS='|' read -r name job env _ <<< "$1"
  local cmd=(sbatch --export="ALL,$env")
  if [[ "$2" == "test" ]]; then cmd+=(--time=00:30:00 -J "test_$name"); else cmd+=(-J "$name"); fi
  # shellcheck disable=SC2206
  cmd+=("jobs/$job" $(job_args "$1" "$2"))
  echo "${cmd[*]}"
  [[ -n "${DRY_RUN:-}" ]] || "${cmd[@]}"
}

mode=${1:-list}
case "$mode" in
  list)
    printf "%-20s %-12s %-50s %s\n" NAME JOB SETTINGS "MODEL / OPTIONS"
    for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n j s a <<< "$e"; printf "%-20s %-12s %-50s %s\n" "$n" "$j" "$s" "$a"; done ;;
  check|test|run)
    shift
    [[ $# -gt 0 ]] || { echo "Give experiment names, or 'all'" >&2; exit 1; }
    selected=()
    if [[ "$1" == "all" ]]; then
      selected=("${EXPERIMENTS[@]}")
    else
      for n in "$@"; do e=$(find_exp "$n") || exit 1; selected+=("$e"); done
    fi
    echo ">> code: $(git branch --show-current) $(git log --oneline -1)"
    git diff --quiet -- src jobs configs || echo ">> WARNING: uncommitted changes in src/, jobs/ or configs/: jobs use the files on disk when they start"
    failed=()
    for e in "${selected[@]}"; do check "$e" "$mode" || failed+=("${e%%|*}"); done
    if [[ ${#failed[@]} -gt 0 ]]; then
      echo ">> CHECK FAILED for: ${failed[*]}  ->  nothing submitted" >&2; exit 2
    fi
    echo ">> all checks passed"
    [[ "$mode" == "check" ]] && exit 0
    for e in "${selected[@]}"; do submit "$e" "$mode"; done ;;
  *) echo "Usage: bash jobs/submit_runs.sh list | check <name>|all | test <name> | run <name>...|all" >&2; exit 1 ;;
esac
