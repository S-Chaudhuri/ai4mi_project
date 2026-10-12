#!/bin/bash
# Submit experiments with the right job file and settings (run from the project root).
#
#   bash jobs/submit_runs.sh list                 show the experiments
#   bash jobs/submit_runs.sh test <name>          1-epoch check, 30 min limit
#   bash jobs/submit_runs.sh run <name> [<name>]  full runs ("all" = every experiment)
#   DRY_RUN=1 bash jobs/submit_runs.sh run all    print the sbatch commands only

set -euo pipefail

# name | job file | settings (--export) | model and extra train.py options
EXPERIMENTS=(
  "ResUNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|ResUNet3D"
  "MedNeXt3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|MedNeXt3D --training.batch_size 4"
  "UNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|UNet3D"
  "VNet3D_3ch_base|3d_npy.job|CHANNELS=3,SKIP_AUGMENTED=True|VNet3D"
  "ResUNet3D_3ch_aug|3d_npy.job|CHANNELS=3|ResUNet3D"
  "MedNeXt3D_3ch_aug|3d_npy.job|CHANNELS=3|MedNeXt3D --training.batch_size 4"
  "ResUNet3D_skel1|3d_npy.job|CHANNELS=1,SKIP_AUGMENTED=True,SKELETON_WEIGHT=1|ResUNet3D"
  "ENet_2d_base|2d_png.job|SKIP_AUGMENTED=True,ONLINE_AUG=False|ENet"
  "ENet_2d_partial|2d_png.job|SKIP_AUGMENTED=True,ONLINE_AUG=False,MODE=partial|ENet"
)

[[ -d jobs && -f src/train.py ]] || { echo "Run this from the project root (~/project)" >&2; exit 1; }

find_exp() {
  for e in "${EXPERIMENTS[@]}"; do [[ "${e%%|*}" == "$1" ]] && { echo "$e"; return; }; done
  echo "Unknown experiment '$1' (see: bash jobs/submit_runs.sh list)" >&2; exit 1
}

submit() {  # submit <experiment line> <test|run>
  IFS='|' read -r name job env args <<< "$1"
  local cmd=(sbatch --export="ALL,$env")
  if [[ "$2" == "test" ]]; then
    cmd+=(--time=00:30:00 -J "test_$name" "jobs/$job" $args --training.epochs 1)
  else
    cmd+=(-J "$name" "jobs/$job" $args)
  fi
  echo "${cmd[*]}"
  [[ -n "${DRY_RUN:-}" ]] || "${cmd[@]}"
}

case "${1:-list}" in
  list)
    printf "%-20s %-12s %-50s %s\n" NAME JOB SETTINGS "MODEL / OPTIONS"
    for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n j s a <<< "$e"; printf "%-20s %-12s %-50s %s\n" "$n" "$j" "$s" "$a"; done ;;
  test|run)
    mode=$1; shift
    [[ $# -gt 0 ]] || { echo "Give experiment names, or 'all'" >&2; exit 1; }
    echo ">> code: $(git branch --show-current) $(git log --oneline -1)"
    git diff --quiet -- src || echo ">> WARNING: uncommitted changes in src/: jobs use the code on disk when they start"
    selected=()
    if [[ "$1" == "all" ]]; then
      selected=("${EXPERIMENTS[@]}")
    else
      for n in "$@"; do e=$(find_exp "$n") || exit 1; selected+=("$e"); done
    fi
    for e in "${selected[@]}"; do submit "$e" "$mode"; done ;;
  *) echo "Usage: bash jobs/submit_runs.sh list | test <name> | run <name>... | run all" >&2; exit 1 ;;
esac
