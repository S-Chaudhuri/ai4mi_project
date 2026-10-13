#!/bin/bash
# Overview of all runs made with the job files (run from the project root).
#
#   bash jobs/results_table.sh            table of every run
#   bash jobs/results_table.sh skel       only runs whose name contains "skel"
#   CSV=1 bash jobs/results_table.sh      also write results/summary_<date>_<time>.csv
#
# Mean Dice / HD95 are the MEAN rows of eval/metrics.csv averaged over the four
# organs (3D runs only; 2D runs have no whole-volume evaluation).

set -uo pipefail
[[ -d results ]] || { echo "Run this from the project root (cd ~/project)" >&2; exit 1; }
filter=${1:-}

field() { grep -m1 "^$1:" "$2" | sed "s/^$1: *//"; }

rows=()
for info in results/*/*/run_info.txt; do
  [[ -f "$info" ]] || continue
  dir=$(dirname "$info")
  name=$(field run_name "$info"); job=$(field job_id "$info")
  [[ -n "$filter" && "$name" != *"$filter"* ]] && continue
  model=$(grep -o -- "--model.name [A-Za-z0-9]*" "$info" | tail -1 | cut -d' ' -f2)
  settings=$(field settings "$info" | sed 's/^model=[A-Za-z0-9]* //; s/ extra: *$//')
  status=$(field status "$info" | sed 's/ at .*//')
  dice=-; hd=-; eso=-
  m="$dir/eval/metrics.csv"
  if [[ -f "$m" ]]; then
    dice=$(awk -F, '/^MEAN/{s+=$3; n++} END{if(n) printf "%.3f", s/n}' "$m")
    hd=$(awk -F, '/^MEAN/{s+=$4; n++} END{if(n) printf "%.1f", s/n}' "$m")
    eso=$(awk -F, '/^MEAN,esophagus/{printf "%.3f", $3}' "$m")
  fi
  rows+=("$name|$job|$model|$status|$dice|$hd|$eso|$settings|$dir")
done

if [[ ${#rows[@]} -eq 0 ]]; then echo "No runs found in results/*/*/run_info.txt"; exit 0; fi

printf "%-22s %-10s %-10s %-20s %-9s %-9s %-8s %s\n" RUN JOB MODEL STATUS MEAN_DICE MEAN_HD95 ESO_DICE FOLDER
printf '%s\n' "${rows[@]}" | sort | while IFS='|' read -r n j mo st d h e s dir; do
  printf "%-22s %-10s %-10s %-20s %-9s %-9s %-8s %s\n" "$n" "$j" "$mo" "$st" "$d" "$h" "$e" "$dir"
  printf "%-22s   settings: %s\n" "" "$s"
done

if [[ -n "${CSV:-}" ]]; then
  csv=results/summary_$(date +%Y%m%d_%H%M%S).csv
  [[ -e "$csv" ]] && { echo "$csv exists, try again in a second" >&2; exit 1; }
  { echo "run,job,model,status,mean_dice,mean_hd95,esophagus_dice,settings,folder"
    printf '%s\n' "${rows[@]}" | sort | awk -F'|' -v OFS=',' '{print $1,$2,$3,$4,$5,$6,$7,"\""$8"\"",$9}'
  } > "$csv"
  echo ">> wrote $csv"
fi
