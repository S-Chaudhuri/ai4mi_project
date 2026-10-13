# Results handling shared by the job files (sourced, not run).
#
# Every run gets its own folder that is never overwritten:
#     results/<run name>/<job id>/
#         run_info.txt     what ran: settings, command, code version, status
#         out.log          link to the job's log
#         bestweights.pt   best checkpoint
#         eval/            metrics.csv (3D) and submission/ (prediction NIfTIs)
#
# <run name> is the -J name of the job. Set RESULTS_DIR=... to choose another
# folder; it must not exist yet.

setup_results() {  # setup_results "$@": defines RESULTS_DIR, LOG_FILE, RUN_NAME, JOB_ID
  local a
  for a in "$@"; do
    [[ "$a" == --paths.results_dir* ]] && fail "don't pass --paths.results_dir: set RESULTS_DIR=<folder> instead (default results/<run name>/<job id>)"
  done
  RUN_NAME=${SLURM_JOB_NAME:-local}
  JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
  RESULTS_DIR=${RESULTS_DIR:-results/$RUN_NAME/$JOB_ID}
  LOG_FILE=logs/$RUN_NAME/$JOB_ID/out.log
}

create_results() {  # create_results "<settings summary>": make the folder and write run_info.txt
  mkdir -p "$(dirname "$RESULTS_DIR")"
  mkdir "$RESULTS_DIR" 2>/dev/null || fail "$RESULTS_DIR already exists: refusing to overwrite (set RESULTS_DIR to a new folder)"
  [[ -f "$LOG_FILE" ]] && ln -s "$PWD/$LOG_FILE" "$RESULTS_DIR/out.log"
  {
    echo "run_name:  $RUN_NAME"
    echo "job_id:    $JOB_ID"
    echo "started:   $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
    echo "code:      $(git branch --show-current 2>/dev/null) $(git log --oneline -1 2>/dev/null)"
    echo "settings:  $1"
    echo "command:   uv run python -O src/train.py ${TRAIN_ARGS[*]}"
    echo "log:       $LOG_FILE"
    echo "status:    running"
  } > "$RESULTS_DIR/run_info.txt"
  echo ">> results folder: $RESULTS_DIR"
}

finish_results() {  # finish_results <exit code>: record the outcome and print where everything is
  local rc=$1 status
  if [[ $rc -eq 0 ]]; then status="finished"; else status="FAILED (exit $rc)"; fi
  sed -i "s|^status: .*|status:    $status at $(date '+%Y-%m-%d %H:%M:%S')|" "$RESULTS_DIR/run_info.txt"
  echo ">> $status"
  echo ">> results folder: $RESULTS_DIR"
  [[ -f "$RESULTS_DIR/bestweights.pt" ]] && echo ">> checkpoint:     $RESULTS_DIR/bestweights.pt"
  if [[ -f "$RESULTS_DIR/eval/metrics.csv" ]]; then
    echo ">> metrics:        $RESULTS_DIR/eval/metrics.csv"
    grep "^MEAN" "$RESULTS_DIR/eval/metrics.csv" | sed 's/^/   /'
  fi
  echo ">> overview of all runs: bash jobs/results_table.sh"
}
