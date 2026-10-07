# Shared screen log for code/External_val/*/demo.sh.
# Appends terminal output to one dataset .log and records each step plus the
# whole invocation. Source this file; do not execute it.

extval_hms() {
  local t="$1"
  printf '%02d:%02d:%02d' $((t / 3600)) $(((t % 3600) / 60)) $((t % 60))
}

extval_say() {
  printf '%s\n' "$*" | tee -a "${EXTVAL_LOG}"
}

# Time one step. The command's stdout/stderr is copied to the terminal and the log.
# Prints elapsed time even when the command fails, then returns its status.
extval_time_step() {
  local label="$1"
  local step_name="$2"
  shift 2
  local start end elapsed rc errexit=0
  case $- in *e*) errexit=1 ;; esac
  start=$(date +%s)
  extval_say "=== ${label} ${step_name} start $(date '+%Y-%m-%d %H:%M:%S')  SAMPLE=${SAMPLE:-all}  FORCE=${FORCE:-0}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  MPP_UM=${MPP_UM:-} ==="
  set +e
  "$@" 2>&1 | stdbuf -oL -eL tee -a "${EXTVAL_LOG}"
  rc=${PIPESTATUS[0]}
  if [[ "${errexit}" -eq 1 ]]; then
    set -e
  fi
  end=$(date +%s)
  elapsed=$((end - start))
  extval_say "=== ${label} ${step_name} elapsed $(extval_hms "${elapsed}") (${elapsed}s)  exit=${rc} ==="
  return "${rc}"
}

# Run the remaining arguments (a function) with a header and a total elapsed line.
# Usage: extval_run_logged /path/NAME.log NAME step_name run_steps
extval_run_logged() {
  local logfile="$1"
  local label="$2"
  local step="$3"
  shift 3
  EXTVAL_LOG="${logfile}"
  mkdir -p "$(dirname "${EXTVAL_LOG}")"
  local start end elapsed rc
  start=$(date +%s)
  extval_say ""
  extval_say "========== ${label} ${step}  $(date '+%Y-%m-%d %H:%M:%S') =========="
  extval_say "SAMPLE=${SAMPLE:-all}  FORCE=${FORCE:-0}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  MPP_UM=${MPP_UM:-}  host=$(hostname)  pid=$$"
  extval_say "log: ${EXTVAL_LOG}"
  set +e
  "$@"
  rc=$?
  set -e
  end=$(date +%s)
  elapsed=$((end - start))
  extval_say "========== ${label} ${step} finished $(date '+%Y-%m-%d %H:%M:%S')  exit=${rc}  elapsed=$(extval_hms "${elapsed}") (${elapsed}s) =========="
  return "${rc}"
}
