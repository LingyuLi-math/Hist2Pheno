#!/usr/bin/env bash
## 2026.11.17, retrain the whole UNI2 ablation on the current seven-cohort index
## 在当前 7 个数据集（含修复后的 SPORA CRC/GBM）上从头重训全部消融实验。
##
##   cd /home/lingyu/ssd2/Python/Hist2Pheno
##   conda activate SeededNTM
##   bash code/PanCancer/retrain_all.sh                 # check -> index -> prepare -> all arms on 4 GPUs -> summary
##   STEP=check bash code/PanCancer/retrain_all.sh      # only the SPORA axis / UNI2 check
##   STEP=summary bash code/PanCancer/retrain_all.sh    # only the ablation table
##
## Settings (environment, all optional):
##   DATA_TAG=new1066axisfix   output suffix; old folders are never overwritten
##   GPUS="0 1 2 3"            one queue per GPU
##   SEEDS="42"                e.g. "42 43 44" for paired seeds (folds stay identical, SPLIT_SEED=42)
##   ARMS="..."                default: the 11 arms of run_ablation.sh (legacy_nospatial ... l12_balanced_nospatial)
##   WITH_K_SENSITIVITY=0      1 adds spatial_k4 and spatial_k16
##   FIX_AXIS=0                1 runs `run_spora_pancancer.sh fix-axis` when the check finds stale SPORA data
##   REBUILD_INDEX=1           0 skips prepare_pancancer_index.py
##   HIST2PHENO_AMP=bf16       same precision for every arm (fp16 overflowed with balanced L12)
##   HIST2PHENO_STORE_DTYPE=float16   halves the per-process scaled copies on disk
##   DECISION_OBJECTIVE=f1_acc the decision layer maximises (macro-F1 + accuracy) / 2
##
## Disk: each running process keeps a scaled train/val copy of the pooled
## features under data/PanCancer/scaled_cache (about N_cells x 1536 x 2 bytes
## with float16). Four GPUs means four copies at once.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
export DATA_TAG="${DATA_TAG:-new1066axisfix}"
export HIST2PHENO_AMP="${HIST2PHENO_AMP:-bf16}"
export HIST2PHENO_STORE_DTYPE="${HIST2PHENO_STORE_DTYPE:-float16}"
export DECISION_OBJECTIVE="${DECISION_OBJECTIVE:-f1_acc}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
read -r -a SEEDS <<< "${SEEDS:-42}"
LOGDIR="$REPO/code/log/retrain_${DATA_TAG}"
mkdir -p "$LOGDIR"
STEP="${STEP:-all}"

py() { conda run --no-capture-output -n "$CONDA_ENV" python -u "$@"; }
die() { echo "ERROR: $*" >&2; exit 1; }

if [[ -n "${ARMS:-}" ]]; then
  read -r -a ARM_LIST <<< "$ARMS"
else
  read -r -a ARM_LIST <<< "$(bash code/PanCancer/run_ablation.sh list | head -n 11 | tr '\n' ' ')"
fi
if [[ "${WITH_K_SENSITIVITY:-0}" == 1 ]]; then ARM_LIST+=(spatial_k4 spatial_k16); fi
NOSPATIAL=(legacy_nospatial adjacent_nospatial l12_balanced_nospatial)

# ---------------------------------------------------------------------------
# 1. SPORA check: tables column-first and UNI2 stores newer than their tables
check_spora() {
  echo "===== [1/5] SPORA axis / UNI2 check $(date '+%F %T') ====="
  local report
  report="$(py code/spora/prepare_spora_sections.py --fix-centroid-axis --dry-run \
            --datasets schurch2020coordinated hoefflin2026spatial | tee "$LOGDIR/spora_check.log")"
  echo "$report"
  local bad
  bad="$(echo "$report" | grep -Eo '(would swap|unresolved|would retire|no UNI2 store yet) [0-9]+' \
         | awk '{s+=$NF} END {print s+0}')"
  if ! echo "$report" | grep -q "would retire"; then
    die "prepare_spora_sections.py is the old version (no UNI2 staleness report). Apply the latest code first."
  fi
  if [[ "$bad" -gt 0 ]]; then
    if [[ "${FIX_AXIS:-0}" == 1 ]]; then
      echo "fixing SPORA tables / UNI2 stores"
      bash code/spora/run_spora_pancancer.sh fix-axis 2>&1 | tee "$LOGDIR/spora_fix_axis.log"
    else
      die "SPORA data needs work (see above). Rerun with FIX_AXIS=1, or run: bash code/spora/run_spora_pancancer.sh fix-axis"
    fi
  fi
}

# ---------------------------------------------------------------------------
# 2. index, 3. one prepare that builds uni2_aligned/ and the shared pooled features
rebuild_index() {
  [[ "${REBUILD_INDEX:-1}" == 1 ]] || { echo "===== [2/5] index skipped ====="; return; }
  echo "===== [2/5] section index $(date '+%F %T') ====="
  bash code/PanCancer/1_demo_train.sh index 2>&1 | tee "$LOGDIR/index.log"
}

warm_prepare() {
  echo "===== [3/5] prepare (shared caches) on GPU ${GPUS[0]} $(date '+%F %T') ====="
  CUDA_DEVICE="${GPUS[0]}" TRAIN_SEED="${SEEDS[0]}" TRAIN_STEPS="prepare" FIT_DECISION=0 \
    bash code/PanCancer/run_ablation.sh spatial_k8 2>&1 | tee "$LOGDIR/prepare.log"
}

# ---------------------------------------------------------------------------
# 4. all arms x seeds, one sequential queue per GPU.
# GPU 0 starts with spatial_k8 and builds the k=8 neighbor cache; the other
# queues start with arms that need no neighbors, and every other k=8 arm
# waits until that cache is ready, so the 1,536-wide cache is built once.
train_all() {
  echo "===== [4/5] train ${#ARM_LIST[@]} arms x ${#SEEDS[@]} seed(s) on GPUs ${GPUS[*]} $(date '+%F %T') ====="
  local -a jobs=()
  local seed arm
  for seed in "${SEEDS[@]}"; do
    jobs+=("spatial_k8:$seed")
    for arm in "${ARM_LIST[@]}"; do
      [[ " ${NOSPATIAL[*]} " == *" $arm "* ]] && jobs+=("$arm:$seed")
    done
    for arm in "${ARM_LIST[@]}"; do
      [[ "$arm" == spatial_k8 || " ${NOSPATIAL[*]} " == *" $arm "* ]] && continue
      jobs+=("$arm:$seed")
    done
  done
  local n=${#GPUS[@]} i=0
  local -a queue
  for ((i=0; i<n; i++)); do queue[i]=""; done
  queue[0]="${jobs[0]}"
  for ((i=1; i<${#jobs[@]}; i++)); do
    local q=$(( i % n ))
    queue[q]="${queue[q]} ${jobs[i]}"
  done
  local k8_log
  k8_log="$REPO/code/log/pancancer_section_uni2_A2_adjacent_spatial_k8mean_${DATA_TAG}_paired_v2_seed${SEEDS[0]}.log"
  [[ "${SPLITTER:-cross_dataset_section}" != cross_dataset_section ]] && \
    k8_log="$REPO/code/log/pancancer_section_uni2_A2_adjacent_spatial_k8mean_stratsec_${DATA_TAG}_paired_v2_seed${SEEDS[0]}.log"
  local -a pids=()
  for ((i=0; i<n; i++)); do
    local gpu="${GPUS[i]}"
    (
      for job in ${queue[i]}; do
        arm="${job%%:*}"; seed="${job##*:}"
        if [[ "$i" != 0 && " ${NOSPATIAL[*]} " != *" $arm "* && "$arm" != spatial_k4 && "$arm" != spatial_k16 ]]; then
          # wait for the k=8 cache, at most CACHE_WAIT_MIN minutes (then build it here too)
          local waited=0
          until grep -q "context cache ready" "$k8_log" 2>/dev/null || (( waited >= ${CACHE_WAIT_MIN:-360} )); do
            sleep 60; waited=$((waited + 1))
          done
        fi
        echo "[GPU $gpu] $(date '+%F %T') start $arm seed $seed"
        if ! CUDA_DEVICE="$gpu" TRAIN_SEED="$seed" \
          bash code/PanCancer/run_ablation.sh "$arm" >"$LOGDIR/${arm}_seed${seed}.out" 2>&1; then
          echo "[GPU $gpu] FAILED $arm seed $seed (see $LOGDIR/${arm}_seed${seed}.out)" | tee -a "$LOGDIR/${arm}_seed${seed}.out"
        fi
        echo "[GPU $gpu] $(date '+%F %T') done  $arm seed $seed"
      done
    ) &
    pids+=($!)
    echo "GPU $gpu queue:${queue[i]}"
  done
  wait "${pids[@]}"
  if grep -h FAILED "$LOGDIR"/*.out >/dev/null 2>&1; then echo "some arms failed; see $LOGDIR"; fi
}

summary() {
  echo "===== [5/5] ablation table $(date '+%F %T') ====="
  bash code/PanCancer/run_ablation.sh summary 2>&1 | tee "$LOGDIR/summary.log"
}

case "$STEP" in
  check) check_spora ;;
  index) rebuild_index ;;
  prepare) warm_prepare ;;
  train) train_all ;;
  summary) summary ;;
  all) check_spora; rebuild_index; warm_prepare; train_all; summary ;;
  *) die "STEP must be all | check | index | prepare | train | summary" ;;
esac
