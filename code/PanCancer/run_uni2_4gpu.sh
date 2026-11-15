#!/usr/bin/env bash
## Five-fold UNI2 training across four GPUs. One architecture at a time.
##
## Fold 0 starts alone and builds that run's spatial cache. Folds 1–3 start
## after the cache file exists. Fold 4 uses the first GPU once fold 0 has
## finished. The last step resumes every checkpoint, writes best_mlp_gpu.pt,
## and runs he_validate.
##
##   bash code/PanCancer/run_uni2_4gpu.sh A8
##   bash code/PanCancer/run_uni2_4gpu.sh A9
##   bash code/PanCancer/run_uni2_4gpu.sh A10
##   bash code/PanCancer/run_uni2_4gpu.sh A11
##   GPUS="0 1 2 3" bash code/PanCancer/run_uni2_4gpu.sh A9
##
## A10 writes data/PanCancer/mpp_map.csv first. A finished hce_group_fold_N.pt
## is reused. Delete that result's fold files before a clean rerun.
## Do not start two of these at once: they share the four GPUs.

set -euo pipefail

ARCH="${1:-}"
case "$ARCH" in
  A8) TAG_BASE="A8_l12_balanced"; RESULT="result_uni2_A8_l12_balanced" ;;
  A9) TAG_BASE="A9_nbr_comp"; RESULT="result_uni2_A9_nbr_comp" ;;
  A10) TAG_BASE="A10_nbr_comp_geom"; RESULT="result_uni2_A10_nbr_comp_geom" ;;
  A11) TAG_BASE="A11_nbr_tissue"; RESULT="result_uni2_A11_nbr_tissue" ;;
  *)
    echo "Usage: bash code/PanCancer/run_uni2_4gpu.sh {A8|A9|A10|A11}" >&2
    exit 1
    ;;
esac

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
SEED="${TRAIN_SEED:-42}"
SUFFIX="paired_v2_seed${SEED}"
TAG="${TAG_BASE}_${SUFFIX}"
CKPT="$REPO/data/PanCancer/${RESULT}_${SUFFIX}/cross_dataset_cv/${TAG_BASE}"
LOGDIR="$REPO/code/log"
mkdir -p "$LOGDIR"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "ERROR: GPUS must list 4 devices (got ${GPUS[*]})" >&2
  exit 1
fi

if [[ "$ARCH" == A10 ]]; then
  echo "===== mpp map $(date '+%F %T') ====="
  conda run --no-capture-output -n "$CONDA_ENV" python -u code/PanCancer/make_mpp_map.py
fi

shopt -s nullglob
existing=("$CKPT"/hce_group_fold_*.pt)
if [[ "${#existing[@]}" -gt 0 ]]; then
  echo "WARNING: ${#existing[@]} fold checkpoint(s) already exist and will be reused."
  echo "Delete $CKPT/hce_group_fold_*.pt first if every fold should train again."
fi

run_arch() {
  local gpu=$1
  shift
  TRAIN_STEPS="$1" LOG_TAG="$2" ONLY_FOLDS="${3:-}" CUDA_DEVICE="$gpu" \
    bash code/PanCancer/1_demo_train.sh uni2 "$ARCH"
}

echo "===== $ARCH prepare on GPU ${GPUS[0]} $(date '+%F %T') ====="
run_arch "${GPUS[0]}" prepare prepare

declare -a PIDS=()
launch() {
  local gpu=$1 fold=$2
  local wrapper="$LOGDIR/${ARCH,,}_4gpu_fold${fold}.wrapper.log"
  echo "===== $ARCH fold $fold on GPU $gpu $(date '+%F %T') ====="
  run_arch "$gpu" train "fold${fold}" "$fold" >"$wrapper" 2>&1 &
  PIDS[$fold]=$!
}

wait_fold() {
  local fold=$1 pid=${PIDS[$fold]}
  if wait "$pid"; then
    echo "===== $ARCH fold $fold done $(date '+%F %T') ====="
  else
    echo "ERROR: $ARCH fold $fold failed. See $LOGDIR/pancancer_section_uni2_${TAG}_fold${fold}.log" >&2
    exit 1
  fi
}

launch "${GPUS[0]}" 0
fold0_log="$LOGDIR/pancancer_section_uni2_${TAG}_fold0.log"
echo "waiting for the spatial cache from $ARCH fold 0"
until grep -q "context cache ready " "$fold0_log" 2>/dev/null; do
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
    echo "ERROR: $ARCH fold 0 exited before the context cache was ready" >&2
    wait "${PIDS[0]}" || true
    exit 1
  fi
  sleep 20
done
echo "===== context cache ready; starting folds 1-3 ====="

launch "${GPUS[1]}" 1
launch "${GPUS[2]}" 2
launch "${GPUS[3]}" 3
wait_fold 0
launch "${GPUS[0]}" 4
wait_fold 1
wait_fold 2
wait_fold 3
wait_fold 4

for fold in 0 1 2 3 4; do
  [[ -f "$CKPT/hce_group_fold_${fold}.pt" ]] || {
    echo "ERROR: missing $CKPT/hce_group_fold_${fold}.pt" >&2
    exit 1
  }
done

echo "===== $ARCH OOF and best_mlp on GPU ${GPUS[0]} $(date '+%F %T') ====="
run_arch "${GPUS[0]}" "train he_validate" finalize
echo "===== $ARCH 4-GPU run finished $(date '+%F %T') ====="
