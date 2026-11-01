#!/usr/bin/env bash
########################################################
# 2026.11.01, Virchow2 A2 deployment predict.
# One best_mlp on every section. Not the five-fold OOF score.
# CUDA_DEVICE=1 bash code/PanCancer/pred_virchow2_A2.sh
# CUDA_DEVICE=1 bash code/PanCancer/pred_virchow2_A2.sh --datasets Lung
# Train: bash code/PanCancer/train_virchow2_A2.sh
########################################################
set -euo pipefail
REPO="${HIST2PHENO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export HIST2PHENO_CONTEXT_CACHE="${REPO}/data/PanCancer/virchow2_context"
CKPT="${REPO}/data/PanCancer/result_virchow2_A2/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
ROOT="${REPO}/data/PanCancer/pred_virchow2_A2_deployment"
LOG="${REPO}/code/log/pancancer_pred_virchow2_A2.log"
mkdir -p "$(dirname "$LOG")" "${HIST2PHENO_CONTEXT_CACHE}"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: missing Virchow2 A2 checkpoint ${CKPT}" >&2
  echo "Train first: bash code/PanCancer/train_virchow2_A2.sh" >&2
  exit 1
fi
cd "${REPO}"
{
  echo "===== start $(date '+%F %T') gpu=${CUDA_DEVICE} ckpt=${CKPT} ====="
  "${PYTHON}" -u code/PanCancer/PanCancer_predict_visualize.py \
    --cuda-device "${CUDA_DEVICE}" \
    --embedder virchow2 \
    --checkpoint "${CKPT}" \
    --pred-root "${ROOT}" \
    --deployment \
    --spatial-k 8 \
    --skip-figures \
    --skip-existing \
    "$@"
  echo "===== end $(date '+%F %T') ====="
} 2>&1 | tee -a "${LOG}"
