#!/usr/bin/env bash
########################################################
# 2026.11.01, UNI A2 deployment predict.
# Same scope as: CUDA_DEVICE=2 bash code/PanCancer/3_demo_pred.sh uni2 A2
# One best_mlp on every section in section_index.csv (BRCA, CRC, GBM, GIST,
# HCC, Lung, PDAC). Not the five-fold OOF score. Not Lung incomplete.
#
# CUDA_DEVICE=0 bash code/PanCancer/pred_uni_A2.sh
# CUDA_DEVICE=0 bash code/PanCancer/pred_uni_A2.sh --datasets Lung
# Train: bash code/PanCancer/train_uni_A2.sh
########################################################
set -euo pipefail
REPO="${HIST2PHENO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
export HIST2PHENO_CONTEXT_CACHE="${REPO}/data/PanCancer/uni_context"
CKPT="${REPO}/data/PanCancer/result_uni_A2/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
ROOT="${REPO}/data/PanCancer/pred_uni_A2_deployment"
LOG="${REPO}/code/log/pancancer_pred_uni_A2.log"
mkdir -p "$(dirname "$LOG")" "${HIST2PHENO_CONTEXT_CACHE}"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: missing UNI A2 checkpoint ${CKPT}" >&2
  echo "Train first: bash code/PanCancer/train_uni_A2.sh" >&2
  exit 1
fi
cd "${REPO}"
unset CUDA_VISIBLE_DEVICES || true
{
  echo "===== start $(date '+%F %T') gpu=${CUDA_DEVICE} ckpt=${CKPT} ====="
  echo "sections: all cohorts in section_index.csv unless --datasets is set"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    code/PanCancer/PanCancer_predict_visualize.py \
    --cuda-device "${CUDA_DEVICE}" \
    --embedder uni \
    --checkpoint "${CKPT}" \
    --pred-root "${ROOT}" \
    --deployment \
    --spatial-k 8 \
    --skip-figures \
    --skip-existing \
    "$@"
  echo "===== end $(date '+%F %T') ====="
} 2>&1 | tee -a "${LOG}"
