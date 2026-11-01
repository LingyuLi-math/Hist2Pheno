#!/usr/bin/env bash
########################################################
# 2026.10.29, UNI A2 / 独立特征、权重、scaler；默认不启用 gate。
# bash code/PanCancer/train_uni_A2.sh section train
# bash code/PanCancer/train_uni_A2.sh 
# Only choose datasets whose UNI GT embeddings have been extracted.
########################################################
set -euo pipefail
REPO="${HIST2PHENO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
exec "${PYTHON}" -u "${REPO}/code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py" \
  --data-root "${REPO}/data/PanCancer" --embedder uni --steps prepare train \
  --save-result result_uni_A2 --ablation-tag A2_adjacent_spatial_k8mean \
  --use-spatial-context --spatial-mode mean --spatial-k 8 \
  --hierarchy adjacent --legacy-l2-l1-weight 2 --seed "${TRAIN_SEED:-42}" \
  --split-seed "${SPLIT_SEED:-42}" --context-cache-dir "${REPO}/data/PanCancer/uni_context" "$@"
