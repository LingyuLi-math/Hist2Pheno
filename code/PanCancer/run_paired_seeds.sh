#!/usr/bin/env bash
########################################################
# 2026.10.26, paired seeds / 固定 cross-section 划分，顺序运行避免 GPU/缓存竞争
# SEEDS="42 43 44" MODELS="A2 A3" bash code/PanCancer/run_paired_seeds.sh
########################################################
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
read -r -a seeds <<< "${SEEDS:-42 43 44}"
read -r -a models <<< "${MODELS:-A0 A1 A2 A3}"
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 1; }
  for model in "${models[@]}"; do
    case "$model" in A0|A1|A2|A3) ;; *) echo "Unsupported model: $model" >&2; exit 1;; esac
    TRAIN_SEED="$seed" SPLIT_SEED="${SPLIT_SEED:-42}" RUN_SUFFIX="paired_v2_seed${seed}" \
      bash "$REPO/code/PanCancer/1_demo_train.sh" uni2 "$model"
  done
done
