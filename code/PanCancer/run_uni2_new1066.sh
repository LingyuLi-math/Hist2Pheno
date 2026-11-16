#!/usr/bin/env bash
## Retrain published A2 or A3 on the current 1,066-section index.
## The 896-section directories stay where they are:
##   data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/
##   data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/
## This writes result_uni2_<arch>_new1066_paired_v2_seed42/ instead.
##
## A2L12 is A8's balanced L12 on the A2 trunk (k=8 mean, no gate).
## It does not write the finished A8 directory.
## fp16 overflows on the fold whose Neural L12 weight is hundreds.
## Rerun that one in bf16 (model.py is in the protocol hash, so all 5 folds retrain):
##   HIST2PHENO_AMP=bf16 CUDA_DEVICE=2 bash code/PanCancer/run_uni2_new1066.sh A2L12
##
## One GPU each:
##   CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A2
##   CUDA_DEVICE=1 bash code/PanCancer/run_uni2_new1066.sh A3
##   CUDA_DEVICE=2 bash code/PanCancer/run_uni2_new1066.sh A2L12
## Predict the new weight later with the same suffix:
##   RUN_SUFFIX=new1066_paired_v2_seed42 bash code/PanCancer/3_demo_pred.sh uni2 A3
set -euo pipefail

arch="${1:-}"
case "$arch" in
  A2|A3|A2L12) ;;
  *)
    echo "usage: CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A2|A3|A2L12" >&2
    exit 1
    ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"

export RUN_SUFFIX="new1066_paired_v2_seed${TRAIN_SEED:-42}"
if [[ "$RUN_SUFFIX" != new1066_* ]]; then
  echo "refusing to write the published paired_v2_seed42 directory" >&2
  exit 1
fi

exec bash code/PanCancer/1_demo_train.sh uni2 "$arch"
