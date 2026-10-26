#!/usr/bin/env bash
## h5ad + A3 predict for HCC Nature slides whose UNI2 embeddings are already on disk.
## Does not extract features.
##
##   CUDA_DEVICE=1 bash code/External_val/HCC_nature/predict_ready_uni2.sh
##
## Default GPU is 1. Log: code/log/external_val_hcc_nature_uni2_predict_ready.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE="${CUDA_DEVICE:-1}"
export READY="${READY:-1}"
export LOG="${LOG:-/home/lingyu/ssd2/Python/Hist2Pheno/code/log/external_val_hcc_nature_uni2_predict_ready.log}"

exec bash "${SCRIPT_DIR}/demo_uni2.sh" predict-ready
