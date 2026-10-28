#!/usr/bin/env bash
## h5ad + paired_v2 A2 best_mlp for HCC Nature slides whose UNI2 embeddings are already on disk.
## Does not extract features.
##
##   CUDA_DEVICE=2 bash code/External_val/HCC_nature/predict_ready_uni2_A2.sh
##
## Default GPU is 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE="${CUDA_DEVICE:-2}"
export READY="${READY:-1}"
export LOG="${LOG:-/home/lingyu/ssd2/Python/Hist2Pheno/code/log/external_val_hcc_nature_uni2_A2_paired_v2_seed42_predict_ready.log}"

exec bash "${SCRIPT_DIR}/demo_uni2_A2.sh" predict-ready
