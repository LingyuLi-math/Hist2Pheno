#!/usr/bin/env bash
## Predict slides whose UNI embeddings are already on disk.
## Does not extract features. Safe to run while demo.sh all is still on GPU 0.
##
## Current ready set (UNI done, no pred h5ad yet):
##   L-8001_HE L-8002_HE L-8003_HE L-8004_HE L-8005_HE
##
##   CUDA_DEVICE=1 bash code/External_val/HCC_nature/predict_ready.sh
##   CUDA_DEVICE=2 SAMPLE=L-8001_HE,L-8002_HE bash code/External_val/HCC_nature/predict_ready.sh
##
## Default GPU is 1. Writes a separate log so it does not mix with demo.sh all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE="${CUDA_DEVICE:-1}"
export READY="${READY:-1}"
export LOG="${LOG:-${SCRIPT_DIR}/HCC_nature_predict_ready.log}"

exec bash "${SCRIPT_DIR}/demo.sh" predict-ready
