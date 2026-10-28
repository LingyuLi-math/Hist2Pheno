#!/usr/bin/env bash
## h5ad + paired_v2 A3 best_mlp for HCC Nature slides whose UNI2 embeddings are already on disk.
## Does not extract features. 2026.10.28: morning A3 checkpoint is not used.
##
##   CUDA_DEVICE=3 bash code/External_val/HCC_nature/predict_ready_uni2.sh
##
## Default GPU is 3. A2: predict_ready_uni2_A2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE="${CUDA_DEVICE:-3}"
export READY="${READY:-1}"
export LOG="${LOG:-/home/lingyu/ssd2/Python/Hist2Pheno/code/log/external_val_hcc_nature_uni2_A3_paired_v2_seed42_predict_ready.log}"

exec bash "${SCRIPT_DIR}/demo_uni2_A3.sh" predict-ready
