#!/usr/bin/env bash
## Deploy the finished UNI2 A8 weight (k=8 mean + gate, balanced L12).
## best_mlp_gpu.pt is hce_group_fold_0.pt, so --fold 0 is required.
## Writes a new tree and does not touch the published A2 deployment.
##
##   ONLY=BRCA CUDA_DEVICE=0 bash code/Internal_val/demo_uni2_A8.sh predict

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -z "${PYTHON:-}" ]]; then
  for cand in \
    /nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python \
    /ssd2/users/lingyu/conda_envs/SeededNTM/bin/python
  do
    if [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi
PYTHON="${PYTHON:-$(command -v python || true)}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: SeededNTM python not found. Activate the env or set PYTHON=..." >&2
  exit 1
fi

STEP="${1:-}"
if [[ "${STEP}" != "predict" ]]; then
  echo "Usage: ONLY=BRCA CUDA_DEVICE=0 bash code/Internal_val/demo_uni2_A8.sh predict" >&2
  exit 1
fi

cd "${REPO}"
UNI2="${REPO}/data/PanCancer/result_uni2_A8_l12_balanced_paired_v2_seed42"
PRED_CMD=(
  "${PYTHON}" -u code/PanCancer/PanCancer_predict_visualize.py
  --embedder uni2
  --checkpoint "${UNI2}/cross_dataset_cv/A8_l12_balanced/best_mlp_gpu.pt"
  --pred-root "${REPO}/data/PanCancer/pred_uni2_A8_l12_balanced_paired_v2_seed42_deployment"
  --cuda-device "${CUDA_DEVICE}"
  --fold 0
  --spatial-k 8
  --deployment
  --skip-existing
  --skip-figures
)
if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a _ds <<< "${ONLY}"
  PRED_CMD+=(--datasets "${_ds[@]}")
fi
"${PRED_CMD[@]}"
