#!/usr/bin/env bash
## Internal validation of the UNI2 A2 best model.
## best_mlp_gpu.pt is the fold-1 checkpoint, so the gallery is fold 0.
## Predictions already cover all 896 training sections in
## pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment/.
## `fold0` below lists the fold-0 sections.
## `predict` is skip-existing and should be a no-op. It does not write the
## 4x3 panel PNGs (the deployment run used --skip-figures).
##
##   bash code/Internal_val/demo.sh status
##   bash code/Internal_val/demo.sh fold0
##   CUDA_DEVICE=1 bash code/Internal_val/demo.sh predict
##   ONLY=HCC bash code/Internal_val/demo.sh notebooks

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
usage() {
  cat <<EOF
Usage: bash code/Internal_val/demo.sh {status|fold0|predict|notebooks}

  status     UNI2 A2 fold-0 counts and whether pred files exist
  fold0      print fold-0 ids + the notebook viz subset
  predict    UNI2 A2 best_mlp on fold 0 only (--skip-existing, no new panels)
  notebooks  run Pred_visual_*.ipynb (does not rewrite the .ipynb)

ONLY=Lung,HCC     subset notebooks / status
CUDA_DEVICE=0     GPU for predict
EOF
}

if [[ -z "${STEP}" || "${STEP}" == "-h" || "${STEP}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${REPO}"

case "${STEP}" in
  status)
    "${PYTHON}" -u code/Internal_val/internal_val.py status ${ONLY:+--dataset "${ONLY%%,*}"}
    ;;
  fold0)
    "${PYTHON}" -u code/Internal_val/internal_val.py fold0 ${ONLY:+--dataset "${ONLY%%,*}"}
    ;;
  predict)
    UNI2="${REPO}/data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42"
    PRED_CMD=(
      "${PYTHON}" -u code/PanCancer/PanCancer_predict_visualize.py
      --embedder uni2
      --checkpoint "${UNI2}/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
      --pred-root "${REPO}/data/PanCancer/pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment"
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
    ;;
  notebooks)
    "${PYTHON}" -u code/Internal_val/run_pred_visual_notebooks.py ${ONLY:+--only "${ONLY}"}
    ;;
  *)
    usage
    exit 1
    ;;
esac
