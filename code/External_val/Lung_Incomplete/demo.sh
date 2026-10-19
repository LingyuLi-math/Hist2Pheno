#!/usr/bin/env bash
## Xenium lung Incomplete_Cases — pan-cancer cross-section Hist2Pheno (no GT).
## UNI StarDist-all h5ads already exist. Predict is already in
## data/PanCancer/pred_all_spatial/incomplete_lung/. Notebooks draw 5 slides.
## No TLS / SRI / TNI.
##
##   bash code/External_val/Lung_Incomplete/demo.sh status
##   bash code/External_val/Lung_Incomplete/demo.sh rank
##   bash code/External_val/Lung_Incomplete/demo.sh predict
##   SAMPLE=VUILD141MA bash code/External_val/Lung_Incomplete/demo.sh plot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export CUDA_DEVICE

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
SAMPLE_ARG=()
if [[ -n "${SAMPLE:-}" ]]; then
  SAMPLE_ARG=(--sample "${SAMPLE}")
fi
FORCE_ARG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARG=(--force)
fi
ALL_ARG=()
if [[ "${ALL:-0}" == "1" ]]; then
  ALL_ARG=(--all)
fi

usage() {
  cat <<EOF
Usage: bash code/External_val/Lung_Incomplete/demo.sh {status|dry-run|rank|predict|plot}

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Preds (already written): data/PanCancer/pred_all_spatial/incomplete_lung/
Notebook: code/External_val/Lung_Incomplete/Pred_visual_lung_incomplete.ipynb
SAMPLE=   one slide   ALL=1  every Incomplete_Cases slide
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extra=()
  if [[ "$1" == "plot" ]]; then
    extra+=(--save)
  fi
  extval_time_step "Lung_Incomplete" "$1" \
    "${PYTHON}" -u code/External_val/unlabeled_external.py Lung_Incomplete "$1" \
      --cuda-device "${CUDA_DEVICE}" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" "${ALL_ARG[@]}" "${extra[@]}"
}

run_steps() {
  case "${STEP}" in
    status|dry-run|rank|predict|plot)
      run_step "${STEP}" || return $?
      ;;
    all)
      run_step predict || return $?
      ;;
    *)
      usage
      return 1
      ;;
  esac
}

extval_run_logged "${SCRIPT_DIR}/Lung_Incomplete.log" "Lung_Incomplete" "${STEP}" run_steps
