#!/usr/bin/env bash
## CODEX PDAC Incomplete_Cases — pan-cancer cross-section Hist2Pheno (no GT).
## 195 cores without a cell-type CSV. UNI h5ads already exist under s1167/.
## Default predict/plot uses the 5 largest. No TLS / SRI / TNI.
##
##   bash code/External_val/PDAC_unlabel/demo.sh status
##   CUDA_DEVICE=0 bash code/External_val/PDAC_unlabel/demo.sh predict
##   bash code/External_val/PDAC_unlabel/demo.sh plot

## UNI2 paired_v2 best_mlp (separate pred directories):
##   bash code/External_val/PDAC_unlabel/demo_uni2_A2.sh predict
##   bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh predict

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
Usage: bash code/External_val/PDAC_unlabel/demo.sh {status|dry-run|predict|plot}

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs: data/PanCancer/pred_unlabel/PDAC_unlabel/
Notebook: code/External_val/PDAC_unlabel/Pred_visual_pdac_unlabel.ipynb
SAMPLE=   one core   ALL=1  all 195 Incomplete_Cases
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
  extval_time_step "PDAC_unlabel" "$1" \
    "${PYTHON}" -u code/External_val/unlabeled_external.py PDAC_unlabel "$1" \
      --cuda-device "${CUDA_DEVICE}" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" "${ALL_ARG[@]}" "${extra[@]}"
}

run_steps() {
  case "${STEP}" in
    status|dry-run|predict|plot)
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

extval_run_logged "${SCRIPT_DIR}/PDAC_unlabel.log" "PDAC_unlabel" "${STEP}" run_steps
