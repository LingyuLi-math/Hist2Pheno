#!/usr/bin/env bash
## Xenium lung Incomplete_Cases — UNI2 A2 Hist2Pheno (no GT).
## Reads StarDist UNI2 tokens (sc_pth_14_14) next to the existing h5ad.
## Does not write data/PanCancer/pred_all_spatial/incomplete_lung/.
## No TLS / SRI / TNI.
##
## 2026.10.29: paired_v2 A2 best_mlp. The morning unsuffixed checkpoint was removed.
##   bash code/External_val/Lung_Incomplete/demo_uni2.sh status
##   CUDA_DEVICE=0 bash code/External_val/Lung_Incomplete/demo_uni2.sh predict
##   CUDA_DEVICE=0 bash code/External_val/Lung_Incomplete/demo_uni2.sh all
##   ALL=1 bash code/External_val/Lung_Incomplete/demo_uni2.sh predict
##   SAMPLE=VUILD141MA bash code/External_val/Lung_Incomplete/demo_uni2.sh plot
## GPU: CUDA_DEVICE=0 (default). Python sees that card as cuda:0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export CUDA_DEVICE
export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
PROB_TAG="$(extval_prob_tag)" || {
  echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2
  exit 1
}
export UNI2_PRED_DIRNAME="pred_uni2_A2_bestmlp_paired_v2_seed42${PROB_TAG}"

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
Usage: bash code/External_val/Lung_Incomplete/demo_uni2.sh {status|dry-run|rank|predict|plot|all}

Checkpoint: ${UNI2_CKPT}
Outputs: data/PanCancer/${UNI2_PRED_DIRNAME}/unlabel/Lung_Incomplete/
Log: code/log/external_val_lung_incomplete_uni2.log
SAMPLE=   one slide   ALL=1  every Incomplete_Cases slide
A slide is skipped when project_all_UNI2/.../sc_pth_14_14 is missing.
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
  extval_time_step "Lung_Incomplete-UNI2" "$1" \
    "${PYTHON}" -u code/External_val/unlabeled_external.py Lung_Incomplete "$1" \
      --embedder uni2 --cuda-device "${CUDA_DEVICE}" \
      "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" "${ALL_ARG[@]}" "${extra[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_lung_incomplete_uni2.log" "Lung_Incomplete-UNI2" "${STEP}" run_steps
