#!/usr/bin/env bash
########################################################
# 2026.10.29, UNI A2 external deployment / 默认 A2，不回退到 UNI2 或 A3。
# bash code/External_val/demo_uni.sh CRC all
########################################################
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HIST2PHENO_REPO:-$(cd "${HERE}/../.." && pwd)}"
COHORT="${1:?cohort required}"; STEP="${2:?step required}"; shift 2
PYTHON="${PYTHON:-$(command -v python)}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export UNI_CKPT="${UNI_CKPT:-${REPO}/data/PanCancer/result_uni_A2/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt}"
export UNI_PRED_DIRNAME="${UNI_PRED_DIRNAME:-pred_uni_A2}"
export HIST2PHENO_EMBEDDING_STORAGE=h5ad
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
extra=(); if [[ -n "${SAMPLE:-}" ]]; then extra+=(--sample "${SAMPLE}"); fi
if [[ "${FORCE:-0}" == 1 ]]; then extra+=(--force); fi
if [[ "${ALLOW_CPU:-0}" == 1 ]]; then extra+=(--allow-cpu); fi
case "${COHORT}" in
  BRCA|BRCA_ST|Prostate)
    case "${COHORT}" in BRCA) module=BRCA/brca_external.py;; BRCA_ST) module=BRCA_ST/brca_st_external.py;; Prostate) module=Prostate/prostate_external.py;; esac
    cmd=("${PYTHON}" -u "${REPO}/code/External_val/${module}" uni)
    ;;
  HCC_unlabel|PDAC_unlabel|Lung_Incomplete)
    if [[ "${ALL:-0}" == 1 ]]; then extra+=(--all); fi
    if [[ "${STEP}" == all ]]; then STEP=predict; fi
    exec "${PYTHON}" -u "${REPO}/code/External_val/unlabeled_external.py" "${COHORT}" "${STEP}" --embedder uni --cuda-device "${CUDA_DEVICE:-0}" "${extra[@]}" "$@"
    ;;
  CRC|HCC|NPC|HCC_NATURE)
    cmd=("${PYTHON}" -u "${REPO}/code/Hist2Pheno_pkg/external_validation.py" "${COHORT}" uni)
    ;;
  *) echo "Unknown external cohort ${COHORT}" >&2; exit 1;;
esac
if [[ "${STEP}" == all ]]; then
  for stage in features h5ad predict; do "${cmd[@]}" "${stage}" "${extra[@]}" "$@"; done
else
  "${cmd[@]}" "${STEP}" "${extra[@]}" "$@"
fi
