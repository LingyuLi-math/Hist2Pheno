#!/usr/bin/env bash
## 2026.10.28: this file loads the paired_v2 A3 best_mlp (fold 0, k=8 mean with gate).
## The morning unsuffixed A3 checkpoint is not used.
## CODEX PDAC Incomplete_Cases — UNI2 A3 Hist2Pheno (no GT).
## Reads StarDist UNI2 tokens (sc_pth_14_14) next to the existing h5ad.
## Does not write data/PanCancer/pred_unlabel/PDAC_unlabel/.
## Default predict/plot uses the 5 largest. No TLS / SRI / TNI.
##
##   bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh status
##   CUDA_DEVICE=0 bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh predict
##   ALL=1 bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh predict
## GPU: CUDA_DEVICE=0 (default). Python sees that card as cuda:0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
# 2026.10.28, one directory per cutoff / 0 keeps the unsuffixed folder
PROB_TAG="$(extval_prob_tag)" || {
  echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2
  exit 1
}
export UNI2_PRED_DIRNAME="pred_uni2_A3_bestmlp_paired_v2_seed42${PROB_TAG}"
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
Usage: bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh {status|dry-run|predict|plot|all}

Checkpoint: ${UNI2_CKPT}
Outputs: data/PanCancer/pred_uni2_A3_bestmlp_paired_v2_seed42/unlabel/PDAC_unlabel/
Log: code/log/external_val_pdac_unlabel_uni2_A3_paired_v2_seed42.log
SAMPLE=   one core   ALL=1  all 195 Incomplete_Cases
A core is skipped when project_all_UNI2/.../sc_pth_14_14 is missing.
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
  extval_time_step "PDAC_unlabel-UNI2-A3" "$1" \
    "${PYTHON}" -u code/External_val/unlabeled_external.py PDAC_unlabel "$1" \
      --embedder uni2 --cuda-device "${CUDA_DEVICE}" \
      "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" "${ALL_ARG[@]}" "${extra[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_pdac_unlabel_uni2_A3_paired_v2_seed42.log" "PDAC_unlabel-UNI2-A3" "${STEP}" run_steps
