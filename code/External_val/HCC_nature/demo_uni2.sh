#!/usr/bin/env bash
## 2026.10.26 HCC_nature external validation — UNI2 A3 Hist2Pheno
##
## patch_size=14 on the 0.5 µm/px grid is a 7 µm field, the UNI2 A3 crop.
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/HCC_nature/demo.sh (UNI, patch_size=16).
##
##   bash code/External_val/HCC_nature/demo_uni2.sh dry-run
##   bash CUDA_DEVICE=0 code/External_val/HCC_nature/demo_uni2.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=...   One slide: SAMPLE=...
##

## predict-ready scores slides whose UNI2 embeddings are already on disk.
##   bash code/External_val/HCC_nature/demo_uni2.sh predict-ready

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -n "${MPP_UM:-}" ]]; then
  export MPP_UM
fi

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
  IFS=',' read -ra _ids <<< "${SAMPLE}"
  for _id in "${_ids[@]}"; do
    _id="${_id// /}"
    if [[ -n "${_id}" ]]; then
      SAMPLE_ARG+=(--sample "${_id}")
    fi
  done
fi
READY_ARG=()
if [[ "${READY:-0}" == "1" ]]; then
  READY_ARG=(--ready)
fi
FORCE_ARG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARG=(--force)
fi

usage() {
  cat <<EOF
Usage: bash code/External_val/HCC_nature/demo_uni2.sh {dry-run|features|h5ad|predict|probs|plot|predict-ready|all}

predict-ready: h5ad + predict for slides that already have UNI2 embeddings (see predict_ready_uni2.sh).
Checkpoint: data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate/.../A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt
Outputs:
  data/Visium/HCC/nature_data/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/HCC/nature_data/pred_uni2_A3_bestmlp/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/HCC/nature_data/pred_uni2_A3_bestmlp/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/HCC_nature/Pred_visual_hcc_nature_uni2.ipynb
Log: code/log/external_val_hcc_nature_uni2.log  (appended; screen print plus per-step and whole-run elapsed time)
Distinct from External_val/HCC (seven Visium JPEGs under data/Visium/HCC/).
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extra=()
  if [[ "$1" == "h5ad" || "$1" == "predict" || "$1" == "probs" || "$1" == "dry-run" || "$1" == "plot" ]]; then
    extra+=("${READY_ARG[@]}")
  fi
  extval_time_step "HCC_nature-UNI2" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py HCC_NATURE uni2 "$1" \
      "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" "${extra[@]}"
}

run_steps() {
  case "${STEP}" in
    dry-run|features|h5ad|predict|probs|plot)
      run_step "${STEP}" || return $?
      ;;
    predict-ready)
      READY_ARG=(--ready)
      run_step h5ad || return $?
      run_step predict || return $?
      ;;
    all)
      run_step features || return $?
      run_step h5ad || return $?
      run_step predict || return $?
      ;;
    *)
      usage
      return 1
      ;;
  esac
}

extval_run_logged "${LOG:-${REPO}/code/log/external_val_hcc_nature_uni2.log}" "HCC_nature-UNI2" "${STEP}" run_steps
