#!/usr/bin/env bash
## 2026.10.26 CRC external validation — UNI2 A2 Hist2Pheno
##
## patch_size=14 on the 0.5 µm/px grid is a 7 µm field, the UNI2 A2 crop.
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/CRC/demo.sh (UNI, patch_size=16).
##
## 2026.10.29: paired_v2 A2 best_mlp. The morning unsuffixed checkpoint was removed.
##   bash code/External_val/CRC/demo_uni2.sh dry-run
##   CUDA_DEVICE=0 bash code/External_val/CRC/demo_uni2.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=...   One slide: SAMPLE=...
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
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

usage() {
  cat <<EOF
Usage: bash code/External_val/CRC/demo_uni2.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: ${UNI2_CKPT}
Outputs:
  data/Visium/CRC/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/CRC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/CRC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/CRC/Pred_visual_crc_uni2.ipynb
Log: code/log/external_val_crc_uni2.log  (appended; screen print plus per-step and whole-run elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "CRC-UNI2" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py CRC uni2 "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
}

run_steps() {
  case "${STEP}" in
    dry-run|features|h5ad|predict|probs|plot)
      run_step "${STEP}" || return $?
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

extval_run_logged "${REPO}/code/log/external_val_crc_uni2.log" "CRC-UNI2" "${STEP}" run_steps
