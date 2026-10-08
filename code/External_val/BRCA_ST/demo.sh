#!/usr/bin/env bash
## 2026.10.08 HER2ST BRCA external validation — pan-cancer cross-section Hist2Pheno
##
## 36 HE JPEGs + StarDist nuclei under data/ST/BRCA. No 10x tissue_positions JSON.
## No OME PhysicalSize. 72 dpi is not a pixel size. µm/px is the 200 µm ST spot
## pitch on each image (~0.92 for patient A / 9769_*_small, ~0.69 for B–H).
## Features are rescaled to 0.5 µm/px; patch_size=16 is then an 8 µm field.
##
## About 2.4 million nuclei. Run one patient or one slide in screen, not all 36.
##

##   bash code/External_val/BRCA_ST/demo.sh dry-run

##   SAMPLE=A1 bash code/External_val/BRCA_ST/demo.sh features
##   SAMPLE=A bash code/External_val/BRCA_ST/demo.sh all
##   CUDA_DEVICE=2 SAMPLE=D bash code/External_val/BRCA_ST/demo.sh all
##
## SAMPLE is a HER2ST id (A1), a patient letter (A–H), or the HE stem
## (9769_C1_HE_small). Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default).
##
## One notebook per patient, so figures are not stored for all 36 slides:
##   code/External_val/BRCA_ST/Pred_visual_brca_st_{A,B,C,D,E,F,G,H}.ipynb
## A is 9769 (~0.92 µm/px, A1–A6). B–H are ~0.69 µm/px
## (B 10737, C 5714, D 10426 have 6 sections; E–H have 3).
## The notebooks draw the maps. SAVE_FIGURES=False only displays them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
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
Usage: bash code/External_val/BRCA_ST/demo.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/ST/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/ST/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad
  data/ST/BRCA/pred_pancancer_section/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebooks: code/External_val/BRCA_ST/Pred_visual_brca_st_{A,B,C,D,E,F,G,H}.ipynb
Log: code/External_val/BRCA_ST/BRCA_ST.log  (appended; each step and the whole run record elapsed time)

SAMPLE=A1 or SAMPLE=A or SAMPLE=9769_C1_HE_small. About 2.4M nuclei across 36 slides.
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "BRCA_ST" "$1" \
    "${PYTHON}" -u code/External_val/BRCA_ST/brca_st_external.py "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${SCRIPT_DIR}/BRCA_ST.log" "BRCA_ST" "${STEP}" run_steps
