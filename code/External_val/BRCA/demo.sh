#!/usr/bin/env bash
## 2026.10.07 BRCA external validation — pan-cancer cross-section Hist2Pheno
## 2026.10.15 add two SpatialVDJ pathology HE JPEGs (Visium-corrected µm/px)
##
## Samples under data/Visium/BRCA/StarDist_Segment/:
##   GSM7782699_tissue_image                          OME TIFF ~0.548 µm/px
##   200715_BC_SA2_C_D_V19F24-113.B1-Spot000001       JPEG; Visium V10F24-113_B1
##   200219_BC_SA3_V19T26-031_CE_V4-Spot000001        JPEG; Visium V19T26-031_D1
##
## Zenodo JPEGs are ~3.333× Space Ranger fullres. µm/px on the StarDist HE is
## (65 / spot_diameter_fullres) / (HE÷SR), ~0.172 µm/px — not raw 65/diameter.
## Entry point: code/External_val/BRCA/brca_external.py (not simple_cohort).

## 2026.10.07 LLY Run
##   bash code/External_val/BRCA/demo.sh dry-run

##   bash code/External_val/BRCA/demo.sh features
##   bash code/External_val/BRCA/demo.sh h5ad
##   bash code/External_val/BRCA/demo.sh predict
##   bash code/External_val/BRCA/demo.sh plot

##   bash code/External_val/BRCA/demo.sh all
##
## One slide:
##   SAMPLE=GSM7782699_tissue_image bash code/External_val/BRCA/demo.sh features
##   SAMPLE=200715_BC_SA2_C_D_V19F24-113.B1-Spot000001 bash code/External_val/BRCA/demo.sh all
##   SAMPLE=200219_BC_SA3_V19T26-031_CE_V4-Spot000001 bash code/External_val/BRCA/demo.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default)

## 2026.10.15 Predict with new model
## 必须 FORCE=1 predict，否则会跳过旧 pred
# CUDA_DEVICE=0 FORCE=1 bash code/External_val/BRCA/demo.sh predict

## UNI2 paired_v2 best_mlp (patch 14, separate directories):
##   bash code/External_val/BRCA/demo_uni2_A2.sh all
##   bash code/External_val/BRCA/demo_uni2_A3.sh all

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
Usage: bash code/External_val/BRCA/demo.sh {dry-run|features|h5ad|predict|probs|plot|all}

SAMPLE=GSM7782699_tissue_image
SAMPLE=200715_BC_SA2_C_D_V19F24-113.B1-Spot000001
SAMPLE=200219_BC_SA3_V19T26-031_CE_V4-Spot000001
FORCE=1                     redo a finished step
CUDA_DEVICE=0               GPU

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/BRCA/pred_uni_A2/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/BRCA/pred_uni_A2/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/BRCA/Pred_visual_brca.ipynb
Log: code/External_val/BRCA/BRCA.log  (appended; each step and the whole run record elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "BRCA" "$1" \
    "${PYTHON}" -u code/External_val/BRCA/brca_external.py "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${SCRIPT_DIR}/BRCA.log" "BRCA" "${STEP}" run_steps
