#!/usr/bin/env bash
## 2026.10.07 Prostate external validation — pan-cancer cross-section Hist2Pheno
##
## Erickson et al. Nature 2022. JPEG HE + StarDist, two patients.
## Section names repeat, so a sample id is Patient_1__H1_2.
## JPEG 72 dpi is not used. This is Visium: each section with
## Visium_with_annotation/scalefactors_json.json uses
## µm/px = 65 / spot_diameter_fullres (same physical scale as NPC OME).
## No BRCA_ST 200 µm pitch. Else HEST (~0.34–0.35 for Patient_2).
## Features are rescaled to 0.5 µm/px; patch_size=16 is then an 8 µm field.
##
## Do not start with every slide: 35 sections, several million nuclei.
## One section or one patient in screen is the practical unit.

############################
##   bash code/External_val/Prostate/demo.sh dry-run

##   先确认一张
##   SAMPLE=Patient_1__H1_2 bash code/External_val/Prostate/demo.sh features
##   再跑 Patient 1 的 7 张
##   CUDA_DEVICE=2 SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh all
############################

## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default)
##
## Figures:
##   code/External_val/Prostate/Pred_visual_prostate_patient1_{H1,H2,V1}.ipynb
##     Patient_1 split by series (7 sections, about 0.57–0.69 µm/px)
##   code/External_val/Prostate/Pred_visual_prostate_patient2_{H1,H2,H3,V1,V2}.ipynb
##     Patient_2 split by series (28 sections, about 0.34–0.35 µm/px)
## The notebooks draw the maps. SAVE_FIGURES=False only displays them.
##   SAMPLE=Patient_2 bash code/External_val/Prostate/demo.sh all

## 2026.10.15 Predict with new model
## Prostate 不用环境变量 MPP_UM。每个 section 读自己的 scalefactors_json.json：
## µm/px = 65 / spot_diameter_fullres。没有 json 时退回 HEST。
# CUDA_DEVICE=1 FORCE=1 SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh predict
# CUDA_DEVICE=3 FORCE=1 SAMPLE=Patient_2 bash code/External_val/Prostate/demo.sh predict

## 2026.10.15 Patient_1 有 Visium_with_annotation 时，用 Visium 像素尺寸。
# SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh dry-run
# CUDA_DEVICE=1 FORCE=1 SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh all


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
Usage: bash code/External_val/Prostate/demo.sh {dry-run|features|h5ad|predict|probs|plot|all}

SAMPLE=Patient_1            all sections of one patient
SAMPLE=Patient_1__H1        one Patient_1 series (H1 / H2 / V1)
SAMPLE=Patient_2__H1        one Patient_2 series (H1 / H2 / H3 / V1 / V2)
SAMPLE=Patient_1__H1_2      one section
FORCE=1                     redo a finished step
CUDA_DEVICE=0               GPU

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/Prostate/project_all_UNI/{Patient_1__H1_2}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/Prostate/pred_pancancer_section/{Patient_1__H1_2}/{Patient_1__H1_2}_stardist_pred.h5ad
  data/Visium/Prostate/pred_pancancer_section/{Patient_1__H1_2}/{Patient_1__H1_2}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebooks: code/External_val/Prostate/Pred_visual_prostate_patient1_{H1,H2,V1}.ipynb
          code/External_val/Prostate/Pred_visual_prostate_patient2_{H1,H2,H3,V1,V2}.ipynb
Log: code/External_val/Prostate/Prostate.log  (appended; each step and the whole run record elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "Prostate" "$1" \
    "${PYTHON}" -u code/External_val/Prostate/prostate_external.py "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${SCRIPT_DIR}/Prostate.log" "Prostate" "${STEP}" run_steps
