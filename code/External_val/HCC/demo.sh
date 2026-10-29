#!/usr/bin/env bash
## 2026.10.07 HCC external validation — pan-cancer cross-section Hist2Pheno
##
## Seven HE JPEGs + StarDist nuclei. No 10x tissue_positions / scalefactors JSON.
## The JPEGs have no OME PhysicalSize. 96 dpi is a Windows Photo Viewer placeholder.
## Set MPP_UM to the scanner µm/px. Features are then rescaled to 0.5 µm/px so
## patch_size=16 is an 8 µm field, the same crop the checkpoint was trained on.
##

##   MPP_UM=0.5 bash code/External_val/HCC/demo.sh dry-run

##   MPP_UM=0.5 bash code/External_val/HCC/demo.sh features
##   MPP_UM=0.5 bash code/External_val/HCC/demo.sh h5ad
##   MPP_UM=0.5 bash code/External_val/HCC/demo.sh predict
##   MPP_UM=0.5 bash code/External_val/HCC/demo.sh plot

##   CUDA_DEVICE=3 MPP_UM=0.5 bash code/External_val/HCC/demo.sh all
##
## One slide:  MPP_UM=0.5 SAMPLE=P1_T bash code/External_val/HCC/demo.sh features
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=2 (default for this cohort). Override with CUDA_DEVICE=0.
## 0.5 above is only an example. Replace it with the real scanner calibration.

## 2026.10.15 Predict with new model
## HCC（7 张，~134 万核）：必须带 MPP_UM。上次成功的 features 用的是 0.5。
# CUDA_DEVICE=0 FORCE=1 MPP_UM=0.5 bash code/External_val/HCC/demo.sh predict


## UNI2 A3 (patch 14, separate directories):
##   MPP_UM=0.5 bash code/External_val/HCC/demo_uni2.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -z "${MPP_UM:-}" ]]; then
  echo "ERROR: set MPP_UM to the scanner µm/px." >&2
  echo "The HCC JPEGs have no OME PhysicalSize. 96 dpi is not a pixel size." >&2
  echo "Example: MPP_UM=0.5 bash code/External_val/HCC/demo.sh dry-run" >&2
  exit 1
fi
export MPP_UM

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
Usage: MPP_UM=<µm/px> bash code/External_val/HCC/demo.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/HCC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/HCC/pred_uni_A2/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/HCC/pred_uni_A2/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/HCC/Pred_visual_hcc.ipynb
Log: code/External_val/HCC/HCC.log  (appended; each step and the whole run record elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "HCC" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py HCC "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${SCRIPT_DIR}/HCC.log" "HCC" "${STEP}" run_steps
