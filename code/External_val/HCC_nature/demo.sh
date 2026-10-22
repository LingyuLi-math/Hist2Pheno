#!/usr/bin/env bash
## 2026.10.21 HCC Nature HE + StarDist — pan-cancer cross-section Hist2Pheno
##
## 17 TIFFs under data/Visium/HCC/nature_data (mixed plain .tif and .ome.tif).
## No 10x tissue_positions / scalefactors JSON and no cell-type ground truth.
## µm/px is read per slide: OME PhysicalSize when plausible (~0.501), else Aperio
## MPP, else 0.5. L-8015's OME 84.67 µm/px is a 300 dpi placeholder and is skipped.
## Features are rescaled to 0.5 µm/px so patch_size=16 is an 8 µm field.
##
##   bash code/External_val/HCC_nature/demo.sh dry-run
##   bash code/External_val/HCC_nature/demo.sh features
##   bash code/External_val/HCC_nature/demo.sh h5ad
##   bash code/External_val/HCC_nature/demo.sh predict
##   bash code/External_val/HCC_nature/demo.sh plot
##   CUDA_DEVICE=2 bash code/External_val/HCC_nature/demo.sh all
##
## One slide:  SAMPLE=L-8001_HE bash code/External_val/HCC_nature/demo.sh features
## OME slide:  SAMPLE=L-8006_HE.ome bash code/External_val/HCC_nature/demo.sh features
## Several:    SAMPLE=L-8001_HE,L-8002_HE bash code/External_val/HCC_nature/demo.sh predict
## UNI done, skip the rest of features:
##   CUDA_DEVICE=1 bash code/External_val/HCC_nature/predict_ready.sh
##   CUDA_DEVICE=1 bash code/External_val/HCC_nature/demo.sh predict-ready
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default for demo.sh). predict_ready.sh defaults to GPU 1.
## Optional: MPP_UM=0.5  is only the fallback for TIFFs with no calibration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
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
Usage: bash code/External_val/HCC_nature/demo.sh {dry-run|features|h5ad|predict|probs|plot|predict-ready|all}

predict-ready: h5ad + predict for slides that already have UNI (see predict_ready.sh).
Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/HCC/nature_data/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/HCC/nature_data/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/HCC/nature_data/pred_pancancer_section/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/HCC_nature/Pred_visual_hcc_nature.ipynb
Log: code/External_val/HCC_nature/HCC_nature.log
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
  extval_time_step "HCC_nature" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py HCC_NATURE "$1" \
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

extval_run_logged "${LOG:-${SCRIPT_DIR}/HCC_nature.log}" "HCC_nature" "${STEP}" run_steps
