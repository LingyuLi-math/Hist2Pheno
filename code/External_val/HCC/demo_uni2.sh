#!/usr/bin/env bash
## 2026.10.26 HCC external validation — UNI2 A3 Hist2Pheno
##
## patch_size=14 on the 0.5 µm/px grid is a 7 µm field, the UNI2 A3 crop.
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/HCC/demo.sh (UNI, patch_size=16).
##
##   bash code/External_val/HCC/demo_uni2.sh dry-run
##   bash code/External_val/HCC/demo_uni2.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=...   One slide: SAMPLE=...
##

## MPP_UM is required. The finished UNI run used 0.5.
##   MPP_UM=0.5 bash code/External_val/HCC/demo_uni2.sh all
##   MPP_UM=0.5 CUDA_DEVICE=0 bash code/External_val/HCC/demo_uni2.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -z "${MPP_UM:-}" ]]; then
  echo "ERROR: set MPP_UM to the scanner µm/px." >&2
  echo "The HCC JPEGs have no OME PhysicalSize. 96 dpi is not a pixel size." >&2
  echo "Example: MPP_UM=0.5 bash code/External_val/HCC/demo_uni2.sh dry-run" >&2
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
Usage: MPP_UM=<µm/px> bash code/External_val/HCC/demo_uni2.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate/.../A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt
Outputs:
  data/Visium/HCC/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/HCC/pred_uni2_A3_bestmlp/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/HCC/pred_uni2_A3_bestmlp/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/HCC/Pred_visual_hcc_uni2.ipynb
Log: code/log/external_val_hcc_uni2.log  (appended; screen print plus per-step and whole-run elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "HCC-UNI2" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py HCC uni2 "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_hcc_uni2.log" "HCC-UNI2" "${STEP}" run_steps
