#!/usr/bin/env bash
## 2026.10.07 BRCA external validation — pan-cancer cross-section Hist2Pheno
##
## One HE slide + StarDist nuclei. No 10x tissue_positions / scalefactors JSON.
## OME PhysicalSize is ~0.548 µm/px, so a native 16px crop is ~8.76 µm.
## Features are rescaled to 0.5 µm/px first; patch_size=16 is then an 8 µm field,
## the same crop the pan-cancer UNI checkpoint was trained on.
##

## 2026.10.07 LLY Run
##   bash code/External_val/BRCA/demo.sh dry-run

##   bash code/External_val/BRCA/demo.sh features
##   bash code/External_val/BRCA/demo.sh h5ad
##   bash code/External_val/BRCA/demo.sh predict
##   bash code/External_val/BRCA/demo.sh plot

##   bash code/External_val/BRCA/demo.sh all
##
## One slide:  SAMPLE=GSM7782699_tissue_image bash code/External_val/BRCA/demo.sh features
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default)

set -euo pipefail

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

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/BRCA/pred_pancancer_section/figures/

Notebook: code/External_val/BRCA/Pred_visual_brca.ipynb
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  echo "=== BRCA ${1}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
  "${PYTHON}" -u code/External_val/BRCA/brca_external.py "${1}" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
}

case "${STEP}" in
  dry-run|features|h5ad|predict|probs|plot)
    run_step "${STEP}"
    ;;
  all)
    run_step features
    run_step h5ad
    run_step predict
    run_step plot
    ;;
  *)
    usage
    exit 1
    ;;
esac
