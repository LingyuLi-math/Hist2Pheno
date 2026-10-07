#!/usr/bin/env bash
## 2026.10.06 NPC external validation — pan-cancer cross-section Hist2Pheno
##
## Two HE slides + StarDist nuclei. No 10x tissue_positions / scalefactors JSON.
## OME PhysicalSize is ~0.465 µm/px, so a native 16px crop is ~7.44 µm.
## Features are rescaled to 0.5 µm/px first; patch_size=16 is then an 8 µm field,
## the same crop the pan-cancer UNI checkpoint was trained on.
##

## 2026.10.06 LLY Run
##   bash code/External_val/NPC/demo.sh dry-run

##   bash code/External_val/NPC/demo.sh features
##   bash code/External_val/NPC/demo.sh h5ad
##   bash code/External_val/NPC/demo.sh predict
##   bash code/External_val/NPC/demo.sh plot

##   bash code/External_val/NPC/demo.sh all
##
## One slide:  SAMPLE=20210809-C-AH4199551 bash code/External_val/NPC/demo.sh features
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=0 (default)

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
Usage: bash code/External_val/NPC/demo.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_all_spatial/.../D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
Outputs:
  data/Visium/NPC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/
  data/Visium/NPC/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/NPC/pred_pancancer_section/figures/

Notebook: code/External_val/NPC/Pred_visual_npc.ipynb
Log: code/External_val/NPC/NPC.log  (appended; each step and the whole run record elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "NPC" "$1" \
    "${PYTHON}" -u code/External_val/NPC/npc_external.py "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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
      run_step plot || return $?
      ;;
    *)
      usage
      return 1
      ;;
  esac
}

extval_run_logged "${SCRIPT_DIR}/NPC.log" "NPC" "${STEP}" run_steps
