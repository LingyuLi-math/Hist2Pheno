#!/usr/bin/env bash
## 2026.10.26 NPC external validation — UNI2 A3 Hist2Pheno
##
## Two HE slides + StarDist nuclei. No 10x tissue_positions / scalefactors JSON.
## OME PhysicalSize is ~0.465 µm/px, so a native 14px crop is ~6.51 µm.
## Features are rescaled to 0.5 µm/px first; patch_size=14 is then a 7 µm field,
## the same crop the UNI2 A3 checkpoint was trained on.
##
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/NPC/demo.sh (UNI, patch_size=16).
##

## 2026.10.26 LLY Run
##   bash code/External_val/NPC/demo_uni2.sh dry-run

##   bash code/External_val/NPC/demo_uni2.sh features
##   bash code/External_val/NPC/demo_uni2.sh h5ad
##   bash code/External_val/NPC/demo_uni2.sh predict

##   bash code/External_val/NPC/demo_uni2.sh all    # use this to run all steps
##
## One slide:  SAMPLE=20210809-C-AH4199551 bash code/External_val/NPC/demo_uni2.sh features
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
Usage: bash code/External_val/NPC/demo_uni2.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate/.../best_mlp_gpu.pt
Outputs:
  data/Visium/NPC/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/NPC/pred_uni2_A3_bestmlp/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/NPC/pred_uni2_A3_bestmlp/{sample}/{sample}_l2_probs.npz
Figures: code/External_val/NPC/Pred_visual_npc_uni2.ipynb

Log: code/log/external_val_npc_uni2.log  (appended; screen print plus per-step and whole-run elapsed time)
UNI (patch 16) remains: bash code/External_val/NPC/demo.sh
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "NPC-UNI2" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py NPC uni2 "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_npc_uni2.log" "NPC-UNI2" "${STEP}" run_steps
