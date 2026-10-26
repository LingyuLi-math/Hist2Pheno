#!/usr/bin/env bash
## 2026.10.26 BRCA_ST external validation — UNI2 A3 Hist2Pheno
##
## patch_size=14 on the 0.5 µm/px grid is a 7 µm field, the UNI2 A3 crop.
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/BRCA_ST/demo.sh (UNI, patch_size=16).
##
##   bash code/External_val/BRCA_ST/demo_uni2.sh dry-run
##   bash code/External_val/BRCA_ST/demo_uni2.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=...   One slide: SAMPLE=...
##

## 36 slides. Prefer SAMPLE=A or SAMPLE=A1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
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
Usage: bash code/External_val/BRCA_ST/demo_uni2.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate/.../A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt
Outputs:
  data/ST/BRCA/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/ST/BRCA/pred_uni2_A3_bestmlp/{sample}/{sample}_stardist_pred.h5ad
  data/ST/BRCA/pred_uni2_A3_bestmlp/{sample}/{sample}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebooks: code/External_val/BRCA_ST/Pred_visual_brca_st_{A,B,C,D,E,F,G,H}_uni2.ipynb
Log: code/log/external_val_brca_st_uni2.log  (appended; screen print plus per-step and whole-run elapsed time)

SAMPLE=A1 or SAMPLE=A or SAMPLE=9769_C1_HE_small. About 2.4M nuclei across 36 slides.
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "BRCA_ST-UNI2" "$1" \
    "${PYTHON}" -u code/External_val/BRCA_ST/brca_st_external.py uni2 "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_brca_st_uni2.log" "BRCA_ST-UNI2" "${STEP}" run_steps
