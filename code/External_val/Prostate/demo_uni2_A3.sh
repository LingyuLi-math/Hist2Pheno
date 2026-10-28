#!/usr/bin/env bash
## 2026.10.28: this file loads the paired_v2 A3 best_mlp (fold 0, k=8 mean with gate).
## The morning unsuffixed A3 checkpoint is not used.
## 2026.10.26 Prostate external validation — UNI2 A3 Hist2Pheno
##
## patch_size=14 on the 0.5 µm/px grid is a 7 µm field, the UNI2 A3 crop.
## This does not write project_all_UNI or pred_pancancer_section.
## Those stay with code/External_val/Prostate/demo.sh (UNI, patch_size=16).
##
##   bash code/External_val/Prostate/demo_uni2_A3.sh dry-run
##   CUDA_DEVICE=0 bash code/External_val/Prostate/demo_uni2_A3.sh all
## Redo a finished step: FORCE=1
## GPU: CUDA_DEVICE=...   One slide: SAMPLE=...
##

## 35 sections. Prefer SAMPLE=Patient_1 or SAMPLE=Patient_1__H1_2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt"
export UNI2_PRED_DIRNAME="pred_uni2_A3_bestmlp_paired_v2_seed42"

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
Usage: bash code/External_val/Prostate/demo_uni2_A3.sh {dry-run|features|h5ad|predict|probs|plot|all}

SAMPLE=Patient_1            all sections of one patient
SAMPLE=Patient_1__H1        one Patient_1 series (H1 / H2 / V1)
SAMPLE=Patient_2__H1        one Patient_2 series (H1 / H2 / H3 / V1 / V2)
SAMPLE=Patient_1__H1_2      one section
FORCE=1                     redo a finished step
CUDA_DEVICE=0               GPU

Checkpoint: ${UNI2_CKPT}
Outputs:
  data/Visium/Prostate/project_all_UNI2/{Patient_1__H1_2}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/Prostate/pred_uni2_A3_bestmlp_paired_v2_seed42/{Patient_1__H1_2}/{Patient_1__H1_2}_stardist_pred.h5ad
  data/Visium/Prostate/pred_uni2_A3_bestmlp_paired_v2_seed42/{Patient_1__H1_2}/{Patient_1__H1_2}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebooks: code/External_val/Prostate/Pred_visual_prostate_patient1_{H1,H2,V1}_uni2.ipynb
          code/External_val/Prostate/Pred_visual_prostate_patient2_{H1,H2,H3,V1,V2}_uni2.ipynb
Log: code/log/external_val_prostate_uni2_A3_paired_v2_seed42.log  (appended; screen print plus per-step and whole-run elapsed time)
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "Prostate-UNI2-A3" "$1" \
    "${PYTHON}" -u code/External_val/Prostate/prostate_external.py uni2 "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${REPO}/code/log/external_val_prostate_uni2_A3_paired_v2_seed42.log" "Prostate-UNI2-A3" "${STEP}" run_steps
