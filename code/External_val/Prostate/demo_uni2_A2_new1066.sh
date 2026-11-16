#!/usr/bin/env bash
## 2026.11.16 Prostate external validation — UNI2 A2 refit on the 1,066-section index.
## Same recipe as published A2 (k=8 mean, no gate). Fold 4.
## Does not write project_all_UNI, pred_pancancer_section, or pred_uni2_A2_bestmlp_paired_v2_seed42.
## UNI2 embeddings under project_all_UNI2 are reused.
##
##   bash code/External_val/Prostate/demo_uni2_A2_new1066.sh dry-run
##   SAMPLE=Patient_1__H2 CUDA_DEVICE=0 bash code/External_val/Prostate/demo_uni2_A2_new1066.sh predict
## Redo a finished step: FORCE=1
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_new1066_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
PROB_TAG="$(extval_prob_tag)" || {
  echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2
  exit 1
}
export UNI2_PRED_DIRNAME="pred_uni2_A2_new1066_paired_v2_seed42${PROB_TAG}"

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
Usage: bash code/External_val/Prostate/demo_uni2_A2_new1066.sh {dry-run|features|h5ad|predict|probs|plot|all}

SAMPLE=Patient_1            all sections of one patient
SAMPLE=Patient_1__H2        one Patient_1 series (H1 / H2 / V1)
SAMPLE=Patient_1__H2_5      one section
FORCE=1                     redo a finished step
CUDA_DEVICE=0               GPU

Checkpoint: ${UNI2_CKPT}
Outputs:
  data/Visium/Prostate/project_all_UNI2/{Patient_1__H2_5}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/Prostate/${UNI2_PRED_DIRNAME}/{Patient_1__H2_5}/{Patient_1__H2_5}_stardist_pred.h5ad
  data/Visium/Prostate/${UNI2_PRED_DIRNAME}/{Patient_1__H2_5}/{Patient_1__H2_5}_l2_probs.npz
Figures are drawn in the notebook (SAVE_FIGURES=False by default).

Notebook: code/External_val/Prostate/Pred_visual_prostate_patient1_H2_uni2_A2_new1066.ipynb
Log: code/log/external_val_prostate_uni2_A2_new1066_paired_v2_seed42.log
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "Prostate-UNI2-A2-new1066" "$1" \
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

extval_run_logged "${REPO}/code/log/external_val_prostate_uni2_A2_new1066_paired_v2_seed42.log" "Prostate-UNI2-A2-new1066" "${STEP}" run_steps
