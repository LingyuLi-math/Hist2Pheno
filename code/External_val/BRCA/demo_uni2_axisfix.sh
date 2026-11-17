#!/usr/bin/env bash
## BRCA 10x slide — finished 2026-11-17 ablation weights.
##
## Slide: V1_Breast_Cancer_Block_A_Section_1_image
## (10x Human Breast Cancer Block A Section 1, Space Ranger fullres TIFF).
## The other three BRCA slides are not predicted.
##
## Names are the ablation arms. The older A2 / A3 / A7 / A8 ids are not used.
## Weights:
##   data/PanCancer/result_uni2_ablation_<arm>_new1066axisfix_paired_v2_seed42/
##       cross_dataset_cv/<arm>/best_mlp_gpu.pt
##
## UNI2 tokens already live under project_all_UNI2 and are not rewritten.
## Each arm writes:
##   data/Visium/BRCA/pred_uni2_axisfix_<arm>/{sample}/
##
##   CUDA_DEVICE=2 bash code/External_val/BRCA/demo_uni2_axisfix.sh predict
##   CUDA_DEVICE=2 bash code/External_val/BRCA/demo_uni2_axisfix.sh predict spatial_k8
## Notebook: code/External_val/BRCA/Pred_visual_brca_10x_axisfix.ipynb
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"

ARMS=(
  legacy_nospatial
  adjacent_nospatial
  spatial_k8
  spatial_k8_gate
  l12_links_1
  l12_links_2
  l12_balanced
  focal_gamma1
  balanced_sampler
  l12_balanced_nospatial
)
export SAMPLE="${SAMPLE:-V1_Breast_Cancer_Block_A_Section_1_image}"

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
WHICH="${2:-all}"

usage() {
  cat <<EOF
Usage: CUDA_DEVICE=2 bash code/External_val/BRCA/demo_uni2_axisfix.sh {dry-run|predict} [arm|all]

10x slide only: ${SAMPLE}
Arms: ${ARMS[*]}
Output: data/Visium/BRCA/pred_uni2_axisfix_<arm>/${SAMPLE}/
Notebook: code/External_val/BRCA/Pred_visual_brca_10x_axisfix.ipynb
EOF
}

known_arm() {
  local arm="$1" one
  for one in "${ARMS[@]}"; do
    [[ "$one" == "$arm" ]] && return 0
  done
  return 1
}

run_arm() {
  local arm="$1"
  local result="${REPO}/data/PanCancer/result_uni2_ablation_${arm}_new1066axisfix_paired_v2_seed42"
  export UNI2_CKPT="${result}/cross_dataset_cv/${arm}/best_mlp_gpu.pt"
  export UNI2_PRED_DIRNAME="pred_uni2_axisfix_${arm}"
  if [[ ! -f "${UNI2_CKPT}" ]]; then
    echo "ERROR: missing ${UNI2_CKPT}" >&2
    return 1
  fi
  echo "===== ${arm}  ckpt=${UNI2_CKPT}"
  extval_time_step "BRCA-axisfix-${arm}" "${STEP}" \
    "${PYTHON}" -u code/External_val/BRCA/brca_external.py uni2 "${STEP}" \
      --sample "${SAMPLE}"
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi
case "${STEP}" in
  dry-run|predict) ;;
  *) usage; exit 1 ;;
esac

cd "${REPO}"
chosen=()
if [[ "${WHICH}" == all ]]; then
  chosen=("${ARMS[@]}")
else
  known_arm "${WHICH}" || { echo "ERROR: unknown arm ${WHICH}" >&2; usage; exit 1; }
  chosen=("${WHICH}")
fi

LOG="${LOG:-${REPO}/code/log/external_val_brca_uni2_axisfix_10x.log}"
run_all() {
  local arm rc=0
  for arm in "${chosen[@]}"; do
    run_arm "${arm}" || rc=1
  done
  return "${rc}"
}
extval_run_logged "${LOG}" "BRCA-axisfix" "${STEP}" run_all
