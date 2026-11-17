#!/usr/bin/env bash
## NPC patient 1 — finished 2026-11-17 ablation weights.
##
## Names are the ablation arms. The older A0–A8 ids are not used here.
## Every arm was trained on the same 1,066 sections (7,461,039 cells), bf16,
## fold plan seed 42, and selected fold 4. Weights:
##
##   data/PanCancer/result_uni2_ablation_<arm>_new1066axisfix_paired_v2_seed42/
##       cross_dataset_cv/<arm>/best_mlp_gpu.pt
##
##   legacy_nospatial         old loss, no neighbors          L2 AUROC 0.8820
##   adjacent_nospatial       adjacent loss, no neighbors     0.8827
##   spatial_k8               trunk: adjacent + k=8 mean      0.9044
##   spatial_k8_gate          trunk + spatial gate            0.9052
##   l12_links_1              trunk, L12 link weights = 1     0.9060
##   l12_links_2              trunk, L12 link weights = 2     0.9060
##   l12_balanced             trunk + balanced L12            0.9046
##   focal_gamma1             trunk + focal γ=1               0.9047
##   balanced_sampler         trunk + sampler, no class weight 0.8864
##   l12_balanced_nospatial   adjacent loss + balanced L12, no neighbors  0.8791
##
## Still training, not in this list: gate_l12_balanced.
##
## Patient 1 only (20210809-C-AH4199551). The second slide is not predicted.
## UNI2 tokens already live under project_all_UNI2 and are not rewritten.
## Each arm writes its own tree:
##   data/Visium/NPC/pred_uni2_axisfix_<arm>[_probX.XX]/{sample}/
## STAR_PROB_MIN=0 keeps every StarDist nucleus.
##
##   CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh predict
##   CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh predict spatial_k8
## Notebook: code/External_val/NPC/Pred_visual_npc_patient1_axisfix.ipynb
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
# 2026.11.17, STAR_PROB_MIN other than 0 gets its own prediction folder (_probX.XX), as in demo_uni2_A7_stratsec.sh
PROB_TAG="$(extval_prob_tag)" || {
  echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2
  exit 1
}

# Order matches the ablation table: no-neighbor controls, then the spatial trunk and its one-change arms.
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
# This comparison is patient 1. Override only if a second slide is requested on purpose.
export SAMPLE="${SAMPLE:-20210809-C-AH4199551}"

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
Usage: CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh {dry-run|predict} [arm|all]

Patient 1 only: ${SAMPLE}
Arms: ${ARMS[*]}
Output: data/Visium/NPC/pred_uni2_axisfix_<arm>[_probX.XX]/${SAMPLE}/
Notebook: code/External_val/NPC/Pred_visual_npc_patient1_axisfix.ipynb
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
  export UNI2_PRED_DIRNAME="pred_uni2_axisfix_${arm}${PROB_TAG}"
  if [[ ! -f "${UNI2_CKPT}" ]]; then
    echo "ERROR: missing ${UNI2_CKPT}" >&2
    return 1
  fi
  echo "===== ${arm}  ckpt=${UNI2_CKPT}"
  echo "===== out=data/Visium/NPC/${UNI2_PRED_DIRNAME}/${SAMPLE}"
  extval_time_step "NPC-axisfix-${arm}" "${STEP}" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py NPC uni2 "${STEP}" \
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

LOG="${LOG:-${REPO}/code/log/external_val_npc_uni2_axisfix_patient1.log}"
run_all() {
  local arm rc=0
  for arm in "${chosen[@]}"; do
    run_arm "${arm}" || rc=1
  done
  return "${rc}"
}
extval_run_logged "${LOG}" "NPC-axisfix" "${STEP}" run_all
