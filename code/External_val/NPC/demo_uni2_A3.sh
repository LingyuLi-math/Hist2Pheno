#!/usr/bin/env bash
## 2026.10.27 NPC external validation — this retrain's UNI2 A3 best_mlp
##
## Weight: result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42
##         cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt
## That file is fold 0 (best three-tier sum). Pooled OOF L2 macro AUROC was 0.8897.
## Spatial k=8 mean with gate. The checkpoint metadata supplies the neighbor config.
##
## 2026.10.27 STAR_PROB_MIN=0.40, the valley cutoff. UNI2 A2 is demo_uni2.sh.
## 2026.10.28 the output folder follows STAR_PROB_MIN (0.40 -> ..._prob0.40).
## STAR_PROB_MIN=0 (default since 2026.11.15) keeps the unsuffixed unfiltered tree.
##
## Does not rewrite:
##   data/Visium/NPC/project_all_UNI2
##
##   bash code/External_val/NPC/demo_uni2_A3.sh dry-run
##   CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A3.sh all
##   CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A3.sh predict
## One slide: SAMPLE=20210809-C-AH4199551
## Redo a finished prediction: FORCE=1
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt"
# 2026.11.15, default 0 like demo_uni2.sh / A5 / A6 and the NPC notebooks
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
# 2026.10.28, one directory per cutoff / 0.40 写入 ..._prob0.40
PROB_TAG="$(awk -v v="${STAR_PROB_MIN}" 'BEGIN {
  if (v !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/) exit 2
  x = v + 0
  if (x < 0 || x > 1) exit 3
  if (x == 0) exit 0
  printf "_prob%.2f", x
}')" || {
  echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2
  exit 1
}
export UNI2_PRED_DIRNAME="pred_uni2_A3_bestmlp_paired_v2_seed42${PROB_TAG}"

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
Usage: bash code/External_val/NPC/demo_uni2_A3.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: \${UNI2_CKPT}
Outputs:
  data/Visium/NPC/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/NPC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/NPC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_l2_probs.npz

Notebook: code/External_val/NPC/Pred_visual_npc_uni2_A3.ipynb
Log: code/log/external_val_npc_uni2_A3_paired_v2_seed42.log
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "NPC-UNI2-A3" "$1" \
    "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py NPC uni2 "$1" \
      "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}"
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

extval_run_logged "${LOG:-${REPO}/code/log/external_val_npc_uni2_A3_paired_v2_seed42.log}" "NPC-UNI2-A3" "${STEP}" run_steps
