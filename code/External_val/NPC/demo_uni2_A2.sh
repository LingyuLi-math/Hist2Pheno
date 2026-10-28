#!/usr/bin/env bash
## 2026.10.27 NPC external validation — this retrain's UNI2 A2 best_mlp
##
## Weight: result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42
##         cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt
## That file is fold 1 (best three-tier sum). Pooled OOF L2 macro AUROC was 0.8894.
## Spatial k=8 mean, no gate. The checkpoint metadata supplies the neighbor config.
##
## 2026.10.27 STAR_PROB_MIN=0.40. Both slides' probability histogram is flat
## from about 0.30 to 0.40 and rises after that, so 0.40 is the right edge of
## the valley between the prob=0.01 tail and the nucleus mode.
## 2026.10.28 the output folder follows STAR_PROB_MIN (0.40 -> ..._prob0.40).
## STAR_PROB_MIN=0 keeps the unsuffixed unfiltered tree.
##
## Does not rewrite:
##   data/Visium/NPC/project_all_UNI2
##
##   bash code/External_val/NPC/demo_uni2_A2.sh dry-run
##   CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A2.sh predict
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

export UNI2_CKPT="${REPO}/data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0.40}"
# 2026.10.28, one directory per cutoff / 默认 0.40 仍是 ..._prob0.40
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
export UNI2_PRED_DIRNAME="pred_uni2_A2_bestmlp_paired_v2_seed42${PROB_TAG}"

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
Usage: bash code/External_val/NPC/demo_uni2_A2.sh {dry-run|features|h5ad|predict|probs|plot|all}

Checkpoint: \${UNI2_CKPT}
Outputs:
  data/Visium/NPC/project_all_UNI2/{sample}/ImgEmbeddings_all_stardist/sc_pth_14_14/
  data/Visium/NPC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_stardist_pred.h5ad
  data/Visium/NPC/${UNI2_PRED_DIRNAME}/{sample}/{sample}_l2_probs.npz

Notebook: code/External_val/NPC/Pred_visual_npc_uni2_A2.ipynb
Log: code/log/external_val_npc_uni2_A2_paired_v2_seed42.log
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"

run_step() {
  extval_time_step "NPC-UNI2-A2" "$1" \
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

extval_run_logged "${LOG:-${REPO}/code/log/external_val_npc_uni2_A2_paired_v2_seed42.log}" "NPC-UNI2-A2" "${STEP}" run_steps
