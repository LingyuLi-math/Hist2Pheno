#!/usr/bin/env bash
## 2026.11.17 Hist2Pheno as released = mean of the five fold models (optionally of several seeds).
## 发布版 Hist2Pheno = 同一个 run 的 5 个 fold 模型取平均（也可以再跨 seed 平均），不再只用 fold 4。
##
##   cd /home/lingyu/ssd2/Python/Hist2Pheno
##   conda activate SeededNTM
##   # NPC patient 1, cohort_safe l12_links_1, seed 42 (5 models) and seeds 42+44 (10 models)
##   CUDA_DEVICE=2 bash code/External_val/predict_fold_ensemble.sh NPC
##   # BRCA 10x slide (pathologist map), same runs
##   CUDA_DEVICE=2 bash code/External_val/predict_fold_ensemble.sh BRCA
##   # only rebuild the averages from existing member predictions
##   STEP=ensemble bash code/External_val/predict_fold_ensemble.sh NPC
##
## Settings (environment, optional):
##   RUNS="..."       result folders (default: cohort_safe l12_links_1 seeds 42 and 44)
##   ARM=l12_links_1  checkpoint subfolder in cross_dataset_cv/
##   FOLDS="0 1 2 3 4"
##   SAMPLE=...       default: NPC 20210809-C-AH4199551, BRCA V1_Breast_Cancer_Block_A_Section_1_image
##   STAR_PROB_MIN=0  as in the other external scripts; other values get _probX.XX folders
##   STEP=all         predict | ensemble | all
##   CONSISTENCY=l2_parent   pred_sublineage / pred_lineage = ontology parents of pred_CT;
##                    fused = product of the three heads, then parents; independent = old per-head argmax
##   STEP=ensemble CONSISTENCY=fused bash ...   rebuilds the means only, no GPU
## Rebuilding a mean moves what earlier notebook runs wrote next to it (niche h5ad, confidence
## sidecars, label-mode csv, <pred root>/figures/) into <pred root>/_previous_<time>/; nothing is deleted.
##
## Folders under data/Visium/<cohort>/ (one per member, then the averages):
##   pred_uni2_fold<k>_<run>/<sample>/        one fold model (normal verified pipeline)
##   pred_uni2_ens5_<run>/<sample>/           mean of that run's five folds
##   pred_uni2_ens<N>_<name>/<sample>/        mean over all RUNS x FOLDS (only when RUNS has 2+ runs)
## Each average has <sample>_stardist_pred.h5ad, <sample>_l2_probs.npz and
## <sample>_ensemble_report.csv (L2 composition per member and agreement with the mean).
## UNI2 token pooling is done once: the first member's verified feature h5ad is hard-linked
## into the other member folders (fold_ensemble_external.py seed-features).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/log_run.sh"

REPO="${REPO:-/home/lingyu/ssd2/Python/Hist2Pheno}"
cd "${REPO}"
COHORT="${1:-${COHORT:-NPC}}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export STAR_PROB_MIN="${STAR_PROB_MIN:-0}"
PROB_TAG="$(extval_prob_tag)" || { echo "ERROR: STAR_PROB_MIN must be in [0, 1], got ${STAR_PROB_MIN}" >&2; exit 1; }
STEP="${STEP:-all}"
# 2026.11.18, labels follow the ontology: l2_parent (default) | fused | independent
CONSISTENCY="${CONSISTENCY:-l2_parent}"
ARM="${ARM:-l12_links_1}"
read -r -a FOLDS <<< "${FOLDS:-0 1 2 3 4}"
PAN="data/PanCancer"
read -r -a RUNS <<< "${RUNS:-${PAN}/result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42 ${PAN}/result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed44}"

case "${COHORT}" in
  NPC)
    export SAMPLE="${SAMPLE:-20210809-C-AH4199551}"
    DATA="data/Visium/NPC"
    runner() { "${PYTHON}" -u code/Hist2Pheno_pkg/external_validation.py NPC uni2 "$1" --sample "${SAMPLE}"; }
    ;;
  BRCA)
    export SAMPLE="${SAMPLE:-V1_Breast_Cancer_Block_A_Section_1_image}"
    DATA="data/Visium/BRCA"
    runner() { "${PYTHON}" -u code/External_val/BRCA/brca_external.py uni2 "$1" --sample "${SAMPLE}"; }
    ;;
  *) echo "usage: bash code/External_val/predict_fold_ensemble.sh {NPC|BRCA}" >&2; exit 1 ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
  for cand in /nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python /ssd2/users/lingyu/conda_envs/SeededNTM/bin/python; do
    [[ -x "${cand}" ]] && { PYTHON="${cand}"; break; }
  done
fi
PYTHON="${PYTHON:-$(command -v python || true)}"
[[ -x "${PYTHON}" ]] || { echo "ERROR: SeededNTM python not found. Activate the env or set PYTHON=..." >&2; exit 1; }

short() {  # result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42 -> l12_links_1_ivsafe_seed42
  local b; b="$(basename "$1")"; b="${b#result_uni2_ablation_}"
  b="${b/_new1066axisfix/}"; b="${b/_paired_v2/}"; echo "${b}"
}
member_dir() { echo "${DATA}/pred_uni2_fold$2_$(short "$1")${PROB_TAG}/${SAMPLE}"; }

LOG="${LOG:-${REPO}/code/log/external_val_${COHORT,,}_fold_ensemble.log}"

predict_members() {
  local run k first=""
  for run in "${RUNS[@]}"; do
    for k in "${FOLDS[@]}"; do
      local ckpt="${run}/cross_dataset_cv/${ARM}/hce_group_fold_${k}.pt"
      [[ -f "${ckpt}" && -f "${ckpt}.artifact.pt" ]] || { echo "ERROR: missing ${ckpt} (or its .artifact.pt)" >&2; return 1; }
      local dir; dir="$(member_dir "${run}" "${k}")"
      export UNI2_CKPT="${REPO}/${ckpt}"
      export UNI2_PRED_DIRNAME="$(basename "$(dirname "${dir}")")"
      if [[ -n "${first}" ]]; then   # reuse the first member's pooled features
        "${PYTHON}" -u code/External_val/fold_ensemble_external.py seed-features --sample "${SAMPLE}" \
          --members "${first}" "${dir}"
      fi
      extval_time_step "${COHORT}-$(short "${run}")-fold${k}" predict runner predict
      [[ -n "${first}" ]] || first="${dir}"
    done
  done
}

build_ensembles() {
  local run k all=()
  for run in "${RUNS[@]}"; do
    local members=()
    for k in "${FOLDS[@]}"; do members+=("$(member_dir "${run}" "${k}")"); done
    all+=("${members[@]}")
    echo "===== ens${#members[@]} $(short "${run}")"
    "${PYTHON}" -u code/External_val/fold_ensemble_external.py ensemble --sample "${SAMPLE}" \
      --consistency "${CONSISTENCY}" --members "${members[@]}" --out "${DATA}/pred_uni2_ens${#members[@]}_$(short "${run}")${PROB_TAG}/${SAMPLE}"
  done
  if (( ${#RUNS[@]} > 1 )); then
    local name; name="$(short "${RUNS[0]}")"; name="${name%_seed*}_seeds"
    for run in "${RUNS[@]}"; do name="${name}_$(short "${run}" | sed 's/.*_seed//')"; done
    echo "===== ens${#all[@]} ${name}"
    "${PYTHON}" -u code/External_val/fold_ensemble_external.py ensemble --sample "${SAMPLE}" \
      --consistency "${CONSISTENCY}" --members "${all[@]}" --out "${DATA}/pred_uni2_ens${#all[@]}_${name}${PROB_TAG}/${SAMPLE}"
  fi
}

run_all() {
  case "${STEP}" in
    predict) predict_members ;;
    ensemble) build_ensembles ;;
    all) predict_members && build_ensembles ;;
    *) echo "STEP must be predict | ensemble | all" >&2; return 1 ;;
  esac
}
extval_run_logged "${LOG}" "${COHORT}-fold-ensemble" "${STEP}" run_all
