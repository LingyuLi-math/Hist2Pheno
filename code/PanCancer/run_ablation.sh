#!/usr/bin/env bash
## 2026.11.17, UNI2 ablation with descriptive names / 消融实验（用描述性名字，不用 A 编号）
##
## Each arm changes one thing against the trunk `spatial_k8`
## (adjacent loss + k=8 neighbor mean, no gate). Folder names are the arm
## names, e.g. data/PanCancer/result_uni2_ablation_l12_balanced_<DATA_TAG>_paired_v2_seed42.
## All arms share one fold plan (SPLIT_SEED=42), so folds are paired.
##
##   arm                     change                                   recipe in 1_demo_train.sh
##   legacy_nospatial        legacy loss, no neighbors                A0_legacy_nospatial
##   adjacent_nospatial      adjacent loss, no neighbors              A1_adjacent_nospatial
##   spatial_k8   (trunk)    + k=8 neighbor mean                      A2_adjacent_spatial_k8mean
##   spatial_k8_gate         trunk + spatial gate                     A3_adjacent_spatial_k8mean_gate
##   l12_links_1             trunk, L12 link weights 1                A5_l12_links_1
##   l12_links_2             trunk, L12 link weights 2                A6_l12_links_2
##   balanced_sampler        trunk + balanced sampler (no class wt)   balanced_sampler
##   focal_gamma1            trunk + focal γ=1 (class wt kept)        focal_gamma1
##   l12_balanced            trunk + balanced L12                     A2_l12_balanced
##   gate_l12_balanced       gate + balanced L12 (interaction)        A8_l12_balanced
##   l12_balanced_nospatial  adjacent_nospatial + balanced L12        l12_balanced_nospatial
##   spatial_k4              trunk with k=4   (sensitivity)           spatial_k4
##   spatial_k16             trunk with k=16  (sensitivity)           spatial_k16
##
## 2026.11.17, arms stacked on l12_links_1 (compare with REFERENCE=l12_links_1):
##   l12_links_1_focal       l12_links_1 + focal γ=1                  l12_links_1_focal
##   l12_links_1_balanced    l12_links_1 + balanced L12               l12_links_1_balanced
##   l12_links_1p5           link weights 1.5, CE(L12) 1              l12_links_1p5
##   bash code/PanCancer/run_ablation.sh stack        # l12_links_1 + the three above
##
## 2026.11.17, INNER_VAL=cohort_safe: the early-stopping split never takes a
## cohort's last section or an L2 class out of training (BRCA in folds 0/1).
## The tag becomes <DATA_TAG>_ivsafe, so these runs are paired only with each other.
##   INNER_VAL=cohort_safe REFERENCE=l12_links_1 bash code/PanCancer/run_ablation.sh summary
##
## The old "A6" (sampler + focal + no class weight in one arm) is split into
## balanced_sampler and focal_gamma1. After each run 1_demo_train.sh fits the
## decision layer (posthoc.py fit-decision), so balanced-weight arms are also
## compared on cross-fitted Accuracy / macro-F1, not only AUROC.
##
## DATA_TAG names the data version. Use a new tag after the SPORA axis fix so
## the pre-fix folders stay for comparison:
##
##   DATA_TAG=new1066axisfix CUDA_DEVICE=0 bash code/PanCancer/run_ablation.sh spatial_k8
##   DATA_TAG=new1066axisfix CUDA_DEVICE=0 bash code/PanCancer/run_ablation.sh all
##   DATA_TAG=new1066axisfix SEEDS="42 43 44" CUDA_DEVICE=0 bash code/PanCancer/run_ablation.sh spatial_k8 l12_balanced gate_l12_balanced
##   DATA_TAG=new1066axisfix bash code/PanCancer/run_ablation.sh summary     # table, paired against spatial_k8
##   bash code/PanCancer/run_ablation.sh list
##
## Arms run one after another on CUDA_DEVICE. To use several GPUs, start one
## command per GPU with different arm lists, after the first prepare has built
## data/PanCancer/uni2_aligned/. SPLITTER=stratified_section is passed through
## (folders get the stratsec_ prefix in the suffix).
## run_uni2_new1066.sh stays for the finished pre-fix runs only.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
out="${repo}/data/PanCancer"

DATA_TAG="${DATA_TAG:-new1066axisfix}"
if [[ ! "${DATA_TAG}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "DATA_TAG may contain letters, digits and _ only (got ${DATA_TAG})" >&2
  exit 1
fi
# 2026.11.17, cohort_safe inner validation gets its own tag
if [[ "${INNER_VAL:-random}" == cohort_safe ]]; then
  [[ "${DATA_TAG}" == *_ivsafe ]] || DATA_TAG="${DATA_TAG}_ivsafe"
  export INNER_VAL
elif [[ "${INNER_VAL:-random}" != random ]]; then
  echo "INNER_VAL must be random or cohort_safe" >&2; exit 1
fi
if [[ "${DATA_TAG}" == new1066 ]]; then
  echo "DATA_TAG=new1066 is the pre-fix series; use run_uni2_new1066.sh for it, or pick a new tag." >&2
  exit 1
fi

ARMS=(legacy_nospatial adjacent_nospatial spatial_k8 spatial_k8_gate l12_links_1 l12_links_2
      balanced_sampler focal_gamma1 l12_balanced gate_l12_balanced l12_balanced_nospatial)
EXTRA_ARMS=(spatial_k4 spatial_k16)
STACK_ARMS=(l12_links_1 l12_links_1_focal l12_links_1_balanced l12_links_1p5)  # 2026.11.17
REFERENCE="${REFERENCE:-spatial_k8}"

recipe_of() {
  case "$1" in
    legacy_nospatial) echo A0 ;;
    adjacent_nospatial) echo A1 ;;
    spatial_k8) echo A2 ;;
    spatial_k8_gate) echo A3 ;;
    l12_links_1) echo A5 ;;
    l12_links_2) echo A6 ;;
    balanced_sampler) echo balanced_sampler ;;
    focal_gamma1) echo focal_gamma1 ;;
    l12_balanced) echo A2_l12_balanced ;;
    gate_l12_balanced) echo A8 ;;
    l12_balanced_nospatial) echo l12_balanced_nospatial ;;
    spatial_k4) echo spatial_k4 ;;
    spatial_k16) echo spatial_k16 ;;
    l12_links_1_focal|l12_links_1_balanced|l12_links_1p5) echo "$1" ;;
    *) return 1 ;;
  esac
}

usage() {
  echo "usage: DATA_TAG=<tag> [SEEDS=\"42 43 44\"] CUDA_DEVICE=0 bash code/PanCancer/run_ablation.sh {<arm> ...|all|summary|list}" >&2
  echo "arms: ${ARMS[*]} ${EXTRA_ARMS[*]} ${STACK_ARMS[*]:1}" >&2
  exit 1
}

run_arm() {
  local arm="$1" seed="$2" recipe
  recipe="$(recipe_of "$arm")" || { echo "unknown arm: $arm" >&2; usage; }
  echo "===== ${arm} seed ${seed} data ${DATA_TAG} $(date '+%F %T') ====="
  TRAIN_SEED="$seed" SPLIT_SEED="${SPLIT_SEED:-42}" \
  RUN_SUFFIX="${DATA_TAG}_paired_v2_seed${seed}" \
  RESULT_NAME="result_uni2_ablation_${arm}" ABLATION_TAG="${arm}" \
    bash code/PanCancer/1_demo_train.sh uni2 "$recipe"
}

summary() {
  local prefix=""
  [[ "${SPLITTER:-cross_dataset_section}" != cross_dataset_section ]] && prefix="stratsec_"
  conda run --no-capture-output -n "${CONDA_ENV:-SeededNTM}" python -u code/Hist2Pheno_pkg/posthoc.py compare \
    --root "$out" --pattern "result_uni2_ablation_*_${prefix}${DATA_TAG}_paired_v2_seed*" \
    --reference "$REFERENCE" --data-tag "${DATA_TAG}" --out "$out/ablation_${prefix}${DATA_TAG}_summary.csv"
}

[[ $# -ge 1 ]] || usage
case "$1" in
  list) printf '%s\n' "${ARMS[@]}" "${EXTRA_ARMS[@]}" "${STACK_ARMS[@]:1}"; exit 0 ;;
  summary) summary; exit 0 ;;
esac
arms=("$@")
[[ "$1" == all ]] && arms=("${ARMS[@]}")
[[ "$1" == stack ]] && arms=("${STACK_ARMS[@]}")
for arm in "${arms[@]}"; do recipe_of "$arm" >/dev/null || { echo "unknown arm: $arm" >&2; usage; }; done
read -r -a seeds <<< "${SEEDS:-${TRAIN_SEED:-42}}"
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "invalid seed $seed" >&2; exit 1; }
  for arm in "${arms[@]}"; do
    run_arm "$arm" "$seed"
  done
done
