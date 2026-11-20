#!/usr/bin/env bash
## Train scripts split out of demo.sh on 2026.10.26.
## One argument runs one job. No argument prints usage and does not train.
##
##   bash code/PanCancer/1_demo_train.sh index
##   bash code/PanCancer/1_demo_train.sh section  {smoke|train}
##   bash code/PanCancer/1_demo_train.sh celltype {smoke|train}
##   bash code/PanCancer/1_demo_train.sh uni2 {smoke|A0|A1|A2|A3|A5|A6|A7|A8}
## The 1,066-section ablation is run_uni2_new1066.sh A0–A8.
## Do not train that index with this file's default suffix.
##   CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A3
##
## Published UNI section/celltype commands pin:
##   --embedder uni --hierarchy legacy --legacy-l2-l1-weight 2 --inner-val-fraction 0
## UNI2 ablations are cross-section, inner-val 0.2, class-weight balanced.
## k=8 attention was removed: the gather is too expensive. k=16 and k=32
## were never commands. Neighbor and tissue auxiliary heads were removed.
## Do not start several uni2 prepares at once. The first one builds
## data/PanCancer/uni2_aligned/; the rest should wait until that cache exists.
## Predict commands are in 3_demo_pred.sh.

########################################################
# 2026.10.26 UNI: cross-section CV
########################################################
# bash code/PanCancer/1_demo_train.sh index
# bash code/PanCancer/1_demo_train.sh section smoke
# bash code/PanCancer/1_demo_train.sh section train
# bash code/PanCancer/1_demo_train.sh celltype smoke
# bash code/PanCancer/1_demo_train.sh celltype train
########################################################
# 2026.10.26 UNI2 ablation: cross-section CV
########################################################
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 smoke
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A0
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A1
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A2
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A3
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A4
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A5
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A6
# CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A7
# 2026.11.17, after training, posthoc.py fit-decision writes decision_rule*.json and
# decision_report.csv into the result folder (FIT_DECISION=0 skips it;
# DECISION_OBJECTIVE=macro_f1|accuracy|f1_acc|balanced_accuracy|weighted_f1).
# For a finished run: python code/Hist2Pheno_pkg/posthoc.py fit-decision data/PanCancer/<result_dir>
# 2026.11.17, the 1,066-section series is code/PanCancer/run_uni2_new1066.sh
# CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A0
# ... through A8. A2 is the model to use. See PanCancer/README.md.


set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="$REPO/data/PanCancer"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
cd "$REPO"
unset CUDA_VISIBLE_DEVICES || true

usage() {
  cat <<EOF
Usage:
  bash code/PanCancer/1_demo_train.sh index
  bash code/PanCancer/1_demo_train.sh section  {smoke|train}
  bash code/PanCancer/1_demo_train.sh celltype {smoke|train}
  bash code/PanCancer/1_demo_train.sh uni2 {smoke|A0|A1|A2|A3|A5|A6|A7|A8}
  bash code/PanCancer/run_uni2_new1066.sh {A0|A1|A2|A3|A4|A5|A6|A7|A8}

GPU: CUDA_DEVICE=${CUDA_DEVICE}   env: ${CONDA_ENV}
Logs: data/PanCancer/train*.log and code/log/pancancer_section_uni2_<arch>.log
Predict: bash code/PanCancer/3_demo_pred.sh
EOF
}

# 2026.10.26, portable timing / 保留 GNU time 指标，无 /usr/bin/time 时仍可运行
run_timed() {
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' "$@"
  else
    local started=$SECONDS
    "$@"
    echo "===== wall $((SECONDS-started))s ====="
  fi
}

die() { echo "ERROR: $*" >&2; exit 1; }

run_py() {
  local log=$1
  local mode=$2
  shift 2
  mkdir -p "$(dirname "$log")"
  {
    echo "===== start $(date '+%F %T') gpu=${CUDA_DEVICE} ====="
    run_timed \
      conda run --no-capture-output -n "$CONDA_ENV" python -u "$@"
    echo "===== end $(date '+%F %T') ====="
  } 2>&1 | if [[ "$mode" == append ]]; then tee -a "$log"; else tee "$log"; fi
}

prepare_uni2_cache() {
  mkdir -p "$REPO/code/log" "$OUT/scaled_cache" "$OUT/context_cache"
  export HIST2PHENO_ARRAY_CACHE="$OUT/scaled_cache"
  export HIST2PHENO_CONTEXT_CACHE="$OUT/context_cache"
}

ADJ=(
  --hierarchy adjacent --legacy-l2-l1-weight 2
  --l2-l12-nll 0.5 --l12-l1-nll 0.5 --l2-l12-js 0.1 --l12-l1-js 0.1
)

cmd_index() {
  conda run --no-capture-output -n "$CONDA_ENV" python -u \
    code/PanCancer/prepare_pancancer_index.py
}

cmd_section_smoke() {
  conda run --no-capture-output -n "$CONDA_ENV" python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --cuda-device "$CUDA_DEVICE" \
      --embedder uni \
      --hierarchy legacy --legacy-l2-l1-weight 2 --inner-val-fraction 0 \
      --group-splitter cross_dataset_section \
      --datasets BRCA GBM --max-cells-per-section 5000 \
      --max-epochs 2 --patience 2 \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_smoke \
      --ablation-tag smoke_brca_gbm
}

cmd_section_train() {
  run_py "$OUT/train.log" append \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --cuda-device "$CUDA_DEVICE" \
      --embedder uni \
      --hierarchy legacy --legacy-l2-l1-weight 2 --inner-val-fraction 0 \
      --group-splitter cross_dataset_section \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer
}

cmd_celltype_smoke() {
  conda run --no-capture-output -n "$CONDA_ENV" python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --cuda-device "$CUDA_DEVICE" \
      --embedder uni \
      --hierarchy legacy --legacy-l2-l1-weight 2 --inner-val-fraction 0 \
      --group-splitter cell_stratified \
      --datasets BRCA GBM --max-cells-per-section 5000 \
      --max-epochs 2 --patience 2 \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_smoke_cellstrat \
      --ablation-tag smoke_brca_gbm_cellstrat
}

cmd_celltype_train() {
  run_py "$OUT/train_cell_stratified.log" append \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --cuda-device "$CUDA_DEVICE" \
      --embedder uni \
      --hierarchy legacy --legacy-l2-l1-weight 2 --inner-val-fraction 0 \
      --group-splitter cell_stratified \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_cell_stratified_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer_cellstrat
}

cmd_uni2() {
  local arch=$1
  prepare_uni2_cache
  local log="$REPO/code/log/pancancer_section_uni2_${arch}.log"
  local -a args=(
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py
    --cuda-device "$CUDA_DEVICE"
    --embedder uni2
    --seed "${TRAIN_SEED:-42}" --split-seed "${SPLIT_SEED:-42}"
    --group-splitter cross_dataset_section
    --inner-val-fraction 0.2
    --class-weight-mode balanced
    --steps prepare train he_validate
  )
  case "$arch" in
    smoke_A2)
      args+=(
        --datasets BRCA GBM --max-cells-per-section 5000
        --cv-k 2 --max-epochs 2 --patience 2
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_smoke_A2
        --ablation-tag smoke_A2
      )
      ;;
    A0_legacy_nospatial)
      args+=(
        --hierarchy legacy --legacy-l2-l1-weight 2
        --save-result result_uni2_A0_legacy_nospatial
        --ablation-tag A0_legacy_nospatial
      )
      ;;
    A1_adjacent_nospatial)
      args+=(
        "${ADJ[@]}"
        --save-result result_uni2_A1_adjacent_nospatial
        --ablation-tag A1_adjacent_nospatial
      )
      ;;
    A2_adjacent_spatial_k8mean)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_A2_adjacent_spatial_k8mean
        --ablation-tag A2_adjacent_spatial_k8mean
      )
      ;;
    A3_adjacent_spatial_k8mean_gate)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean --spatial-gate
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_A3_adjacent_spatial_k8mean_gate
        --ablation-tag A3_adjacent_spatial_k8mean_gate
      )
      ;;
    A4_adjacent_spatial_k8attention)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode attention
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_A4_adjacent_spatial_k8attention
        --ablation-tag A4_adjacent_spatial_k8attention
      )
      ;;
    # Same trunk as A2 (k=8 mean, no gate). L2→L1 NLL stays 2 and both JS stay 0.1.
    # A5 raises the two L12 edges from 0.5 to 1. A6 sets those edges to 2 and CE(L12) to 2.
    A5_l12_links_1)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 1
        --l2-l12-nll 1 --l12-l1-nll 1 --l2-l12-js 0.1 --l12-l1-js 0.1
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_A5_l12_links_1
        --ablation-tag A5_l12_links_1
      )
      ;;
    A6_l12_links_2)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 2
        --l2-l12-nll 2 --l12-l1-nll 2 --l2-l12-js 0.1 --l12-l1-js 0.1
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_A6_l12_links_2
        --ablation-tag A6_l12_links_2
      )
      ;;
    # Same trunk as A2. Sampler replaces class-weighted CE so rare L12
    # (B_Plasma) is seen every batch; focal γ=1 down-weights easy Tumor.
    # Writes result_uni2_A7_bal_focal_paired_v2_seed42/; does not touch A2.
    A7_bal_focal)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --train-balance-sampler
        --train-balance-sampler-mode hierarchical_l1_l12_l2
        --class-weight-mode none
        --l2-focal-gamma 1
        --save-result result_uni2_A7_bal_focal
        --ablation-tag A7_bal_focal
      )
      ;;
    # 2026.11.15, A3 trunk (k=8 mean + gate), all loss weights unchanged.
    # Only CE(L12), NLL(L2→L12) and NLL(L12→L1) become class-weighted (balanced),
    # so the L12 head no longer pulls B_Plasma back to its 3.5% empirical prior.
    A8_l12_balanced)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean --spatial-gate
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --save-result result_uni2_A8_l12_balanced
        --ablation-tag A8_l12_balanced
      )
      ;;
    # 2026.11.16, A8's balanced L12 on the A2 trunk. No spatial gate.
    # Separate result directory from A8 (A3 trunk) and from published A2.
    A2_l12_balanced)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --save-result result_uni2_A2_l12_balanced
        --ablation-tag A2_l12_balanced
      )
      ;;
    # 2026.11.17, recipes for the redesigned ablation (run_ablation.sh).
    # The old A6 (A7_bal_focal) changed three things at once; it is split into
    # balanced_sampler (class weights none) and focal_gamma1 (focal γ=1,
    # balanced class weights kept). l12_balanced_nospatial asks whether balanced L12
    # needs the neighbor context. spatial_k4 / spatial_k16 are the k sensitivity of the trunk.
    balanced_sampler)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --train-balance-sampler
        --train-balance-sampler-mode hierarchical_l1_l12_l2
        --class-weight-mode none
        --save-result result_uni2_balanced_sampler
        --ablation-tag balanced_sampler
      )
      ;;
    focal_gamma1)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --l2-focal-gamma 1
        --save-result result_uni2_focal_gamma1
        --ablation-tag focal_gamma1
      )
      ;;
    # 2026.11.19, l12_links_1 without the k=8 neighbour mean: the reference for the tile-context
    # arms A (uni2_tile_local) and B (uni2_tile_lg), which also run without neighbours.
    # l12_links_1 去掉 k=8 邻居均值；作为 tile-context A / B 的对照。
    l12_links_1_nospatial)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 1
        --l2-l12-nll 1 --l12-l1-nll 1 --l2-l12-js 0.1 --l12-l1-js 0.1
        --save-result result_uni2_l12_links_1_nospatial
        --ablation-tag l12_links_1_nospatial
      )
      ;;
    # 2026.11.17, arms stacked on l12_links_1 (L12 link NLL 1, CE(L12) 1, k=8 mean).
    # Each adds one thing to l12_links_1 and is compared against it (run_ablation.sh, REFERENCE=l12_links_1).
    # 在 l12_links_1 上每次只加一项改动。
    l12_links_1_focal)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 1
        --l2-l12-nll 1 --l12-l1-nll 1 --l2-l12-js 0.1 --l12-l1-js 0.1
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --l2-focal-gamma 1
        --save-result result_uni2_l12_links_1_focal
        --ablation-tag l12_links_1_focal
      )
      ;;
    l12_links_1_balanced)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 1
        --l2-l12-nll 1 --l12-l1-nll 1 --l2-l12-js 0.1 --l12-l1-js 0.1
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --save-result result_uni2_l12_links_1_balanced
        --ablation-tag l12_links_1_balanced
      )
      ;;
    # link weights 1.5 with CE(L12) kept at 1: l12_links_2 changed both at once
    l12_links_1p5)
      args+=(
        --hierarchy adjacent --legacy-l2-l1-weight 2 --l12-ce-weight 1
        --l2-l12-nll 1.5 --l12-l1-nll 1.5 --l2-l12-js 0.1 --l12-l1-js 0.1
        --use-spatial-context --spatial-k 8 --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result result_uni2_l12_links_1p5
        --ablation-tag l12_links_1p5
      )
      ;;
    l12_balanced_nospatial)
      args+=(
        "${ADJ[@]}"
        --l12-class-weight-mode balanced
        --save-result result_uni2_l12_balanced_nospatial
        --ablation-tag l12_balanced_nospatial
      )
      ;;
    spatial_k4|spatial_k16)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k "${arch#spatial_k}" --spatial-mode mean
        --context-cache-dir "$OUT/context_cache"
        --save-result "result_uni2_${arch}"
        --ablation-tag "${arch}"
      )
      ;;
    # 2026.11.15, Hist2Prot neighbour task. A8 + a head that predicts the L2
    # composition of the k=8 spatial neighbours (soft CE, weight 0.2 =
    # Hist2Prot lambda_neighbor). Inference also writes nbr_comp_probs.
    A9_nbr_comp)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean --spatial-gate
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --neighbor-comp-weight 0.2
        --save-result result_uni2_A9_nbr_comp
        --ablation-tag A9_nbr_comp
      )
      ;;
    # 2026.11.15, Hist2Prot topology/density. A9 in µm: per-section mpp map,
    # 50 µm neighbour cap and [log count, log mean distance, log density]
    # geometry. Build the map first: python code/PanCancer/make_mpp_map.py
    A10_nbr_comp_geom)
      [[ -f "$OUT/mpp_map.csv" ]] || die "missing $OUT/mpp_map.csv; run code/PanCancer/make_mpp_map.py"
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean --spatial-gate
        --mpp-map "$OUT/mpp_map.csv" --spatial-radii-um 50 --spatial-geometry
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --neighbor-comp-weight 0.2
        --save-result result_uni2_A10_nbr_comp_geom
        --ablation-tag A10_nbr_comp_geom
      )
      ;;
    # 2026.11.15, Hist2Prot full auxiliary set. A9 + tissue-compartment head
    # (Tumor/Epithelium/Lymphoid/Myeloid/Stroma/Vascular/Mixed from the GT L12
    # mix of the 32 nearest cells; class-weighted CE, weight 0.2 = Hist2Prot
    # lambda_tissue). In the paper removing all auxiliary tasks cost the most.
    A11_nbr_tissue)
      args+=(
        "${ADJ[@]}"
        --use-spatial-context --spatial-k 8 --spatial-mode mean --spatial-gate
        --context-cache-dir "$OUT/context_cache"
        --l12-class-weight-mode balanced
        --neighbor-comp-weight 0.2
        --tissue-comp-weight 0.2 --tissue-comp-k 32
        --save-result result_uni2_A11_nbr_tissue
        --ablation-tag A11_nbr_tissue
      )
      ;;
    *)
      die "uni2 arch must be smoke|A0|A1|A2|A3|A5|A6|A7|A8|A2_l12_balanced|balanced_sampler|focal_gamma1|l12_balanced_nospatial|spatial_k4|spatial_k16|l12_links_1_focal|l12_links_1_balanced|l12_links_1p5|l12_links_1_nospatial"
      ;;
  esac
  # 2026.11.17, RESULT_NAME / ABLATION_TAG rename the output folder and the
  # checkpoint tag (run_ablation.sh uses them so folder names are the
  # descriptive ablation names). Unset keeps the names above.
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ -n "${RESULT_NAME:-}" && "${args[i]}" == --save-result ]]; then args[i+1]="$RESULT_NAME"; fi
    if [[ -n "${ABLATION_TAG:-}" && "${args[i]}" == --ablation-tag ]]; then args[i+1]="$ABLATION_TAG"; fi
  done
  # 2026.11.19, EMBEDDER swaps the feature source (uni2 | uni2_tile_local | uni2_tile_lg; see
  # Hist2Pheno_pkg/tile_context.py). Non-uni2 runs get <embedder>_ in the folder suffix so they
  # never share a folder with the 14 px crop features. 更换特征来源；文件夹名带上 embedder。
  local embedder="${EMBEDDER:-uni2}"
  # 2026.11.20, + uni2_tile_lg_full (local + full 1536-d global, no PCA)
  case "$embedder" in
    uni2|uni2_tile_local|uni2_tile_lg|uni2_tile_lg_full) ;;
    *) die "EMBEDDER must be uni2 | uni2_tile_local | uni2_tile_lg | uni2_tile_lg_full" ;;
  esac
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == --embedder ]]; then args[i+1]="$embedder"; fi
  done
  # 2026.10.26, isolate seed runs / 新实验避免覆盖旧权重及历史结果
  local suffix="${RUN_SUFFIX:-paired_v2_seed${TRAIN_SEED:-42}}"
  # run_ablation.sh names its folders by arm (tile_local, ...), so the embedder tag is only added
  # when RESULT_NAME is not set. run_ablation.sh 的文件夹名已含 arm 名，不再重复加 embedder。
  if [[ "$embedder" != uni2 && -z "${RESULT_NAME:-}" && "$suffix" != *"$embedder"* ]]; then suffix="${embedder}_${suffix}"; fi
  # 2026.11.17, INNER_VAL=cohort_safe: early-stopping split keeps every cohort and L2 class
  # in inner training (pancancer_cv.inner_val_split). Its runs never share a folder with "random".
  if [[ "${INNER_VAL:-random}" == cohort_safe ]]; then
    export HIST2PHENO_INNER_VAL=cohort_safe
    [[ "$suffix" == *ivsafe* ]] || suffix="ivsafe_${suffix}"
  elif [[ "${INNER_VAL:-random}" != random ]]; then
    die "INNER_VAL must be random or cohort_safe"
  fi
  # 2026.11.16, SPLITTER=stratified_section: whole sections, L2-balanced folds.
  # Results go to <save-result>_stratsec_<suffix>, never into the round-robin folder.
  if [[ "${SPLITTER:-cross_dataset_section}" != cross_dataset_section ]]; then
    local -a resplit=()
    for ((i=0; i<${#args[@]}; i++)); do
      if [[ "${args[i]}" == --group-splitter ]]; then i=$((i+1)); continue; fi
      resplit+=("${args[i]}")
    done
    args=("${resplit[@]}" --group-splitter "$SPLITTER")
    suffix="stratsec_${suffix}"
  fi
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == --save-result ]]; then args[i+1]="${args[i+1]}_${suffix}"; fi
  done
  # TRAIN_STEPS replaces the default "prepare train he_validate".
  # ONLY_FOLDS is a space-separated fold list for one GPU in a parallel run.
  if [[ -n "${TRAIN_STEPS:-}" ]]; then
    local -a rebuilt=()
    local skip=0
    for ((i=0; i<${#args[@]}; i++)); do
      if [[ "$skip" -gt 0 ]]; then skip=$((skip - 1)); continue; fi
      if [[ "${args[i]}" == --steps ]]; then
        local -a step_arr=()
        read -r -a step_arr <<< "$TRAIN_STEPS"
        rebuilt+=(--steps "${step_arr[@]}")
        skip=3
        continue
      fi
      rebuilt+=("${args[i]}")
    done
    args=("${rebuilt[@]}")
  fi
  if [[ -n "${ONLY_FOLDS:-}" ]]; then
    local -a fold_arr=()
    read -r -a fold_arr <<< "$ONLY_FOLDS"
    args+=(--only-folds "${fold_arr[@]}")
  fi
  log="${log%.log}_${suffix}.log"
  if [[ -n "${LOG_TAG:-}" ]]; then log="${log%.log}_${LOG_TAG}.log"; fi
  echo "log $log"
  run_py "$log" replace "${args[@]}"
  # 2026.11.17, fit the label decision layer once the full OOF exists
  # 训练（全部 fold）完成后自动拟合决策层；FIT_DECISION=0 关闭
  local save="" j
  for ((j=0; j<${#args[@]}; j++)); do
    if [[ "${args[j]}" == --save-result ]]; then save="${args[j+1]}"; fi
  done
  local rdir="$OUT/$save"
  if [[ "${FIT_DECISION:-1}" == 1 && -n "$save" && -f "$rdir/oof_predictions.npz" ]] \
     && [[ ! -f "$rdir/decision_rule.json" || "$rdir/oof_predictions.npz" -nt "$rdir/decision_rule.json" ]]; then
    run_py "${log%.log}_decision.log" replace code/Hist2Pheno_pkg/posthoc.py fit-decision "$rdir" \
      --objective "${DECISION_OBJECTIVE:-macro_f1}"
  fi
}

case "${1:-}" in
  -h|--help|help|usage|"") usage ;;
  index|prepare) cmd_index ;;
  section|cross-section|cross_section)
    case "${2:-}" in
      smoke) cmd_section_smoke ;;
      train) cmd_section_train ;;
      *) die "section steps: smoke | train" ;;
    esac
    ;;
  celltype|cell-stratified|cell_stratified|cross-celltype|cross_celltype)
    case "${2:-}" in
      smoke) cmd_celltype_smoke ;;
      train) cmd_celltype_train ;;
      *) die "celltype steps: smoke | train" ;;
    esac
    ;;
  uni2)
    case "${2:-}" in
      smoke) cmd_uni2 smoke_A2 ;;
      A0) cmd_uni2 A0_legacy_nospatial ;;
      A1) cmd_uni2 A1_adjacent_nospatial ;;
      A2) cmd_uni2 A2_adjacent_spatial_k8mean ;;
      A3) cmd_uni2 A3_adjacent_spatial_k8mean_gate ;;
      A4) die "A4 attention was removed; the k=8 gather is too expensive" ;;
      A5) cmd_uni2 A5_l12_links_1 ;;
      A6) cmd_uni2 A6_l12_links_2 ;;
      A7) cmd_uni2 A7_bal_focal ;;
      A8) cmd_uni2 A8_l12_balanced ;;
      A2_l12_balanced) cmd_uni2 A2_l12_balanced ;;
      balanced_sampler|focal_gamma1|l12_balanced_nospatial|spatial_k4|spatial_k16|l12_links_1_focal|l12_links_1_balanced|l12_links_1p5|l12_links_1_nospatial) cmd_uni2 "$2" ;;  # 2026.11.17/19, see run_ablation.sh
      A2L12) die "renamed to A7. CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A7" ;;
      A9|A10|A11) die "${2} was removed. The neighbor and tissue heads did not improve L2 macro-F1." ;;
      *) die "uni2 steps: smoke | A0 | A1 | A2 | A3 | A5 | A6 | A7 | A8" ;;
    esac
    ;;
  *) die "unknown command '$1' (index | section | celltype | uni2)" ;;
esac
