#!/usr/bin/env bash
## 2026.11.19, tile-context ablation A / B / C (Hist2Pheno_pkg/tile_context.py)
## 以细胞为中心的多尺度特征消融：先提特征，再训练 4 个 arm，最后出两张对照表。
##
##   cd /home/lingyu/ssd2/Python/Hist2Pheno
##   conda activate SeededNTM
##   bash code/PanCancer/run_tile_context.sh pilot      # BRCA + GBM only: extract, PCA, check speed
##   bash code/PanCancer/run_tile_context.sh extract    # all sections, one shard per GPU
##   bash code/PanCancer/run_tile_context.sh pca        # PCA of the global (tile class) feature
##   bash code/PanCancer/run_tile_context.sh train      # 4 arms on the GPUs (retrain_all.sh)
##   bash code/PanCancer/run_tile_context.sh summary    # A,B vs l12_links_1_nospatial; C vs l12_links_1
##   bash code/PanCancer/run_tile_context.sh all        # extract -> pca -> train -> summary
##
## Arms (same losses as l12_links_1; only the input features change):
##   l12_links_1_nospatial   14 px crop class token, no neighbours     reference for A, B
##   tile_local              A  centroid patch token of 224 px tiles    1536-d
##   tile_local_global       B  A + tile class token (PCA 256)          1792-d
##   tile_local_global_k8    C  B + k=8 neighbour mean                  reference: l12_links_1
##
## Settings (environment):
##   GPUS="0 1 2 3"          extraction: one shard per entry; training: one queue per entry
##   INNER_VAL=cohort_safe   default; pairs with the existing l12_links_1 ivsafe seed runs
##   SEEDS="42"              training seeds (folds fixed by SPLIT_SEED=42)
##   HIST2PHENO_TILE_GLOBAL_DIM=256   global PCA size (0 = full 1536-d class token)
##   DATA_TAG=new1066axisfix
## Disk: data/PanCancer/tile_context holds 2 x 1536 float16 per nucleus (~46 GB for 7.46M nuclei);
## each trained feature set adds one shared pooled .npy (float32) under the cache.
##
## ---------------------------------------------------------------------------------------------
## 2026.11.20 update (this block replaces the arm list above; the old lines are kept for history)
##   bash code/PanCancer/run_tile_context.sh audit      # µm/px of every section; stops on a mismatch
##   bash code/PanCancer/run_tile_context.sh spora      # SPORA 14 px crops re-extracted at 0.5 µm/px
##   bash code/PanCancer/run_tile_context.sh pilot      # audit + BRCA + GBM (incl. SPORA glioma) features
##   bash code/PanCancer/run_tile_context.sh extract    # all sections
##   bash code/PanCancer/run_tile_context.sh train      # 4 tile arms + l12_links_1_mpp05
##   bash code/PanCancer/run_tile_context.sh summary    # every arm vs l12_links_1_mpp05 and vs l12_links_1
##   bash code/PanCancer/run_tile_context.sh all        # audit -> spora -> extract -> train -> summary
##
## Arms (l12_links_1 losses; seed 42, SPLIT_SEED=42 cross_dataset_section folds, INNER_VAL=cohort_safe,
## the same folds as result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42):
##   tile_local              A  centroid patch token, 1536-d, no neighbours
##   tile_local_global       B  A + tile class token, PCA 256 fitted per fold on training rows, 1792-d
##   tile_local_global_full  B without PCA: A + full 1536-d tile class token, 3072-d
##   tile_local_global_k8    C  B + k=8 neighbour mean
##   l12_links_1_mpp05       reference: l12_links_1 (14 px crop, k=8) after the SPORA crops are at 0.5 µm/px
##   l12_links_1_nospatial   dropped (recipe kept in 1_demo_train.sh)
## Every cohort is at 0.5 µm/px: tile features resize each section by mpp/0.5 (SPORA x2), and
## `spora` re-extracts the SPORA crop tokens at x2 so the reference is at 0.5 µm/px as well.
##   HIST2PHENO_TILE_PCA=fold   default; global = one PCA over all sections (fit-pca, 2026.11.19 behaviour)
##
## 2026.11.20 (2), L2-balanced section folds (SPLITTER=stratified_section, default here):
##   whole sections are held out, and sections of the same cohort are swapped between folds until
##   every fold holds ~1/5 of every L2 class (cohort counts and cell counts per fold stay balanced).
##   BRCA has 2 sections, so DCIS is in 2 test folds only (unavoidable).
##   `plan` builds the folds once (prepare of l12_links_1_mpp05) and saves
##   data/PanCancer/fold_plan_stratsec_<tag>.csv; every arm then reads it (HIST2PHENO_FOLD_PLAN),
##   so all 5 arms have identical folds. Folders get stratsec_ in the name. These folds differ from
##   the round-robin folds of the old l12_links_1 run, so that run is no longer fold-paired.
##   SPLITTER=cross_dataset_section gives the previous round-robin folds.
##   bash code/PanCancer/run_tile_context.sh plan
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
export INNER_VAL="${INNER_VAL:-cohort_safe}"
export DATA_TAG="${DATA_TAG:-new1066axisfix}"
export SEEDS="${SEEDS:-42}"
export HIST2PHENO_TILE_GLOBAL_DIM="${HIST2PHENO_TILE_GLOBAL_DIM:-256}"
OUT="$REPO/data/PanCancer"
LOGDIR="$REPO/code/log/tile_context"
mkdir -p "$LOGDIR"
TILE_ARMS="l12_links_1_nospatial tile_local tile_local_global tile_local_global_k8"
# 2026.11.20, no l12_links_1_nospatial; + tile_local_global_full; + the SPORA-0.5 µm reference
TILE_ARMS="tile_local_global_k8 l12_links_1_mpp05 tile_local tile_local_global tile_local_global_full"
export HIST2PHENO_TILE_PCA="${HIST2PHENO_TILE_PCA:-fold}"
# 2026.11.20 (2), L2-balanced section folds by default; one plan shared by all arms
export SPLITTER="${SPLITTER:-stratified_section}"
TAG="$DATA_TAG"; [[ "$INNER_VAL" == cohort_safe ]] && TAG="${DATA_TAG}_ivsafe"
FIRST_SEED="${SEEDS%% *}"
SHARED_PLAN="$OUT/fold_plan_stratsec_${TAG}.csv"

py() { conda run --no-capture-output -n "$CONDA_ENV" python -u "$@"; }
die() { echo "ERROR: $*" >&2; exit 1; }

need_mpp() {
  if [[ ! -f "$OUT/mpp_map.csv" ]]; then
    echo "building $OUT/mpp_map.csv (native µm/px per section)"
    py code/PanCancer/make_mpp_map.py
  fi
}

extract() {  # extract [dataset ...]
  need_mpp
  local -a ds=()
  [[ $# -gt 0 ]] && ds=(--datasets "$@")
  local n=${#GPUS[@]} k pids=()
  for ((k=0; k<n; k++)); do
    CUDA_VISIBLE_DEVICES="${GPUS[k]}" py code/Hist2Pheno_pkg/tile_context.py extract "${ds[@]}" \
      --shard "$k/$n" > "$LOGDIR/extract_shard${k}.log" 2>&1 &
    pids+=($!)
    echo "extract shard $k/$n on GPU ${GPUS[k]} -> $LOGDIR/extract_shard${k}.log"
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  py code/Hist2Pheno_pkg/tile_context.py status "${ds[@]}" || true
  [[ $rc == 0 ]] || die "some sections failed; see $LOGDIR/extract_shard*.log"
}

# 2026.11.20, scale audit (no GPU): mpp_map vs h5ad uns['mpp'] vs UNI2 manifest, nuclei inside the image
audit() {
  need_mpp
  local -a ds=()
  [[ $# -gt 0 ]] && ds=(--datasets "$@")
  py code/Hist2Pheno_pkg/tile_context.py audit-mpp "${ds[@]}" 2>&1 | tee "$LOGDIR/mpp_audit.log" \
    || die "scale audit failed; see $OUT/tile_context/mpp_audit.csv"
}

# 2026.11.20, SPORA 14 px crops at 0.5 µm/px (old 1 µm/px stores are archived, not deleted)
spora() {
  GPUS="${GPUS[*]}" bash code/spora/run_spora_pancancer.sh uni2-mpp05 2>&1 | tee "$LOGDIR/spora_uni2_mpp05.log"
}

pca() {
  # 2026.11.20, the default fold PCA is fitted during training; fit-pca only for HIST2PHENO_TILE_PCA=global
  [[ "$HIST2PHENO_TILE_PCA" == fold ]] && { echo "HIST2PHENO_TILE_PCA=fold: PCA is fitted per fold in training"; return; }
  [[ "$HIST2PHENO_TILE_GLOBAL_DIM" == 0 ]] && { echo "global dim 0: no PCA"; return; }
  py code/Hist2Pheno_pkg/tile_context.py fit-pca --dim "$HIST2PHENO_TILE_GLOBAL_DIM" 2>&1 | tee "$LOGDIR/pca.log"
}

# 2026.11.20 (2), build the L2-balanced fold plan once (prepare only, no training)
plan() {
  [[ "$SPLITTER" == stratified_section ]] || { echo "SPLITTER=$SPLITTER: round-robin folds, no shared plan"; return; }
  if [[ -f "$SHARED_PLAN" ]]; then echo "fold plan exists: $SHARED_PLAN"; return; fi
  spora_ok || die "SPORA crop tokens are not at 0.5 µm/px yet; run: bash code/PanCancer/run_tile_context.sh spora"
  CUDA_DEVICE="${GPUS[0]}" TRAIN_SEED="$FIRST_SEED" TRAIN_STEPS="prepare" FIT_DECISION=0 \
    bash code/PanCancer/run_ablation.sh l12_links_1_mpp05 2>&1 | tee "$LOGDIR/fold_plan.log"
  local src="$OUT/result_uni2_ablation_l12_links_1_mpp05_stratsec_${TAG}_paired_v2_seed${FIRST_SEED}/fold_plan.csv"
  [[ -f "$src" ]] || die "no fold plan at $src; see $LOGDIR/fold_plan.log"
  cp "$src" "$SHARED_PLAN"
  echo "shared fold plan: $SHARED_PLAN"
  py -c '
import sys, pandas as pd
p = pd.read_csv(sys.argv[1])
print(pd.crosstab(p["fold"], p["dataset"]))
' "$SHARED_PLAN" || true
  local bal="${src%/fold_plan.csv}/fold_plan_l2_balance.csv"
  if [[ -f "$bal" ]]; then cp "$bal" "${SHARED_PLAN%.csv}_l2_balance.csv"; fi
}

train() {
  # STEP=train ARMS="$TILE_ARMS" GPUS="${GPUS[*]}" REFERENCE=l12_links_1_nospatial \
  #   bash code/PanCancer/retrain_all.sh
  # 2026.11.20, reference l12_links_1_mpp05; the SPORA crops must already be at 0.5 µm/px
  spora_ok || die "SPORA crop tokens are not at 0.5 µm/px yet; run: bash code/PanCancer/run_tile_context.sh spora"
  # 2026.11.20 (2), every arm reads the same stratified fold plan
  if [[ "$SPLITTER" == stratified_section ]]; then
    [[ -f "$SHARED_PLAN" ]] || plan
    export HIST2PHENO_FOLD_PLAN="$SHARED_PLAN"
  fi
  STEP=train ARMS="$TILE_ARMS" GPUS="${GPUS[*]}" REFERENCE=l12_links_1_mpp05 \
    bash code/PanCancer/retrain_all.sh
}

# 2026.11.20, every SPORA UNI2 store cut at scale 2 (0.5 µm/px)
spora_ok() {
  # python -c, not a heredoc: conda run does not always forward stdin
  py -c '
import json, sys
from pathlib import Path
stores = sorted(Path(sys.argv[1]).glob("*/*/*_project_all_UNI2/ImgEmbeddings_all/sc_pth_14_14/embedding_manifest.json"))
bad = []
for m in stores:
    meta = json.loads(m.read_text())
    if meta.get("state") != "completed" or not meta.get("scale_image") or abs(float(meta.get("scale", 0)) - 2.0) > 1e-6:
        bad.append(m.parent.parent.parent.name)
print(f"SPORA UNI2 stores: {len(stores)}, not at 0.5 µm/px: {len(bad)} {bad[:5]}")
sys.exit(1 if bad or not stores else 0)
' "$REPO/data/CODEX/spora/prepared"
}

summary() {
  local tag="$DATA_TAG"
  [[ "$INNER_VAL" == cohort_safe ]] && tag="${DATA_TAG}_ivsafe"
  local pat="result_uni2_ablation_*_${tag}_paired_v2_seed*"
  # 2026.11.20 (2), stratified folds: folders are <arm>_stratsec_<tag>_...
  [[ "$SPLITTER" == stratified_section ]] && pat="result_uni2_ablation_*_stratsec_${tag}_paired_v2_seed*"
  # 2026.11.20, no nospatial reference any more; pair against the SPORA-0.5 µm l12_links_1 and the old one
  # py code/Hist2Pheno_pkg/posthoc.py compare --root "$OUT" --pattern "$pat" --data-tag "$tag" \
  #   --reference l12_links_1_nospatial --out "$OUT/tile_context_${tag}_vs_nospatial.csv"
  py code/Hist2Pheno_pkg/posthoc.py compare --root "$OUT" --pattern "$pat" --data-tag "$tag" \
    --reference l12_links_1_mpp05 --out "$OUT/tile_context_${tag}_vs_l12_links_1_mpp05.csv"
  # 2026.11.20 (2), the old l12_links_1 used round-robin folds: only fold-paired when SPLITTER is the same
  if [[ "$SPLITTER" != stratified_section ]]; then
  py code/Hist2Pheno_pkg/posthoc.py compare --root "$OUT" --pattern "$pat" --data-tag "$tag" \
    --reference l12_links_1 --out "$OUT/tile_context_${tag}_vs_l12_links_1.csv"
  fi
  echo "main table: $OUT/tile_context_${tag}_vs_l12_links_1_mpp05.csv (all cohorts at 0.5 µm/px)"
  if [[ "$SPLITTER" == stratified_section ]]; then echo "folds: $SHARED_PLAN (L2 balance: ${SHARED_PLAN%.csv}_l2_balance.csv)"; fi
  echo "C vs B = effect of k=8; B vs B_full = effect of the PCA; every arm vs l12_links_1_mpp05 = effect of the tile features"
}

case "${1:-}" in
  pilot)
    # 2026.11.20, audit first; GBM includes the SPORA glioma sections (x2 resize)
    audit
    extract BRCA GBM
    echo "pilot: features for BRCA + GBM; check $LOGDIR/extract_shard*.log for seconds per section" ;;
  audit) shift; audit "$@" ;;
  spora) spora ;;
  extract) shift; extract "$@" ;;
  pca) pca ;;
  train) train ;;
  summary) summary ;;
  # all) extract; pca; train; summary ;;
  # all) audit; spora; extract; pca; train; summary ;;  # 2026.11.20
  plan) plan ;;  # 2026.11.20 (2)
  all) audit; spora; plan; extract; pca; train; summary ;;  # 2026.11.20 (2), + plan
  *) die "usage: bash code/PanCancer/run_tile_context.sh {audit|spora|plan|pilot|extract [datasets]|pca|train|summary|all}" ;;
esac
