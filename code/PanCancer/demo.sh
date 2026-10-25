#!/usr/bin/env bash
## 2026.10.05 LLY: pan-cancer 7-cohort Hist2Pheno — two independent pipelines
##
##   bash code/PanCancer/demo.sh section  {smoke|train|predict|status}
##   bash code/PanCancer/demo.sh celltype {smoke|train|predict|status}
##   bash code/PanCancer/demo.sh index
##   bash code/PanCancer/demo.sh status
##
## Do not run this file with no args expecting a full train (prints usage and exits).
##
## Ontology: code/Hist2Pheno_pkg/dataset/PanCancerCellType.xlsx  sheet celltype_pancancer
## Heads: L2=final_CT (20) / L12=final_sublineage (8) / L1=final_lineage (5)
##
## Protocols (separate trees; never overwrite each other):
##   section  = cross-section GroupKFold (each HE section wholly train or val)
##              result_all_spatial/          pred_all_spatial/          train.log
##   celltype = cross-celltype StratifiedKFold on L1|L12|L2 (cells mixed)
##              result_cell_stratified_spatial/  pred_cell_stratified_spatial/
##              train_cell_stratified.log
##
## Xenium: Lung, BRCA, CRC (CV train: P1CRC+P2CRC+P5CRC).  ESCC omitted.
## CODEX:  HCC, PDAC c001/c003, GIST c009/c011/c013, GBM.
## Full pool ≈ 5.51M cells × 1024-d float32 (~22 GB). Smoke before full train.
## Resume: train reuses hce_group_fold_*.pt and only fits missing folds.
## A fold killed mid-run is treated as DONE (best-so-far .pt). Delete that file to retrain it.
## Exit 137 = Linux OOM killer. Override GPU with CUDA_DEVICE=1 (default 2).


##################################
# Cross-section CV
##################################
## 2026.10.12 Forget CRC P5, now re-train the model
# bash code/PanCancer/demo.sh section train
## 2026.10.15 Predict with new model
# rm -rf data/PanCancer/pred_all_spatial/stardist_roc \
#        data/PanCancer/pred_all_spatial/pooled_roc \
#        data/PanCancer/pred_all_spatial/metrics_per_section.csv
# bash code/PanCancer/demo.sh section predict

##################################
# Cross-celltype CV
##################################
## 2026.10.15 Predict with new model
## 不要动 result_all_spatial/（那是 cross-section）
# rm -f data/PanCancer/result_cell_stratified_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer_cellstrat/hce_group_fold_*.pt
# rm -f data/PanCancer/result_cell_stratified_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer_cellstrat/best_mlp_gpu.pt

## GPU 2 若还在跑 section repredict，换一张卡
# CUDA_DEVICE=1 bash code/PanCancer/demo.sh celltype train

## 训完再刷预测（覆盖旧 pred）：
# CUDA_DEVICE=1 bash code/PanCancer/demo.sh celltype repredict


set -euo pipefail

# Original: REPO=/home/lingyu/ssd2/Python/Hist2Pheno
# 2026.10.24, portable repository/Python override / 保留原用法并支持 farm
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="$REPO/data/PanCancer"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"

TAG_SECTION=D_emph_L2_spatial_pancancer
TAG_CELL=D_emph_L2_spatial_pancancer_cellstrat
CKPT_SECTION="$OUT/result_all_spatial/cross_dataset_cv/${TAG_SECTION}/best_mlp_gpu.pt"
CKPT_CELL="$OUT/result_cell_stratified_spatial/cross_dataset_cv/${TAG_CELL}/best_mlp_gpu.pt"
FOLD_DIR_SECTION="$OUT/result_all_spatial/cross_dataset_cv/${TAG_SECTION}"
FOLD_DIR_CELL="$OUT/result_cell_stratified_spatial/cross_dataset_cv/${TAG_CELL}"

usage() {
  cat <<EOF
Usage:
  bash code/PanCancer/demo.sh section  {smoke|train|predict|repredict|status}
  bash code/PanCancer/demo.sh celltype {smoke|train|predict|repredict|status}
  bash code/PanCancer/demo.sh index
  bash code/PanCancer/demo.sh status

Protocols (independent output trees):
  section   cross-section CV  → result_all_spatial/ + pred_all_spatial/
  celltype  cross-celltype CV → result_cell_stratified_spatial/ + pred_cell_stratified_spatial/

Steps:
  smoke     BRCA+GBM, 5k cells/section, 2 epochs (own result_smoke* folder)
  train     full 7-cohort prepare + train + he_validate (resume existing fold .pt)
  predict   apply best_mlp_gpu.pt; skip sections that already have a panel PNG
  repredict same as predict but overwrite (use after a new train)
  status    print checkpoints / pred dirs for one or both protocols
  index     scan 7-cohort h5ads → section_index + label_cache (shared)

GPU: CUDA_DEVICE=${CUDA_DEVICE}   env: ${CONDA_ENV}
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_ckpt() {
  local ckpt=$1 proto=$2
  [[ -f "$ckpt" ]] || die "$proto predict needs $ckpt (finish that protocol's train first)"
}

py() {
  if [[ "${HIST2PHENO_USE_CONDA_RUN:-0}" == 1 ]]; then
    conda run --no-capture-output -n "$CONDA_ENV" python -u "$@"
  else
    "${PYTHON:-python}" -u "$@"
  fi
}

cd "$REPO"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------
cmd_index() {
  py code/PanCancer/prepare_pancancer_index.py
}

cmd_status() {
  local proto=${1:-both}
  _status_one() {
    local name=$1 ckpt=$2 fold_dir=$3 pred=$4 log=$5
    local n_fold=0
    [[ -d "$fold_dir" ]] && n_fold=$(find "$fold_dir" -maxdepth 1 -name 'hce_group_fold_*.pt' | wc -l)
    local n_pred=0
    [[ -d "$pred/stardist_roc" ]] && n_pred=$(find "$pred/stardist_roc" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "=== $name ==="
    echo "  best_mlp     $( [[ -f "$ckpt" ]] && echo yes || echo NO )  $ckpt"
    echo "  fold ckpts   ${n_fold}/5  $fold_dir"
    echo "  pred roc     ${n_pred} section dirs  $pred"
    echo "  train log    $( [[ -f "$log" ]] && echo yes || echo no )  $log"
  }
  if [[ "$proto" == both || "$proto" == section ]]; then
    _status_one cross-section "$CKPT_SECTION" "$FOLD_DIR_SECTION" \
      "$OUT/pred_all_spatial" "$OUT/train.log"
  fi
  if [[ "$proto" == both || "$proto" == celltype ]]; then
    _status_one cross-celltype "$CKPT_CELL" "$FOLD_DIR_CELL" \
      "$OUT/pred_cell_stratified_spatial" "$OUT/train_cell_stratified.log"
  fi
}

# ---------------------------------------------------------------------------
# cross-section
# ---------------------------------------------------------------------------
cmd_section_smoke() {
  py code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
    --cuda-device "$CUDA_DEVICE" \
    --group-splitter cross_dataset_section \
    --datasets BRCA GBM \
    --max-cells-per-section 5000 \
    --max-epochs 2 --patience 2 \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --steps prepare train he_validate \
    --save-result result_smoke \
    --ablation-tag smoke_brca_gbm
}

cmd_section_train() {
  local log="$OUT/train.log"
  {
    echo "===== PanCancer cross-section train start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n "$CONDA_ENV" python -u \
        code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
          --cuda-device "$CUDA_DEVICE" \
          --group-splitter cross_dataset_section \
          --use-spatial-context --spatial-k 8 --spatial-mode mean \
          --steps prepare train he_validate \
          --save-result result_all_spatial \
          --ablation-tag "$TAG_SECTION"
    echo "===== PanCancer cross-section train end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$log"
}

cmd_section_predict() {
  need_ckpt "$CKPT_SECTION" section
  mkdir -p "$OUT/pred_all_spatial"
  local logp="$OUT/pred_all_spatial/predict.log"
  local skip=()
  [[ "${1:-skip}" == "skip" ]] && skip=(--skip-existing)
  {
    echo "===== PanCancer cross-section predict start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n "$CONDA_ENV" python -u \
        code/PanCancer/PanCancer_predict_visualize.py \
          --cuda-device "$CUDA_DEVICE" \
          --checkpoint "$CKPT_SECTION" \
          --pred-root "$OUT/pred_all_spatial" \
          --include-incomplete-lung "${skip[@]}"
    echo "===== PanCancer cross-section predict end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$logp"
}

# ---------------------------------------------------------------------------
# cross-celltype
# ---------------------------------------------------------------------------
cmd_celltype_smoke() {
  py code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
    --cuda-device "$CUDA_DEVICE" \
    --group-splitter cell_stratified --inner-val-fraction 0 \
    --datasets BRCA GBM \
    --max-cells-per-section 5000 \
    --max-epochs 2 --patience 2 \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --steps prepare train he_validate \
    --save-result result_smoke_cellstrat \
    --ablation-tag smoke_brca_gbm_cellstrat
}

cmd_celltype_train() {
  ## 2026-10-03: folds 0-3 on disk; fold 4 missing after 137 at fold 3 epoch 31.
  ## Re-run this command: skips 0-3, trains fold 4, then he_validate → best_mlp_gpu.pt
  local logc="$OUT/train_cell_stratified.log"
  {
    echo "===== PanCancer cross-celltype train start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n "$CONDA_ENV" python -u \
        code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
          --cuda-device "$CUDA_DEVICE" \
          --group-splitter cell_stratified --inner-val-fraction 0 \
          --use-spatial-context --spatial-k 8 --spatial-mode mean \
          --steps prepare train he_validate \
          --save-result result_cell_stratified_spatial \
          --ablation-tag "$TAG_CELL"
    echo "===== PanCancer cross-celltype train end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$logc"
}

cmd_celltype_predict() {
  need_ckpt "$CKPT_CELL" celltype
  mkdir -p "$OUT/pred_cell_stratified_spatial"
  local logpc="$OUT/pred_cell_stratified_spatial/predict.log"
  local skip=()
  [[ "${1:-skip}" == "skip" ]] && skip=(--skip-existing)
  {
    echo "===== PanCancer cross-celltype predict start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n "$CONDA_ENV" python -u \
        code/PanCancer/PanCancer_predict_visualize.py \
          --cuda-device "$CUDA_DEVICE" \
          --checkpoint "$CKPT_CELL" --deployment \
          --pred-root "$OUT/pred_cell_stratified_spatial" \
          --scaler-path "$OUT/result_cell_stratified_spatial/scaler.joblib" \
          --include-incomplete-lung "${skip[@]}"
    echo "===== PanCancer cross-celltype predict end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$logpc"
}

# ---------------------------------------------------------------------------
dispatch() {
  local proto=$1 step=${2:-}
  case "$proto" in
    -h|--help|help|usage|"") usage; return 0 ;;
    index|prepare) cmd_index ;;
    status) cmd_status both ;;
    section|cross-section|cross_section)
      case "$step" in
        smoke) cmd_section_smoke ;;
        train) cmd_section_train ;;
        predict) cmd_section_predict skip ;;
        repredict) cmd_section_predict overwrite ;;
        status) cmd_status section ;;
        *) die "section steps: smoke | train | predict | repredict | status" ;;
      esac
      ;;
    celltype|cell-stratified|cell_stratified|cross-celltype|cross_celltype)
      case "$step" in
        smoke) cmd_celltype_smoke ;;
        train) cmd_celltype_train ;;
        predict) cmd_celltype_predict skip ;;
        repredict) cmd_celltype_predict overwrite ;;
        status) cmd_status celltype ;;
        *) die "celltype steps: smoke | train | predict | repredict | status" ;;
      esac
      ;;
    *) die "unknown protocol '$proto' (section | celltype | index | status)" ;;
  esac
}

dispatch "${1:-}" "${2:-}"

## ---------------------------------------------------------------------------
## Optional extras (copy a block; not wired to the dispatcher)
##
## force retrain section (ignore fold ckpts):
# conda run --no-capture-output -n SeededNTM python -u \
#   code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
#     --cuda-device "${CUDA_DEVICE:-2}" --no-resume-from-checkpoints \
#     --group-splitter cross_dataset_section \
#     --use-spatial-context --spatial-k 8 --spatial-mode mean \
#     --steps prepare train he_validate \
#     --save-result result_all_spatial \
#     --ablation-tag D_emph_L2_spatial_pancancer
##
## sklearn GroupKFold (own folder; does not overwrite either protocol above):
# conda run --no-capture-output -n SeededNTM python -u \
#   code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
#     --cuda-device "${CUDA_DEVICE:-2}" --group-splitter group_kfold \
#     --use-spatial-context --spatial-k 8 --spatial-mode mean \
#     --steps prepare train he_validate \
#     --save-result result_group_kfold_spatial \
#     --ablation-tag D_emph_L2_spatial_pancancer_groupkfold
##
## retrain one celltype fold (example fold 3), keep the others:
# rm -f data/PanCancer/result_cell_stratified_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer_cellstrat/hce_group_fold_3.pt
# bash code/PanCancer/demo.sh celltype train
##
## section predict CRC only (writes P5CRC; P1/P2 skip-existing):
# conda run --no-capture-output -n SeededNTM python -u \
#   code/PanCancer/PanCancer_predict_visualize.py \
#     --cuda-device "${CUDA_DEVICE:-2}" \
#     --checkpoint data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt \
#     --pred-root data/PanCancer/pred_all_spatial --datasets CRC --skip-existing
##
## pooled ROC: code/PanCancer/Pred_statistic_visual_pancancer_all.ipynb
##   early cells = pred_all_spatial/; later cells = pred_cell_stratified_spatial/
