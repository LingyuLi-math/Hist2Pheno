#!/usr/bin/env bash
## Predict scripts split out of demo.sh on 2026.10.26.
## One argument runs one job. No argument prints usage and does not predict.
##
##   bash code/PanCancer/3_demo_pred.sh section  {predict|repredict|crc}
##   bash code/PanCancer/3_demo_pred.sh celltype {predict|repredict}
##   bash code/PanCancer/3_demo_pred.sh uni2 {A0|A1|A3}
##   CUDA_DEVICE=0 bash code/PanCancer/3_demo_pred.sh uni2 A3
##
## section/celltype use the published UNI checkpoints.
## uni2 A0 and A1 are OOF: each section is scored by the fold that held it out.
## uni2 A3 is deployment: best_mlp_gpu.pt (fold 0) scores every training section.
## That number is not the five-model pooled OOF. Figures are skipped; ROC CSVs
## are written from matched GT cells. All-nuclei StarDist maps need UNI2 tokens
## under ImgEmbeddings_all_stardist/sc_pth_14_14, which are not required here.
## Train commands are in 1_demo_train.sh.

########################################################
# 2026.10.26 Cross-section CV Predict
########################################################
# bash code/PanCancer/3_demo_pred.sh section predict
# bash code/PanCancer/3_demo_pred.sh section repredict
# bash code/PanCancer/3_demo_pred.sh section crc
# bash code/PanCancer/3_demo_pred.sh celltype predict
# bash code/PanCancer/3_demo_pred.sh celltype repredict
# bash code/PanCancer/3_demo_pred.sh uni2 A0
# bash code/PanCancer/3_demo_pred.sh uni2 A1
# bash code/PanCancer/3_demo_pred.sh uni2 A3     # use -- LLY


set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="$REPO/data/PanCancer"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
cd "$REPO"
unset CUDA_VISIBLE_DEVICES || true

TAG_SECTION=D_emph_L2_spatial_pancancer
TAG_CELL=D_emph_L2_spatial_pancancer_cellstrat
CKPT_SECTION="$OUT/result_all_spatial/cross_dataset_cv/${TAG_SECTION}/best_mlp_gpu.pt"
CKPT_CELL="$OUT/result_cell_stratified_spatial/cross_dataset_cv/${TAG_CELL}/best_mlp_gpu.pt"

usage() {
  cat <<EOF
Usage:
  bash code/PanCancer/3_demo_pred.sh section  {predict|repredict|crc}
  bash code/PanCancer/3_demo_pred.sh celltype {predict|repredict}
  bash code/PanCancer/3_demo_pred.sh uni2 {A0|A1|A3}

GPU: CUDA_DEVICE=${CUDA_DEVICE}   env: ${CONDA_ENV}
predict skips sections that already have a panel (published) or a run stamp (UNI2).
repredict overwrites. crc rewrites only CRC into pred_all_spatial/.
Train: bash code/PanCancer/1_demo_train.sh
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_ckpt() {
  [[ -f "$1" ]] || die "missing checkpoint $1"
}

run_pred() {
  local log=$1
  shift
  mkdir -p "$(dirname "$log")" "$OUT/context_cache"
  export HIST2PHENO_CONTEXT_CACHE="$OUT/context_cache"
  {
    echo "===== start $(date '+%F %T') gpu=${CUDA_DEVICE} ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n "$CONDA_ENV" python -u \
        code/PanCancer/PanCancer_predict_visualize.py \
          --cuda-device "$CUDA_DEVICE" \
          "$@"
    echo "===== end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$log"
}

cmd_section() {
  local mode=$1
  need_ckpt "$CKPT_SECTION"
  local -a skip=()
  [[ "$mode" == skip ]] && skip=(--skip-existing)
  run_pred "$OUT/pred_all_spatial/predict.log" \
    --checkpoint "$CKPT_SECTION" \
    --pred-root "$OUT/pred_all_spatial" \
    --include-incomplete-lung \
    "${skip[@]}"
}

cmd_section_crc() {
  need_ckpt "$CKPT_SECTION"
  run_pred "$OUT/pred_all_spatial/predict_crc.log" \
    --checkpoint "$CKPT_SECTION" \
    --pred-root "$OUT/pred_all_spatial" \
    --datasets CRC \
    --skip-existing
}

cmd_celltype() {
  local mode=$1
  need_ckpt "$CKPT_CELL"
  local -a skip=()
  [[ "$mode" == skip ]] && skip=(--skip-existing)
  run_pred "$OUT/pred_cell_stratified_spatial/predict.log" \
    --checkpoint "$CKPT_CELL" \
    --deployment \
    --pred-root "$OUT/pred_cell_stratified_spatial" \
    --scaler-path "$OUT/result_cell_stratified_spatial/scaler.joblib" \
    --include-incomplete-lung \
    "${skip[@]}"
}

cmd_uni2() {
  local which=$1
  local ckpt root log
  local -a extra=()
  case "$which" in
    A0)
      ckpt="$OUT/result_uni2_A0_legacy_nospatial/cross_dataset_cv/A0_legacy_nospatial/best_mlp_gpu.pt"
      root="$OUT/pred_uni2_A0_legacy_nospatial"
      log="$REPO/code/log/pancancer_section_uni2_A0_legacy_nospatial_predict.log"
      extra=(--embedder uni2 --no-spatial-context --skip-figures --skip-existing)
      ;;
    A1)
      ckpt="$OUT/result_uni2_A1_adjacent_nospatial/cross_dataset_cv/A1_adjacent_nospatial/best_mlp_gpu.pt"
      root="$OUT/pred_uni2_A1_adjacent_nospatial"
      log="$REPO/code/log/pancancer_section_uni2_A1_adjacent_nospatial_predict.log"
      extra=(--embedder uni2 --no-spatial-context --skip-figures --skip-existing)
      ;;
    A3)
      ckpt="$OUT/result_uni2_A3_adjacent_spatial_k8mean_gate/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt"
      root="$OUT/pred_uni2_A3_bestmlp"
      log="$REPO/code/log/pancancer_section_uni2_A3_bestmlp_predict.log"
      extra=(--embedder uni2 --deployment --spatial-k 8 --skip-figures --skip-existing)
      ;;
    *)
      die "uni2 predict must be A0 | A1 | A3"
      ;;
  esac
  need_ckpt "$ckpt"
  echo "log $log"
  echo "pred $root"
  run_pred "$log" --checkpoint "$ckpt" --pred-root "$root" "${extra[@]}"
}

case "${1:-}" in
  -h|--help|help|usage|"") usage ;;
  section|cross-section|cross_section)
    case "${2:-}" in
      predict) cmd_section skip ;;
      repredict) cmd_section overwrite ;;
      crc) cmd_section_crc ;;
      *) die "section steps: predict | repredict | crc" ;;
    esac
    ;;
  celltype|cell-stratified|cell_stratified|cross-celltype|cross_celltype)
    case "${2:-}" in
      predict) cmd_celltype skip ;;
      repredict) cmd_celltype overwrite ;;
      *) die "celltype steps: predict | repredict" ;;
    esac
    ;;
  uni2)
    case "${2:-}" in
      A0|A1|A3) cmd_uni2 "$2" ;;
      *) die "uni2 steps: A0 | A1 | A3" ;;
    esac
    ;;
  *) die "unknown command '$1' (section | celltype | uni2)" ;;
esac
