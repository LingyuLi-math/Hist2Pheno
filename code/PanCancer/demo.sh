## 2026.09.29 LLY: pan-cancer 7-cohort Hist2Pheno command index
## Ontology: code/Hist2Pheno_pkg/dataset/PanCancerCellType.xlsx  sheet celltype_pancancer
## Heads: L2=final_CT (20) / L12=final_sublineage (8) / L1=final_lineage (5)
## CV: default 5-fold cross-dataset section (each HE section wholly train or val).
##      --group-splitter cell_stratified mixes cells (StratifiedKFold on L1|L12|L2)
##      into data/PanCancer/result_cell_stratified_spatial/ (does not overwrite result_all_spatial).
## Outputs: data/PanCancer/
##
## Xenium: Lung, BRCA, CRC (CV train: P1CRC+P2CRC; predict also P5CRC).  ESCC omitted.
## CODEX:  HCC, PDAC c001/c003 (278 cores), GIST c009/c011/c013 (550 cores), GBM.
## UNI embeddings stay in each cohort *_matched_features.h5ad (not copied).
## Full pool ≈ 5.51M cells × 1024-d float32 (~22 GB). Smoke-test before full train.

cd /home/lingyu/ssd2/Python/Hist2Pheno/
# conda activate SeededNTM

OUT=data/PanCancer

## Optional clean of index / label cache / train outputs (does not touch cohort h5ads)
# /bin/rm -rf -- "$OUT/label_cache" \
#                "$OUT/result_smoke" \
#                "$OUT/result_all_spatial" \
#                "$OUT/result_cell_stratified_spatial" \
#                "$OUT/result_smoke_cellstrat"
# /bin/rm -f -- "$OUT/section_index.csv" \
#               "$OUT/section_index_summary.csv" \
#               "$OUT/fold_plan.csv" \
#               "$OUT/fold_plan_cell_stratified.csv" \
#               "$OUT/ontology_heads.json"



## 0. Scan 7-cohort matched h5ads, remap to pancancer L1/L12/L2, write label_cache + fold_plan
0_prepare_pancancer_index.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/prepare_pancancer_index.py

  ## subset (debug)
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/prepare_pancancer_index.py --datasets BRCA GBM

  ## same via train CLI
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py --steps index

## 1. Inspect ontology / section_index / fold_plan / one-section remap (no GPU)
1_PanCancer_inspect_index.ipynb

## 2. Smoke: two small cohorts, cap cells / section
2_smoke_train
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --datasets BRCA GBM \
      --max-cells-per-section 5000 \
      --max-epochs 2 --patience 2 \
      --steps prepare train he_validate \
      --save-result result_smoke \
      --ablation-tag smoke_brca_gbm

  ## CPU fallback if no GPU
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --datasets BRCA GBM \
      --max-cells-per-section 5000 \
      --max-epochs 2 --patience 2 \
      --allow-cpu-train \
      --steps prepare train he_validate \
      --save-result result_smoke \
      --ablation-tag smoke_brca_gbm

## 3. Full 7-cohort pretrain (cross-dataset section 5-fold, spatial k=8 mean)
## stdout+stderr stay on the terminal and append to $OUT/train.log; /usr/bin/time prints wall/user/sys.
3_PanCancer_train_validate_cv_UNIlabel.py
  mkdir -p "$OUT"
  LOG="$OUT/train.log"
  {
    echo "===== PanCancer train start $(date '+%F %T') ====="
    # add --cuda-device 2 after python -u if GPU 0 is busy
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n SeededNTM python -u \
        code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
          --cuda-device 2 \
          --use-spatial-context --spatial-k 8 --spatial-mode mean \
          --steps prepare train he_validate \
          --save-result result_all_spatial \
          --ablation-tag D_emph_L2_spatial_pancancer
    echo "===== PanCancer train end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$LOG"

  ## resume from fold ckpts under result_all_spatial/cross_dataset_cv/<tag>/
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer

  ## force retrain (ignore existing fold ckpts)
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --no-resume-from-checkpoints \
      --steps prepare train he_validate \
      --save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer

  ## sklearn GroupKFold instead of dataset-stratified section folds
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --group-splitter group_kfold \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer_groupkfold

## 3b. Cell-stratified CV (mix cells across sections; StratifiedKFold on L1|L12|L2)
## Why: section CV never splits DCIS (BRCA-only) or Injury_epithelial (Lung-only) the
## same way as ubiquitous Macrophage/Endothelial; pooled OvR AUROC then ranks rare
## organ-specific classes very high. Cell-stratified CV puts every abundant L2 type
## in every train *and* val fold. Spatial kNN is masked so train cells cannot see val
## neighbors. Hierarchical L1→L12→L2 sampler is on by default.
## Outputs stay under result_cell_stratified_spatial/ (never overwrites result_all_spatial).
## Checkpoint OOF is optimistic vs a new slide (same-section cells in train and val).
3b_cell_stratified
  ## smoke (same BRCA+GBM cap as step 2; separate folder)
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --group-splitter cell_stratified \
      --datasets BRCA GBM \
      --max-cells-per-section 5000 \
      --max-epochs 2 --patience 2 \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_smoke_cellstrat \
      --ablation-tag smoke_brca_gbm_cellstrat

  ## full 7-cohort (new folder + tag; tee a separate log)
  mkdir -p "$OUT"
  LOGC="$OUT/train_cell_stratified.log"
  {
    echo "===== PanCancer cell-stratified train start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n SeededNTM python -u \
        code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
          --cuda-device 2 \
          --group-splitter cell_stratified \
          --use-spatial-context --spatial-k 8 --spatial-mode mean \
          --steps prepare train he_validate \
          --save-result result_cell_stratified_spatial \
          --ablation-tag D_emph_L2_spatial_pancancer_cellstrat
    echo "===== PanCancer cell-stratified train end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$LOGC"

## writes
##   data/PanCancer/section_index.csv
##   data/PanCancer/fold_plan.csv                       # section CV (default)
##   data/PanCancer/fold_plan_cell_stratified.csv       # only if --group-splitter cell_stratified
##   data/PanCancer/label_cache/*.npz
##   data/PanCancer/train.log                           # tee of step 3 (append)
##   data/PanCancer/train_cell_stratified.log           # tee of step 3b
##   data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt
##   data/PanCancer/result_cell_stratified_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer_cellstrat/best_mlp_gpu.pt

## 4. Predict all sections with the pretrained spatial model + 4×3 GT/pred/ROC/StarDist-all panels
4_PanCancer_predict_visualize.py
  ## smoke one BRCA section (separate scaler so it does not become the production scaler)
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_predict_visualize.py \
      --datasets BRCA --max-sections 1 \
      --scaler-from-work-index \
      --scaler-path "$OUT/pred_all_spatial/_smoke_scaler.joblib" \
      --scaler-max-cells 40000

  ## all 7 cohorts (spatial k=8, matches the checkpoint). Resume with --skip-existing.
  ## first run fits result_all_spatial/scaler.joblib (400k mixed cells unless file exists).
  mkdir -p "$OUT/pred_all_spatial"
  LOGP="$OUT/pred_all_spatial/predict.log"
  {
    echo "===== PanCancer predict start $(date '+%F %T') ====="
    /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
      conda run --no-capture-output -n SeededNTM python -u \
        code/PanCancer/PanCancer_predict_visualize.py \
          --include-incomplete-lung --skip-existing
    echo "===== PanCancer predict end $(date '+%F %T') ====="
  } 2>&1 | tee -a "$LOGP"

  ## CRC P5CRC is not in section_index (held out of pan-cancer CV) but is predicted by default.
  ## Re-run only missing CRC (P1/P2 skip-existing; writes P5CRC):
  conda run --no-capture-output -n SeededNTM python -u \
    code/PanCancer/PanCancer_predict_visualize.py \
      --datasets CRC --skip-existing --cuda-device 2

## 5. Pooled pan-cancer ROC (notebook; needs step 4 AUROC CSVs)
5_Pred_statistic_visual_pancancer_all.ipynb
  ## pooled L2 / L12 / L1 OvR ROC under pred_all_spatial/pooled_roc/
  ## Prostate: data/ST/Prostate and data/Visium/Prostate still need UNI matched h5ads

