# Pan-cancer Hist2Pheno (7 public cohorts)

Joint three-head model (L2 / L12 / L1) trained on all public datasets in
`PanCancerCellType.xlsx` sheet `celltype_pancancer` (ESCC omitted).

| Tech | Dataset | Notes |
|------|---------|--------|
| Xenium | Lung, BRCA, CRC | CRC CV train is P1+P2+P5 transferred RCTD |
| CODEX | HCC, PDAC, GIST, GBM | PDAC coverslips c001/c003; GIST c009/c011/c013 |

Heads follow the workbook:

- **L2** `final_CT` — 20 trainable classes
- **L12** `final_sublineage` — 8 classes
- **L1** `final_lineage` — 5 classes (Epithelial / Immune / Stromal / Endothelial / Neural)

Native labels are remapped on read. UNI embeddings stay in each cohort’s
`*_matched_features.h5ad`; nothing is copied.

## Cross-dataset section CV

Default splitter is **cross-dataset section** 5-fold (`--group-splitter cross_dataset_section`):

- group = one HE section (`dataset::sample`)
- within each dataset, sections are shuffled then assigned round-robin to folds 0–4
- every fold therefore holds out ~1/5 of **each** cohort (CRC has 3 sections, so it appears in three val folds)
- a section is **wholly** train or val; cells from the same slide are never mixed
- the three heads (L1 / L12 / L2) share that same section split; class mix is **not** stratified
- `train_balance_sampler` is off (L2/L1 use `class_weight_mode=balanced` only)

Vanilla sklearn `GroupKFold` is `--group-splitter group_kfold`.

This is why pooled L2 OvR AUROC can rank **organ-specific** classes (DCIS on BRCA, Injury_epithelial on Lung) above ubiquitous Macrophage / Endothelial: the negative class for a BRCA-only type is almost every cell in the other six cohorts, which is an easy OvR problem. Section CV does not mix those types into every val fold the way a cell-stratified split would.

## Cell-stratified CV (optional; new output folder)

`--group-splitter cell_stratified` (aliases: `stratified_cell`, `cell_type_kfold`, `mixed_cell`):

- pools all sections, then `StratifiedKFold` on the joint **L1|L12|L2** label (L2 already determines parents in this ontology)
- every fold’s train **and** val sets contain each abundant cell type
- rare strata (< k cells) collapse L2 → L12 → L1, then round-robin
- spatial kNN is **split-masked** (train cells cannot see val neighbors)
- hierarchical L1→L12→L2 `WeightedRandomSampler` is on by default (`--no-train-balance-sampler` to turn off)

If you pass this splitter without changing `--save-result` / `--ablation-tag`, they are redirected so the existing section-CV checkpoint is not overwritten:

| | Section CV (default) | Cell-stratified |
|--|----------------------|-----------------|
| folder | `result_all_spatial/` | `result_cell_stratified_spatial/` |
| tag | `D_emph_L2_spatial_pancancer` | `D_emph_L2_spatial_pancancer_cellstrat` |
| fold plan | `fold_plan.csv` | `fold_plan_cell_stratified.csv` |
| train log | `train.log` | `train_cell_stratified.log` |

**Caveat:** same-section cells land in both train and val, so OOF AUROC is optimistic relative to a new slide. Use section CV for generalization-to-new-section; use cell-stratified CV to compare L2 classes on a more balanced mix.

A **137 / `Killed`** on the full cell-stratified job is the Linux OOM killer (swap exhausted), not `demo.sh` step 4 predict. Spatial neighbor lookup now pins `X_all` to the GPU **once per fold** (previously a 22 GB copy every batch) and keeps the scaler output as float32. Fold checkpoints: `result_cell_stratified_spatial/cross_dataset_cv/<tag>/hce_group_fold_*.pt`. Re-running the same command resumes remaining folds; a fold killed mid-training is treated as done (its best-so-far `.pt`). Delete that fold file to retrain it.

Smoke, then full train (do not start the overnight job unless you mean to):

```bash
python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
  --group-splitter cell_stratified \
  --datasets BRCA GBM --max-cells-per-section 5000 \
  --max-epochs 2 --patience 2 \
  --use-spatial-context --spatial-k 8 \
  --steps prepare train he_validate \
  --save-result result_smoke_cellstrat \
  --ablation-tag smoke_brca_gbm_cellstrat

python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
  --group-splitter cell_stratified \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --steps prepare train he_validate \
  --save-result result_cell_stratified_spatial \
  --ablation-tag D_emph_L2_spatial_pancancer_cellstrat
```

## Outputs (`data/PanCancer`)

| File | Role |
|------|------|
| `section_index.csv` | every matched h5ad + keep/drop counts |
| `section_index_summary.csv` | per-dataset section / cell totals |
| `fold_plan.csv` | which section is in which val fold (section CV) |
| `fold_plan_cell_stratified.csv` | per-fold L1/L12/L2 cell counts (cell-stratified only) |
| `ontology_heads.json` | L1 / L12 / L2 class order |
| `label_cache/*.npz` | keep indices + remapped labels + xy (no embeddings) |
| `result_all_spatial/` | section-CV OOF plots + `validation_internal_metrics.csv` |
| `result_all_spatial/cross_dataset_cv/<tag>/best_mlp_gpu.pt` | section-CV checkpoint |
| `result_cell_stratified_spatial/` | cell-stratified OOF + checkpoint (separate from above) |
| `train.log` | full section-CV stdout/stderr + wall-clock (`tee -a`; see `demo.sh` step 3) |
| `train_cell_stratified.log` | cell-stratified train log (`demo.sh` step 3b) |

Full pooling is memory-heavy (~5.5M cells × 1024-d float32 ≈ 22 GB embeddings). Use
`--datasets BRCA GBM --max-cells-per-section 5000` to smoke-test.

## Commands

See [`demo.sh`](demo.sh). Env: `conda activate SeededNTM`.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

python -u code/PanCancer/prepare_pancancer_index.py

# live console + append to data/PanCancer/train.log (wall-clock at the end)
mkdir -p data/PanCancer
{
  echo "===== PanCancer train start $(date '+%F %T') ====="
  /usr/bin/time -f '===== wall %E  user %U  sys %S =====' \
    python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --steps prepare train he_validate \
      --save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_pancancer
  echo "===== PanCancer train end $(date '+%F %T') ====="
} 2>&1 | tee -a data/PanCancer/train.log
```

Resume: fold checkpoints under `cross_dataset_cv/<tag>/hce_group_fold_*.pt` are reused unless you pass `--no-resume-from-checkpoints`.

## 2026.10.01 — Predict every section + 4×3 panels + pooled ROC

Checkpoint: `result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt` (spatial k=8).

CLI: `PanCancer_predict_visualize.py` (spatial on by default; `--no-spatial-context` to turn off).

| Step | What |
|------|------|
| 1 | Load each CV matched h5ad, including CRC P1CRC, P2CRC, and P5CRC, remap to pan-cancer L1/L12/L2, predict three heads |
| 2 | If `{sample}_all_features_stardist.h5ad` exists, predict all StarDist nuclei (row 4; Lung Incomplete_Cases via `--include-incomplete-lung`) |
| 3 | Write a **4×3** figure per section (rows: GT, Hist2Pheno matched pred, ROC, StarDist-all; cols: L1, L12, L2; per-column legends) |
| 4 | Notebook `Pred_statistic_visual_pancancer_all.ipynb` pools per-section AUROC CSVs (HCC-style pooled ROC) |

Scaler is fit once (`result_all_spatial/scaler.joblib`, default 400k cells, round-robin across the 7 cohorts) and reused. Smoke uses a separate `_smoke_scaler.joblib` so it does not become the production scaler. Pass `--scaler-max-cells 0` to fit on every matched cell.

### Output tree (`data/PanCancer/pred_all_spatial/`)

```
matched/{dataset}/{sample}/{sample}_matched_pred.h5ad   # GT + pred labels, spatial_HE
matched/{dataset}/{sample}/{sample}_panel_4x3.png
stardist_all/{dataset}/{sample}/{sample}_all_stardist_pred.h5ad
stardist_roc/{dataset}__{sample}/validation_external_stardist_matched_AUROC*.csv
incomplete_lung/{sample}/…   # optional
metrics_per_section.csv
pooled_roc/                  # notebook
```

Prostate (`data/ST/Prostate`, `data/Visium/Prostate`) has HE/StarDist or Visium counts but **no** UNI `*_matched_features.h5ad` yet — do not apply the checkpoint until that h5ad exists.

Smoke (1 BRCA section; uses `_smoke_scaler.joblib`), then full 7-cohort (hours; `--skip-existing` to resume):

```bash
python -u code/PanCancer/PanCancer_predict_visualize.py \
  --datasets BRCA --max-sections 1 \
  --scaler-from-work-index \
  --scaler-path data/PanCancer/pred_all_spatial/_smoke_scaler.joblib \
  --scaler-max-cells 40000

python -u code/PanCancer/PanCancer_predict_visualize.py --include-incomplete-lung --skip-existing

# CRC P5CRC was missing from the first 895-section pass (not in section_index.csv)
python -u code/PanCancer/PanCancer_predict_visualize.py --datasets CRC --skip-existing
```

