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
| `result_*/oof_predictions.npz` | OOF labels and probabilities. From 2026-11-15 also class names, `checkpoint_family`, and `fold_ids` ([below](#2026-11-15-oof-archive-contract)) |
| `result_all_spatial/cross_dataset_cv/<tag>/best_mlp_gpu.pt` | section-CV checkpoint |
| `result_cell_stratified_spatial/` | cell-stratified OOF + checkpoint (separate from above) |
| `train.log` | full section-CV stdout/stderr + wall-clock (`tee -a`; see `demo.sh` step 3) |
| `train_cell_stratified.log` | cell-stratified train log (`demo.sh` step 3b) |

Full pooling is memory-heavy (~5.5M cells × 1024-d float32 ≈ 22 GB embeddings). Use
`--datasets BRCA GBM --max-cells-per-section 5000` to smoke-test.

## 2026-11-15 OOF archive contract

`PanCancer_train_validate_cv_UNIlabel.py` still writes one `oof_predictions.npz` per run. The probabilities are unchanged. The file now also records which class each column is, which training run produced it, and which fold predicted each row. This does not alter a finished checkpoint. An `oof_predictions.npz` from before this date has no `*_class_names`, `checkpoint_family`, or `fold_ids`. Confidence code still reads those older files, and it assumes the test-file class order. Retrain to make the order explicit.

Keys added next to the existing `row_indices`, `{head}_labels`, and `{head}_probs` (`head` is `l2`, `l12`, `l1_head`):

| Key | What it is |
|-----|------------|
| `l2_class_names`, `l12_class_names`, `l1_head_class_names` | Column order of that head. Same lists as `cv["class_names"]`, `class_names_level12`, and `class_names_level1`, which are the lists used to encode the labels. Column `i` is that name. |
| `checkpoint_family` | Family id of `best_mlp_gpu.pt`. Fold id, held-out sections, and best epoch are not part of the family, so every fold of one run shares it. A different recipe (for example A2 vs A3) does not. |
| `fold_ids` | Fold of each probability row. Same order as `l2_probs`: folds concatenated, not global cell order. |

`oof_rows.csv` remains one row per cell in global order (`row_index` 0…N−1), with `fold` in that same order. To line it up with the npz, index the CSV by `row_indices`. `fold_ids[i]` is the fold of probability row `i`, which is cell `row_indices[i]`.

Package-side checks that consume this file (evaluation ontology, confidence class alignment, external L1/L12 archives) are recorded in [`code/Hist2Pheno_pkg/README.md`](../Hist2Pheno_pkg/README.md#2026-11-15-integrity-fixes).

## 2026-10-26 UNI2 run order

The filename prefix is the order. Env: `conda activate SeededNTM`. One UNI2 job at a time. A0 must finish its UNI2 cache (`data/PanCancer/uni2_aligned/`) before A1–A8. A8 reuses that cache; it does not rebuild it. A4 (k=8 attention) is not part of this choice. `demo.sh` still dispatches the older published UNI section/celltype pipeline.

### 1. Train and internal cross-validation — [`1_demo_train.sh`](1_demo_train.sh)

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A0
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A1
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A2
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A3
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A5
CUDA_DEVICE=1 bash code/PanCancer/1_demo_train.sh uni2 A6
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A7
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A8
```

Each job is cross-section 5-fold. It writes `result_uni2_<arch>_paired_v2_seed42/validation_internal_metrics.csv` and `oof_predictions.npz`. The log line `OOF L2 macro AUROC=` is the pooled out-of-fold score on the seven training cohorts (5,627,071 cells, 896 sections). It is not an external test set. A file written before 2026-11-15 has labels and probabilities only; the class-order fields in the [OOF archive contract](#2026-11-15-oof-archive-contract) appear after the next train.

### 2. Choose the architecture — [`2_Pred_statistic_visual_pancancer_uni2_ablation.ipynb`](2_Pred_statistic_visual_pancancer_uni2_ablation.ipynb)

Compare A0–A3, then A5/A6/A7, on that pooled OOF L2 macro AUROC, plus the per-cohort and per-class breakdown of the same OOF cells. A8 is the next run and has no number here until that job finishes. This notebook does not call predict, and those cohorts are not a held-out test set.

Numbers below are the **2026-10-27 `paired_v2_seed42` retrain** (named-layer init, split-seed 42, L2→L1 NLL weight 2). A 2026-10-26 morning A3 of 0.8907 used L2→L1 NLL weight 1 and those checkpoints were removed.

A3 remains the best finished architecture on pooled OOF L2 (**0.8897**). Its deployment weight is fold 0:

`data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt`

0.8897 is the five-fold pooled OOF of the A3 recipe. It is not the score of this one file on every section.

### 2026-11-16 Training L2 label mix

Ground-truth L2 counts on the 5,627,071 training cells (896 sections, seven cohorts). Counted from `l2_labels` in `data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/oof_predictions.npz`. Each cell is held out once, so this file is the full training label mix, not an A2 prediction. A0–A8 train on the same cells. Column order is the workbook L2 order (`ontology_class_names`); this archive has no `l2_class_names` key.

`B_cell` is 135,794 cells, **2.41%**, 12th of 20 classes by count. `Plasma` is 61,271, **1.09%**. Together they are the L12 family `B_Plasma`: 197,065 cells, **3.50%**. That is the empirical prior the unweighted L12 term pulls toward. They are scarce next to Fibroblast (15.14%) and Tumor (14.94%), and they are not the smallest classes: Neural (0.19%), NK (0.21%), and Injury_epithelial (0.25%) are smaller.

| L2 | L12 | n | % |
|----|-----|--:|--:|
| DCIS | Tumor | 41,775 | 0.74 |
| Tumor | Tumor | 840,604 | 14.94 |
| Alveolar | Epithelial | 58,050 | 1.03 |
| Epithelial | Epithelial | 556,973 | 9.90 |
| Injury_epithelial | Epithelial | 14,090 | 0.25 |
| Myoepithelial | Epithelial | 18,010 | 0.32 |
| CD4_T | T_cell | 454,922 | 8.08 |
| CD8_T | T_cell | 426,920 | 7.59 |
| Tregs | T_cell | 322,139 | 5.72 |
| NK | T_cell | 11,655 | 0.21 |
| **B_cell** | **B_Plasma** | **135,794** | **2.41** |
| **Plasma** | **B_Plasma** | **61,271** | **1.09** |
| DC | Myeloid | 316,538 | 5.63 |
| Macrophage | Myeloid | 686,418 | 12.20 |
| Neutrophil | Myeloid | 177,425 | 3.15 |
| Fibroblast | Stromal | 851,747 | 15.14 |
| Myofibroblast | Stromal | 38,977 | 0.69 |
| Stromal | Stromal | 157,868 | 2.81 |
| Endothelial | Endothelial | 445,047 | 7.91 |
| Neural | Neural | 10,848 | 0.19 |

L12 totals from the same file: T_cell 1,215,636 (21.60%), Myeloid 1,180,381 (20.98%), Stromal 1,048,592 (18.63%), Tumor 882,379 (15.68%), Epithelial 647,123 (11.50%), Endothelial 445,047 (7.91%), B_Plasma 197,065 (3.50%), Neural 10,848 (0.19%).

### 2026-11-11 — A5 / A6 L12-link ablation (finished)

Same UNI2 cache, same cross-section 5-fold, same spatial trunk as **A2** (k=8 mean, **no gate**). Only the L12 loss weights change:

| | CE(L12) | L2→L12 NLL | L12→L1 NLL | L2→L1 NLL | JS |
|--|---------|------------|------------|-----------|-----|
| A2 (reference trunk) | 1 | 0.5 | 0.5 | 2 | 0.1 / 0.1 |
| A5 `l12_links_1` | 1 | **1** | **1** | 2 | 0.1 / 0.1 |
| A6 `l12_links_2` | **2** | **2** | **2** | 2 | 0.1 / 0.1 |

A3 keeps A2's L12 weights and adds a learned spatial gate. Fold-mean AUROC is the mean of the five `group_cv_folds.csv` rows (each fold's held-out sections). Pooled OOF L2 is the single AUROC on all 5.6M OOF cells.

| | spatial | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|---------|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| A3 k=8 mean+gate | gate | **0.8897** | **0.8805** | **0.8282** | **0.3013 ± 0.029** | 0.4225 ± 0.044 | 0 |
| A2 k=8 mean | mean | 0.8894 | 0.8798 | 0.8272 | 0.2941 ± 0.030 | 0.4181 ± 0.042 | 1 |
| A5 L12 links = 1 | mean | 0.8894 | 0.8798 | 0.8250 | 0.2980 ± 0.027 | 0.4220 ± 0.037 | 1 |
| A6 L12 links = 2 | mean | 0.8882 | 0.8783 | 0.8253 | 0.2972 ± 0.030 | **0.4391 ± 0.049** | 1 |
| A1 adjacent, no spatial | none | 0.8601 | 0.8468 | 0.7888 | 0.2303 ± 0.018 | 0.3412 ± 0.020 | 0 |
| A0 legacy, no spatial | none | 0.8580 | 0.8453 | 0.7874 | 0.2287 ± 0.017 | 0.3390 ± 0.020 | 0 |

Raising the L12 edges did not beat A2/A3 on L2 or L12 AUROC. A5 is tied with A2 on pooled OOF L2 (0.8894) and slightly better on L2 F1. A6 is 0.0012 below A2 on L2 AUROC; its gain is L1 macro-F1 (0.439 vs A2 0.418). Keep **A3** for deployment if the target is L2. Use A5/A6 only if you want the stronger L12 consistency loss on the A2 trunk.

Weights:

- A5 fold 1: `data/PanCancer/result_uni2_A5_l12_links_1_paired_v2_seed42/cross_dataset_cv/A5_l12_links_1/best_mlp_gpu.pt`
- A6 fold 1: `data/PanCancer/result_uni2_A6_l12_links_2_paired_v2_seed42/cross_dataset_cv/A6_l12_links_2/best_mlp_gpu.pt`

Logs: `code/log/pancancer_section_uni2_A5_l12_links_1_paired_v2_seed42.log` (wall 3:32), `...A6_l12_links_2...` (wall 3:24). NPC maps: [`Pred_visual_npc_uni2_A5.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A5.ipynb), [`Pred_visual_npc_uni2_A6.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A6.ipynb).

### 2026-11-12 — A7 sampler + focal (finished; no improvement)

Same UNI2 cache and A2 trunk (k=8 mean, **no gate**, L12 weights unchanged). Training-side balance only:

| | sampler | `class_weight_mode` | L2 focal γ |
|--|---------|---------------------|------------|
| A2 (reference) | off | `balanced` | 0 |
| A7 `bal_focal` | hierarchical L1→L12→L2 | `none` | **1** |

A7 finished 2026-11-12 00:19 (wall 2:46). Pooled OOF L2 is **0.8581**, level with **A0** (0.8580), **0.031** below A2. Fold-mean L12 AUROC 0.7846 is also below A0. Mean best epoch 5.8 vs A2 10.4.

| | spatial | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|---------|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| A3 k=8 mean+gate | gate | **0.8897** | **0.8805** | **0.8282** | **0.3013 ± 0.029** | 0.4225 ± 0.044 | 0 |
| A2 k=8 mean | mean | 0.8894 | 0.8798 | 0.8272 | 0.2941 ± 0.030 | 0.4181 ± 0.042 | 1 |
| A7 sampler + focal | mean | 0.8581 | 0.8579 | 0.7846 | 0.2410 ± 0.020 | 0.3697 ± 0.014 | 0 |
| A0 legacy, no spatial | none | 0.8580 | 0.8453 | 0.7874 | 0.2287 ± 0.017 | 0.3390 ± 0.020 | 0 |

The intended B / Plasma lift is recall-only and does not survive AUROC or F1 (same 5.6M OOF argmax):

| L2 / L12 | A2 AUROC | A7 AUROC | A2 recall | A7 recall | A2 F1 | A7 F1 | A2 n_pred | A7 n_pred |
|----------|----------|----------|-----------|-----------|-------|-------|-----------|-----------|
| `B_cell` | 0.802 | 0.787 | 0.282 | **0.316** | 0.309 | 0.284 | 112,102 | 166,685 |
| `Plasma` | 0.871 | 0.813 | 0.198 | **0.234** | 0.171 | 0.141 | 79,949 | 141,419 |
| `Tregs` | 0.907 | 0.892 | 0.485 | 0.183 | 0.398 | 0.252 | 462,903 | 145,523 |
| L12 `B_Plasma` | 0.796 | 0.758 | 0.182 | **0.325** | 0.264 | 0.256 | 74,837 | 304,373 |

Mean softmax `P(B)` does rise (A2 0.030 → A7 0.042). Extra mass is not concentrated on B: Endothelial argmax goes 254,110 → **1,957,081** (recall 0.184 → 0.635), Macrophage 192,611 → **4,910** (recall 0.096 → 0.002), Tumor recall 0.655 → 0.466. Keep **A3** for deployment. Do not run `3_demo_pred.sh uni2 A7`.

Weight: fold 0 `data/PanCancer/result_uni2_A7_bal_focal_paired_v2_seed42/cross_dataset_cv/A7_bal_focal/best_mlp_gpu.pt`. Log: `code/log/pancancer_section_uni2_A7_bal_focal_paired_v2_seed42.log`.

### 2026-11-16 A8 balanced L12 (not yet trained)

Same UNI2 cache and the **A3** trunk: k=8 mean, learned spatial gate, scalar loss weights unchanged. A7 changed the sampler and turned class weights off; A8 does not. The only change from A3 is `--l12-class-weight-mode balanced`.

| | spatial | CE(L12) weight | L2→L12 / L12→L1 NLL | L12 class weights |
|--|---------|----------------|---------------------|-------------------|
| A3 (reference) | k=8 mean + gate | 1, unweighted classes | 0.5 / 0.5, unweighted | `none` |
| A8 `l12_balanced` | k=8 mean + gate | 1, **balanced** | 0.5 / 0.5, **balanced** | `balanced` |

L2→L1 NLL stays 2. Both JS terms stay 0.1 and stay unweighted. L1 and L2 cross-entropy stay `balanced`, as in A3. The L12 weights are inverse-frequency on the eight L12 classes, so `B_Plasma` is no longer pulled back to its 3.5% prior by an unweighted L12 term. Mechanism and the `none` default that leaves A0–A7 families unchanged: [`code/Hist2Pheno_pkg/README.md`](../Hist2Pheno_pkg/README.md#2026-11-16-a8-class-weighted-l12-opt-in).

This run does not overwrite A3. Outputs:

- `data/PanCancer/result_uni2_A8_l12_balanced_paired_v2_seed42/`
- checkpoint directory `cross_dataset_cv/A8_l12_balanced/`
- log `code/log/pancancer_section_uni2_A8_l12_balanced_paired_v2_seed42.log`

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A8
```

Fill the pooled OOF L2, the B / Plasma table, and the chosen fold into this section after the job finishes. Do not deploy A8, and do not run `3_demo_pred.sh uni2 A8`, until that OOF number is in hand. NPC predict after that is `code/External_val/NPC/demo_uni2_A8.sh`.

### 3. Predict every training section with that weight — [`3_demo_pred.sh`](3_demo_pred.sh)

```bash
CUDA_DEVICE=0 bash code/PanCancer/3_demo_pred.sh uni2 A3
```

This is `--deployment`: one checkpoint scores all 896 sections. Sections other than fold 0 were in its training set. Do not run `uni2 A0` or `uni2 A1` here; those commands are OOF rescoring of the weaker models. Output: `data/PanCancer/pred_uni2_A3_bestmlp/`. `--skip-existing` resumes. Figures are skipped; matched-cell ROC CSVs are written.

### 4. Pool those predictions — [`4_Pred_statistic_visual_pancancer_all.ipynb`](4_Pred_statistic_visual_pancancer_all.ipynb)

Run the first code cell, then the last two cells.

| Pool | What it is | L2 macro AUROC |
|------|------------|----------------|
| All 896 sections | Includes sections this weight trained on | 0.9164 |
| Fold 0, 182 sections, 1,391,571 cells | Sections this weight did not train on | 0.8983 |

Use 0.8983 as this checkpoint's unseen-section number. The front of this notebook is the older published UNI cross-section vs cross-celltype comparison, not this step.

## Other commands in the same scripts

[`1_demo_train.sh`](1_demo_train.sh) also trains the published UNI pipelines. [`3_demo_pred.sh`](3_demo_pred.sh) also predicts from those UNI checkpoints. Running either file with no arguments prints usage and does not start work. `CUDA_DEVICE` defaults to 2.

```bash
bash code/PanCancer/1_demo_train.sh index
bash code/PanCancer/1_demo_train.sh section smoke
bash code/PanCancer/1_demo_train.sh section train
bash code/PanCancer/1_demo_train.sh celltype smoke
bash code/PanCancer/1_demo_train.sh celltype train
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 smoke
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A4
CUDA_DEVICE=0 bash code/PanCancer/3_demo_pred.sh uni2 A5
CUDA_DEVICE=1 bash code/PanCancer/3_demo_pred.sh uni2 A6

bash code/PanCancer/3_demo_pred.sh section predict
bash code/PanCancer/3_demo_pred.sh section repredict
bash code/PanCancer/3_demo_pred.sh section crc
bash code/PanCancer/3_demo_pred.sh celltype predict
bash code/PanCancer/3_demo_pred.sh celltype repredict
```

Published section/celltype logs append to `data/PanCancer/train.log` and `train_cell_stratified.log`. UNI2 logs replace `code/log/pancancer_section_uni2_<arch>.log`. k=16 and k=32 attention are not commands. Fold checkpoints under `cross_dataset_cv/<tag>/hce_group_fold_*.pt` are reused unless you pass `--no-resume-from-checkpoints`.

## 2026.10.01 — Predict every section + 4×3 panels + pooled ROC

Checkpoint: `result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt` (spatial k=8).

CLI: `PanCancer_predict_visualize.py` (spatial on by default; `--no-spatial-context` to turn off).

| Step | What |
|------|------|
| 1 | Load each CV matched h5ad, including CRC P1CRC, P2CRC, and P5CRC, remap to pan-cancer L1/L12/L2, predict three heads |
| 2 | If `{sample}_all_features_stardist.h5ad` exists, predict all StarDist nuclei (row 4; Lung Incomplete_Cases via `--include-incomplete-lung`) |
| 3 | Write a **4×3** figure per section (rows: GT, Hist2Pheno matched pred, ROC, StarDist-all; cols: L1, L12, L2; per-column legends) |
| 4 | Notebook `4_Pred_statistic_visual_pancancer_all.ipynb` pools per-section AUROC CSVs (HCC-style pooled ROC) |

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

