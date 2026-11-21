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

## 2026-11-20 Tile-context update: fold PCA, B without PCA, SPORA at 0.5 µm/px

Changes to the 2026-11-19 ablation below (that section is kept as written).

**Which `l12_links_1` is the reference.** The existing run
`result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42`:
- outer folds `cross_dataset_section`, `SPLIT_SEED=42`: whole sections; within each cohort the
  sections are shuffled and dealt round-robin, so every fold holds ~1/5 of each cohort's sections;
- `INNER_VAL=cohort_safe`: early stopping never takes a cohort's last training section or an L2
  class out of inner training (0/5 folds lost a class);
- `--class-weight-mode balanced`: L2 / L1 class weights from each fold's own training proportions.
- The outer folds are not L2-stratified (that is `SPLITTER=stratified_section`, folders `stratsec_`).

All tile arms use the same folds, inner split and seed, so they are paired fold by fold.

**2026-11-20 (2): L2-balanced section folds are now the default of `run_tile_context.sh`**
(`SPLITTER=stratified_section`, `pancancer_cv.assign_stratified_section_folds`, added 2026-11-16):
- whole sections are held out (cross-section, as before);
- sections are placed rarest-first, then two sections **of the same cohort** are swapped between
  folds whenever that brings every fold closer to 1/5 of every L2 class (each class weighted
  equally, so B_cell / Plasma count as much as Tumor), while keeping each cohort's section count
  and the cell count per fold balanced;
- BRCA has 2 sections, so they go to 2 different folds and the other 3 test folds have no DCIS
  (unavoidable); every other cohort is balanced as far as its sections allow.
- `run_tile_context.sh plan` builds the folds once (prepare of `l12_links_1_mpp05`) and saves
  `data/PanCancer/fold_plan_stratsec_<tag>.csv` (+ `_l2_balance.csv`, share of each class per fold).
  Every arm copies it into its result folder (`HIST2PHENO_FOLD_PLAN`), so the 5 arms have identical
  folds even though tile arms drop a few cells.
- Folders: `result_uni2_ablation_<arm>_stratsec_<tag>_paired_v2_seed42`. The old `l12_links_1`
  (round-robin folds) is no longer fold-paired, so only the `_vs_l12_links_1_mpp05` table is written.
  `SPLITTER=cross_dataset_section` restores the round-robin folds and the second table.

**Arms now** (`l12_links_1_nospatial` is no longer run; its recipe stays in `1_demo_train.sh`):

| arm | input | width |
|--|--|--|
| `tile_local` (A) | centroid patch token | 1536 |
| `tile_local_global` (B) | A + tile class token, PCA 256 **fitted on each fold's training rows** | 1792 |
| `tile_local_global_full` | A + full tile class token, no PCA (`--embedder uni2_tile_lg_full`) | 3072 |
| `tile_local_global_k8` (C) | B + k=8 neighbour mean | 1792 (+ context) |
| `l12_links_1_mpp05` | `l12_links_1` (14 px crop, k=8) after the SPORA crops are re-extracted at 0.5 µm/px | 1536 |

Comparisons: B vs `tile_local_global_full` = effect of the PCA; C vs B = effect of k=8;
every arm vs `l12_links_1_mpp05` = effect of the tile features with all cohorts at 0.5 µm/px;
`l12_links_1_mpp05` vs `l12_links_1` = effect of the SPORA scale fix alone.

**Fold PCA** (`HIST2PHENO_TILE_PCA=fold`, default). The pooled matrix keeps the 3072-d features
(float16, as stored). `Hist2Pheno_pkg/projected_scaler.py` fits the PCA of the global block on the
outer-train rows of each fold (up to 200k rows) and then the StandardScaler, as one scaler saved in
the checkpoint artifact; the outer test fold never enters the projection. The best-fold reload builds
the MLP at that reduced width (`n_features_out_`, 1792 for B), so `best_mlp_gpu.pt` matches the
network that was trained. `HIST2PHENO_TILE_PCA=global`
restores the 2026-11-19 behaviour (`tile_context.py fit-pca`, one PCA over all sections).

**SPORA at 0.5 µm/px.**
- Tile features: every section is resized by mpp/0.5 (SPORA x2). New checks stop a section when
  `mpp_map.csv` disagrees with the h5ad `uns['mpp']` (SPORA: 1.0) or with the UNI2 manifest scale,
  or when < 98 % of the nuclei fall inside the resized image. CRC add-on H&E is BigTIFF
(`.ome.tif`); Pillow cannot identify it, so the audit reads width and height from the tifffile
header (the same full-resolution size the extractor decodes). `tile_context.py audit-mpp` writes
  `data/PanCancer/tile_context/mpp_audit.csv` (µm/px from every source, old crop size in µm, tile size).
- 14 px crops: `bash code/spora/run_spora_pancancer.sh uni2-mpp05` re-extracts the SPORA UNI2 tokens
  with the 1 µm/px H&E resized x2 (`scale_image=True, scale=2`), so the crop is 7 µm like every other
  cohort. The old stores are moved to `sc_pth_14_14_1mpp_archive_<date>` (not deleted).
  `uni2_aligned`, pooled and k=8 caches are keyed on the token manifests, so they rebuild.
  After this step every uni2 arm sees the 0.5 µm/px SPORA crops; retrain under a new folder name
  (`l12_links_1_mpp05`), the old folders were trained on the 14 µm crops.
  ~~`run_spora_pancancer.sh uni2 / fix-axis / refresh-schurch` still extract at 1 µm/px; run
  `uni2-mpp05` again after any of them.~~ 2026-11-20 (2): all of them now extract at 0.5 µm/px
  (worker default `--target-mpp 0.5`); the 1 µm/px crops are no longer made.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
bash code/PanCancer/run_tile_context.sh audit     # µm/px of every section (no GPU); stops on a mismatch
bash code/PanCancer/run_tile_context.sh pilot     # audit + BRCA + GBM (with SPORA glioma) features
nohup bash code/PanCancer/run_tile_context.sh all > code/log/tile_context.out 2>&1 &
#   all = audit -> spora (crops at 0.5 µm/px) -> plan (L2-balanced folds) -> extract -> train (5 arms) -> summary
```

The default table is `data/PanCancer/tile_context_<tag>_vs_l12_links_1_mpp05.csv`.
`..._vs_l12_links_1.csv` is written only with `SPLITTER=cross_dataset_section`, because the finished
`l12_links_1` run used round-robin folds. `train` refuses to start while a SPORA store is not at scale 2.

Resources: `tile_local_global`, `_full` and `_k8` pool 3072-d float16 (same size as the 1536-d
float32 crop features). Four arms prepare at once on 4 GPUs, each with its own pooled matrix;
use `GPUS="0 1"` if RAM is short. The k=8 cache of C is built from the raw 3072-d features
(~46 GB float16 for 7.46M nuclei) and shared by all folds.

### Finished 2026-11-20, seed 42

All five arms finished on the shared L2-balanced section folds
(`fold_plan_stratsec_new1066axisfix_ivsafe.csv`), `INNER_VAL=cohort_safe`, seed 42.
`tile_local_global` was killed on the first pass while three 3072-d pools were loading at
once; the rerun on one GPU finished the same result directory. The comparison table was
rebuilt after that rerun and includes all five arms:
`data/PanCancer/tile_context_new1066axisfix_ivsafe_vs_l12_links_1_mpp05.csv`
(per fold: `..._per_fold.csv`).

Fold-mean columns are the mean of the five held-out folds. Pooled L2 AUROC is the single
score on `oof_predictions.npz`. The selected fold is the inner-validation
`three_tier_auc_sum`, not the external slide. These folds are not the round-robin folds of
`l12_links_1` (pooled L2 AUROC 0.9071), so that 0.9071 is not a row of this table. The
paired reference is `l12_links_1_mpp05`.

| Arm | Best fold | Pooled L2 AUROC | Fold-mean L2 AUROC | L2 accuracy | L2 macro-F1 | Decision L2 accuracy | Decision L2 macro-F1 | Decision macro-F1 vs mpp05 |
|--|--|--|--|--|--|--|--|--|
| `tile_local` (A) | 1 (2.7872) | 0.9376 | 0.9316 | 0.5887 | 0.4284 | 0.6097 | 0.4433 | +0.1482, 5/5 |
| `tile_local_global` (B) | 2 (2.7904) | 0.9388 | 0.9334 | 0.5831 | 0.4300 | 0.5980 | 0.4448 | +0.1497, 5/5 |
| `tile_local_global_full` | 2 (2.7882) | 0.9380 | 0.9333 | 0.5851 | 0.4287 | 0.6011 | 0.4394 | +0.1443, 5/5 |
| `tile_local_global_k8` (C) | 2 (2.7871) | 0.9376 | 0.9328 | 0.5829 | 0.4242 | 0.6002 | 0.4336 | +0.1385, 5/5 |
| `l12_links_1_mpp05` | 2 (2.6371) | 0.8997 | 0.8944 | 0.4600 | 0.2982 | 0.4865 | 0.2951 | — |

Accuracy and macro-F1 are fold means. Decision columns are cross-fitted `decision full`.
The tile features account for the gain: about +0.04 L2 AUROC and +0.14 decision macro-F1
against the 14 px crop, in every fold. PCA versus the full class token, and k=8 versus no
neighbours, stay inside about 0.01. k=8 is slightly below B on decision macro-F1.

Weights: `data/PanCancer/result_uni2_ablation_<arm>_stratsec_new1066axisfix_ivsafe_paired_v2_seed42/cross_dataset_cv/<arm>/best_mlp_gpu.pt`.
OOF: `oof_predictions.npz` in the same result directory. Logs:
`code/log/retrain_new1066axisfix_ivsafe/<arm>_seed42.out`
(`tile_local_global_seed42_rerun.out` for B), parent log `code/log/tile_context.out`.

**NPC patient1** (`20210809-C-AH4199551`, 142,184 nuclei) uses the five-fold mean of each
arm, not `best_mlp_gpu.pt` alone. `code/External_val/NPC/predict_finished_arms.py` writes
`data/Visium/NPC/pred_uni2_ens5_<arm>_stratsec_ivsafe_seed42/`. The same-slide comparison is
`Pred_visual_npc_uni2_local_global.ipynb`. One notebook per tile arm, with the A2 sections,
is `Pred_visual_npc_uni2_<arm>.ipynb`. Figures:
`data/Visium/NPC/pred_uni2_finished_arms_ivsafe_seed42/figures/`.

Whole-slide L2 argmax of that mean, largest class:

| Arm | Largest class | Tumor |
|--|--|--|
| `tile_local` | B_cell 0.297 | 0.182 |
| `tile_local_global` | B_cell 0.325 | 0.228 |
| `tile_local_global_full` | CD4_T 0.462 | 0.178 |
| `tile_local_global_k8` | CD4_T 0.369 | 0.117 |
| `l12_links_1_mpp05` | Tumor 0.678 | 0.678 |

Expert TLS, L2 softmax inside the three outlines / outside (5,262 nuclei inside). B cell is
`P(B_cell)+P(Plasma)`; T cell is CD4_T + CD8_T + Tregs.

| Arm | B cell | T cell | DC | Tumor |
|--|--|--|--|--|
| `tile_local` | 0.333 / 0.188 (1.77) | 0.333 / 0.235 (1.42) | 0.038 / 0.046 (0.81) | 0.056 / 0.123 (0.46) |
| `tile_local_global` | 0.361 / 0.206 (1.75) | 0.317 / 0.241 (1.32) | 0.026 / 0.037 (0.71) | 0.069 / 0.143 (0.48) |
| `tile_local_global_full` | 0.284 / 0.153 (1.86) | 0.433 / 0.309 (1.40) | 0.042 / 0.053 (0.81) | 0.034 / 0.106 (0.32) |
| `tile_local_global_k8` | 0.369 / 0.222 (1.66) | 0.395 / 0.306 (1.29) | 0.034 / 0.052 (0.66) | 0.037 / 0.090 (0.41) |
| `l12_links_1_mpp05` | 0.019 / 0.018 (1.06) | 0.079 / 0.066 (1.20) | 0.152 / 0.108 (1.41) | 0.164 / 0.339 (0.48) |

On this slide the tile arms raise B and T inside the expert outline and lower Tumor there.
Their whole-slide argmax is immune (and, for A, neutrophil 0.186) rather than Tumor. The
14 px model stays Tumor across the slide, and its B probability inside the outline stays near 0.02.

## 2026-11-19 Tile-context features: ablation A / B / C

**Why.** The UNI2 token used so far (`project_all_UNI2/.../sc_pth_14_14`) is the
class token of a 14 px crop fed to UNI2-h on its own. At 0.5 µm/px that crop
is 7 µm wide (half-width 3.5 µm), a single ViT-H/14 patch: the encoder sees
part of one nucleus and none of the tissue architecture that separates IDC
from DCIS (ducts, myoepithelial rings) or marks a TLS (follicles). The k=8
neighbour mean adds other 7 µm crops, not architecture.

**What.** `Hist2Pheno_pkg/tile_context.py` brings every section to
0.5 µm/px (native µm/px from `mpp_map.csv`), cuts 224 px (112 µm) tiles
with stride 112 and runs UNI2-h once per tile. Per nucleus:

| feature | from the tile | size |
|--|--|--|
| local | patch token at the centroid (bilinear on the 16 x 16 token grid). Same 7 µm footprint as before, but every token has attended to the whole tile | 1536 |
| global | tile class token (112 µm view), PCA to `HIST2PHENO_TILE_GLOBAL_DIM` (default 256; 0 = 1536) | 256 |

Both are averaged over the 2 x 2 covering tiles with the Gaussian centre
weight exp(-(dx²+dy²)/(2·0.5²)) (dx, dy in [-1, 1] from the tile centre; the
overlap weighting in MeowCat's public code), so the tile in which the
nucleus is most central dominates. Features are saved aligned to the h5ad
rows, so no coordinate matching is needed: `data/PanCancer/tile_context/<sha(h5ad)[:32]>/{local,global}.npy`
(float16) + `stamp.json` (h5ad size/mtime, coordinate hash, image, µm/px,
tile/stride/σ, UNI2-h weight SHA, code SHA). A changed h5ad or coordinate
set is refused at training time.

**Arms** (same losses as `l12_links_1`; only the input changes):

| arm | input | reference |
|--|--|--|
| `l12_links_1_nospatial` | 14 px crop class token, no neighbours | — |
| `tile_local` (A) | local | `l12_links_1_nospatial` |
| `tile_local_global` (B) | local + global | `l12_links_1_nospatial`, `tile_local` |
| `tile_local_global_k8` (C) | B + k=8 neighbour mean | `l12_links_1` |

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
bash code/PanCancer/run_tile_context.sh pilot     # BRCA + GBM features only; check the speed in code/log/tile_context/
bash code/PanCancer/run_tile_context.sh all       # extract (one shard per GPU) -> PCA -> 4 arms -> 2 tables
# or step by step: extract | pca | train | summary
```

Defaults: `INNER_VAL=cohort_safe`, `SEEDS=42` (C is paired with the existing
`l12_links_1` cohort_safe seed-42 run), `GPUS="0 1 2 3"`. Tables:
`data/PanCancer/tile_context_<tag>_vs_nospatial.csv` (A, B) and
`..._vs_l12_links_1.csv` (C). A single arm by hand:
`EMBEDDER=uni2_tile_lg bash code/PanCancer/1_demo_train.sh uni2 A5`
(folders without `RESULT_NAME` get `<embedder>_` in the suffix).

Notes:
- SPORA CRC / GBM tissues are prepared at 1 µm/px and their UNI2 crops were
  taken with `scale_image=False`, so their 14 px crop is 14 µm, not 7 µm like
  the other cohorts. The tile features resize SPORA by 2 (mpp 1.0 / 0.5) so
  every tile is 112 µm in every cohort.
- Disk: ~46 GB float16 for 7.46M nuclei, plus one shared pooled float32
  `.npy` per feature set (`tile_local` 1536-d, `tile_lg` 1792-d).
- External slides are not wired yet: `external_validation.py` still reads
  the 14 px crop tokens. Add tile features for external cohorts after the
  internal comparison shows a gain.

## 2026-11-17 Retrain everything on the seven cohorts (start here)

The finished 1,066-section weights (`run_uni2_new1066.sh`, A2 0.9053 / A7 0.9035 /
A8 0.9002, stratified A7 0.8811) stay in their own directories. The Schürch and
Hoefflin tables already store `centroid_x` as the column: the mask check is
140 and 30 correct, the UNI2 stores are newer than those tables, and those
stores were written before the weights. `retrain_all.sh` does not replace them.
It trains a separate series, in bf16 with float16 scaled copies, under
`result_uni2_ablation_<arm>_<DATA_TAG>_paired_v2_seed<N>/`. `STEP=check` should
pass on the current tables. `FIX_AXIS=1` is only for a later check that reports
swaps, unresolved tables, or stale UNI2 stores.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

# 1. SPORA check only (tables column-first, UNI2 stores newer than tables)
STEP=check bash code/PanCancer/retrain_all.sh
#    if it reports work: FIX_AXIS=1 STEP=check bash code/PanCancer/retrain_all.sh

# 2. everything: check -> index -> one prepare -> 11 arms on 4 GPUs -> summary
bash code/PanCancer/retrain_all.sh

# paired seeds, k sensitivity, other GPUs
SEEDS="42 43 44" WITH_K_SENSITIVITY=1 GPUS="0 1 2 3" bash code/PanCancer/retrain_all.sh

# one arm by hand, or the table again
DATA_TAG=new1066axisfix CUDA_DEVICE=0 bash code/PanCancer/run_ablation.sh l12_balanced
DATA_TAG=new1066axisfix bash code/PanCancer/run_ablation.sh summary
```

Arms (each changes one thing against the trunk `spatial_k8`):

| arm | change | old id |
|--|--|--|
| `legacy_nospatial` | legacy loss, no neighbors | A0 |
| `adjacent_nospatial` | adjacent loss, no neighbors | A1 |
| **`spatial_k8`** | trunk: adjacent loss + k=8 neighbor mean | A2 |
| `spatial_k8_gate` | + spatial gate | A3 |
| `l12_links_1` / `l12_links_2` | L12 link weights 1 / 2 | A4 / A5 |
| `balanced_sampler` | balanced sampler, no class weights | (half of old A6) |
| `focal_gamma1` | focal γ=1, balanced class weights kept | (half of old A6) |
| `l12_balanced` | balanced L12 | A7 |
| `gate_l12_balanced` | gate + balanced L12 | A8 |
| `l12_balanced_nospatial` | balanced L12 without neighbors | new |
| `spatial_k4` / `spatial_k16` | k sensitivity (`WITH_K_SENSITIVITY=1`) | new |

Defaults of `retrain_all.sh`: `DATA_TAG=new1066axisfix`, `HIST2PHENO_AMP=bf16`
for every arm, `HIST2PHENO_STORE_DTYPE=float16` (halves the scaled copies in
`scaled_cache/`), `DECISION_OBJECTIVE=f1_acc`, `SPLIT_SEED=42` (one fold plan
for every arm and seed). GPU 0 builds the k=8 neighbor cache with
`spatial_k8`; the other queues start with the arms that need no neighbors
and wait for that cache. Logs: `code/log/retrain_<DATA_TAG>/`.

Each finished run folder `data/PanCancer/result_uni2_ablation_<arm>_<DATA_TAG>_paired_v2_seed<N>/` holds
`oof_predictions.npz`, the weight `cross_dataset_cv/<arm>/best_mlp_gpu.pt`, and the decision layer
(`decision_rule.json`, `decision_rule_l2_only.json`, `decision_report.csv`,
`decision_per_class_f1.csv`, `decision_crossfit_predictions.npz`). The summary
`data/PanCancer/ablation_<DATA_TAG>_summary.csv` lists per arm: L2 macro
AUROC, L2 accuracy and macro-F1 (argmax and cross-fitted decision layer),
and the paired per-fold difference against `spatial_k8` with the number of
folds that improved. `SPLITTER=stratified_section` gives the same run with
L2-balanced section folds (`stratsec_` in the folder name).

### 2026-11-17 Inner validation and the l12_links_1 stack

Folds of every arm above: `cross_dataset_section`, whole sections, round-robin
inside each cohort (no L2 balancing; that is `SPLITTER=stratified_section`).
BRCA has two sections, so fold 0 holds out one and trains on the other, fold 1
the reverse, folds 2–4 train on both. Inside each fold, 20% of the training
sections go to inner validation (early stopping). The old split
(`GroupShuffleSplit`) ignores cohorts, so in folds 0/1 it can take the only BRCA
section and every DCIS cell out of training. Seed 42 did not hit this: mean
P(DCIS) on DCIS cells is 0.62–0.72 and DCIS AUROC 0.84–0.89 on both BRCA
sections. `INNER_VAL=cohort_safe` (`pancancer_cv.inner_val_split`) draws inner
validation per cohort, never takes a cohort's last section, and gives back
sections until every L2 class of the outer training set is in inner training.
Each fold now records `inner_val_sections` and `inner_lost_classes`; the
default (`random`) is the old split, unchanged. cohort_safe runs use the tag
`<DATA_TAG>_ivsafe` and are compared only with each other.

Arms stacked on `l12_links_1` (one change each, compared with `REFERENCE=l12_links_1`):
`l12_links_1_focal` (+ focal γ=1), `l12_links_1_balanced` (+ balanced L12),
`l12_links_1p5` (link weights 1.5, CE(L12) 1).

`code/PanCancer/run_l12_links_1_seeds.sh` trains only `l12_links_1` under `INNER_VAL=cohort_safe`, seeds 42 43 44, same outer folds (`SPLIT_SEED=42`). Seeds change initialisation and batch order. The first seed builds or reuses the shared k=8 cache; the others wait for `context cache ready`. Each seed writes `result_uni2_ablation_l12_links_1_<DATA_TAG>_ivsafe_paired_v2_seed<N>/` (weight, OOF, decision rule). `summary` writes `data/PanCancer/l12_links_1_<DATA_TAG>_ivsafe_seeds.csv` (one row per seed, then mean and sd) and keeps the earlier random-split `l12_links_1` and `spatial_k8` seed-42 rows beside it. It also prints how many folds lost an L2 class in inner training, and how many put a BRCA section there.

```bash
# l12_links_1 only, three seeds. GPU 0 is still on gate_l12_balanced; 2 and 3 are in use.
GPUS="1" bash code/PanCancer/run_l12_links_1_seeds.sh
bash code/PanCancer/run_l12_links_1_seeds.sh summary

# the one-change stack, compared with l12_links_1. Same _ivsafe tag.
INNER_VAL=cohort_safe REFERENCE=l12_links_1 SEEDS="42 43 44" \
  ARMS="l12_links_1 l12_links_1_focal l12_links_1_balanced l12_links_1p5 spatial_k8" \
  REBUILD_INDEX=0 GPUS="1" bash code/PanCancer/retrain_all.sh
INNER_VAL=cohort_safe REFERENCE=l12_links_1 bash code/PanCancer/run_ablation.sh summary
```

### cohort_safe OOF, seeds 42 and 44 (2026-11-18)

`l12_links_1` with `INNER_VAL=cohort_safe` finished for seeds 42 and 44. Seed 43 stopped in fold 1 and has no weight, so it is not in this table. Both selected checkpoints are fold 4 (`three_tier_auc_sum` 2.6370 for seed 42, 2.6447 for seed 44). The numbers below are the pooled five-fold OOF on the same 7,461,039 cells as the [21:37 table](#summary-seed-42-2026-11-17-2137), not the fold-mean rows in `data/PanCancer/l12_links_1_new1066axisfix_ivsafe_seeds.csv`. L2 accuracy and macro-F1 are pooled argmax. Decision columns are cross-fitted `decision full`. The reference row is the earlier random-split `l12_links_1` seed 42.

| Run | Best fold | L2 AUROC | L2 accuracy | L2 macro-F1 | Decision L2 accuracy | Decision L2 macro-F1 |
|--|--:|--:|--:|--:|--:|--:|
| `l12_links_1` cohort_safe, seed 42 | 4 | 0.9071 | 0.4782 | 0.3042 | 0.4988 | 0.3067 |
| `l12_links_1` cohort_safe, seed 44 | 4 | 0.9065 | 0.4711 | 0.3012 | 0.5022 | 0.3108 |
| `l12_links_1` random, seed 42 | 4 | 0.9060 | 0.4758 | 0.3016 | 0.5040 | 0.3079 |

On this pooled OOF, cohort_safe sits with the random-split run. Seed 42 is 0.0011 above it on L2 AUROC and 0.0026 above it on argmax macro-F1. Seed 44 is 0.0005 above it on L2 AUROC and has the higher decision macro-F1 (0.3108). Neither seed raises decision accuracy above the random-split 0.5040. Inner-val audit for both finished seeds: 0/5 folds lost an L2 class in inner training, and 0/5 folds put a BRCA section in inner validation.

Weights: `result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42/cross_dataset_cv/l12_links_1/best_mlp_gpu.pt` and the same path with `seed44`. OOF archives are the `oof_predictions.npz` in those two result directories. Logs: `code/log/l12_links_1_new1066axisfix_ivsafe/seed42.out`, `seed44.out`.

### Summary, seed 42 (2026-11-17, 21:37)

Ten arms have a fold-4 weight and a pooled OOF on the same 7,461,039 cells. `gate_l12_balanced` is still training and has no weight yet, so it is not in this table. `focal_gamma1` finished, including its decision rule. L2 accuracy and macro-F1 are pooled argmax. Decision columns are cross-fitted `decision full`.

| Arm | L2 AUROC | L2 accuracy | L2 macro-F1 | Decision L2 accuracy | Decision L2 macro-F1 |
|--|--:|--:|--:|--:|--:|
| `l12_links_1` | 0.9060 | 0.4758 | 0.3016 | 0.5040 | 0.3079 |
| `l12_links_2` | 0.9060 | 0.4786 | 0.3033 | 0.5032 | 0.3040 |
| `spatial_k8_gate` | 0.9052 | 0.4769 | 0.3027 | 0.5052 | 0.3086 |
| `focal_gamma1` | 0.9047 | 0.4747 | 0.3003 | 0.5061 | 0.3052 |
| `l12_balanced` | 0.9046 | 0.4632 | 0.2978 | 0.5010 | 0.3065 |
| `spatial_k8` | 0.9044 | 0.4733 | 0.2995 | 0.5000 | 0.2965 |
| `balanced_sampler` | 0.8864 | 0.3802 | 0.2532 | 0.4901 | 0.2931 |
| `adjacent_nospatial` | 0.8827 | 0.4301 | 0.2499 | 0.4579 | 0.2475 |
| `legacy_nospatial` | 0.8820 | 0.4237 | 0.2472 | 0.4573 | 0.2471 |
| `l12_balanced_nospatial` | 0.8791 | 0.4245 | 0.2456 | 0.4572 | 0.2468 |

Neighbor context is the gain, from about 0.882 without neighbors to 0.904–0.906 with k=8. `l12_links_1` and `l12_links_2` tie for the highest AUROC. `l12_links_2` has the highest argmax L2 accuracy and argmax macro-F1. `spatial_k8_gate` has the highest decision L2 macro-F1 (0.3086). `focal_gamma1` has the highest decision L2 accuracy (0.5061) and sits with the spatial arms on AUROC (0.9047). `balanced_sampler` argmax accuracy is low; its decision layer recovers most of the F1. The three no-neighbor arms stay near 0.88 and are not used for the external maps.

### 10x breast slide against the pathologist map

> 2026.11.17. The table below used `spot_diameter_fullres / 2` (65 µm disc) and argmax labels, and was computed outside the repository. It is now reproduced, with the 55 µm capture disc and four label modes (argmax, `decision_rule_l2_only.json`, EM prior shift, kNN smoothing), by
> `python code/External_val/BRCA/brca_10x_annotation_eval.py` (output `data/Visium/BRCA/pred_uni2_axisfix_figures/V1_Breast_Cancer_Block_A_Section_1_image_annotation_eval.csv`; `--radius-um 65` gives the old disc). The same cells are at the end of `Pred_visual_brca_10x_axisfix.ipynb`. Use the label mode with the highest `score` for every external map.

Ten weights with a finished prediction, slide `V1_Breast_Cancer_Block_A_Section_1_image` only. `gate_l12_balanced` is still training and is not in this table. Ground truth is `data/Visium/BRCA/10x_data/metadata.tsv`: `fine_annot_type` collapsed to IDC (1,663 spots), DCIS/LCIS (827), tumor edge (823), and healthy (485). Each in-tissue spot is scored by the majority L2 label of StarDist nuclei inside `spot_diameter_fullres / 2` (88.7 px). 157,242 of 423,162 nuclei fall in a spot. Maps: `code/External_val/BRCA/Pred_visual_brca_10x_axisfix.ipynb`.

`l12_links_1` matches the annotated regions best. It calls Tumor or DCIS on 0.77 of IDC+DCIS spots and leaves 0.81 of healthy spots free of those two labels. `spatial_k8` covers more of the cancer (0.83) and more of the healthy tissue as well (only 0.63 stays non-neoplastic). `spatial_k8_gate` is the one that names IDC as Tumor more often than as DCIS (0.43 versus 0.26); it also names 0.60 of DCIS/LCIS spots as Tumor, so the DCIS regions disappear. `balanced_sampler` and `legacy_nospatial` keep healthy tissue free of Tumor/DCIS by calling it Endothelial, and Tumor is the majority in only 0.07 and 0.09 of IDC spots.

| Arm | IDC spots called Tumor | DCIS/LCIS spots called DCIS | Healthy spots not Tumor or DCIS | IDC+DCIS spots called Tumor or DCIS | IDC vs DCIS name, when the call is Tumor or DCIS |
|--|--:|--:|--:|--:|--:|
| `l12_links_1` | 0.238 | 0.379 | 0.812 | 0.769 | 0.370 |
| `l12_links_2` | 0.079 | 0.602 | 0.721 | 0.781 | 0.323 |
| `l12_balanced` | 0.270 | 0.282 | 0.905 | 0.658 | 0.417 |
| `focal_gamma1` | 0.265 | 0.360 | 0.789 | 0.736 | 0.403 |
| `spatial_k8` | 0.285 | 0.393 | 0.634 | 0.827 | 0.388 |
| `spatial_k8_gate` | 0.434 | 0.156 | 0.682 | 0.717 | 0.476 |
| `adjacent_nospatial` | 0.143 | 0.462 | 0.777 | 0.709 | 0.351 |
| `l12_balanced_nospatial` | 0.129 | 0.593 | 0.574 | 0.825 | 0.343 |
| `balanced_sampler` | 0.072 | 0.488 | 0.967 | 0.645 | 0.326 |
| `legacy_nospatial` | 0.092 | 0.401 | 0.926 | 0.604 | 0.322 |

`focal_gamma1` does not change that ranking: it calls Tumor or DCIS on 0.74 of the cancer spots and leaves 0.79 of healthy spots free of those two labels. `l12_links_1` remains the best match to the annotated regions. `l12_links_2` raises DCIS/LCIS called DCIS to 0.60 and drops IDC called Tumor to 0.08, so the two cancer classes collapse into DCIS. `l12_balanced` leaves 0.91 of healthy spots free of Tumor and DCIS, and covers only 0.66 of the cancer spots. IDC spots are still mostly called DCIS except under `spatial_k8_gate`. Healthy spots are mostly Endothelial, not normal epithelium. Tumor-edge spots are mostly Endothelial in every arm.

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

Each job is cross-section 5-fold. It writes `result_uni2_<arch>_paired_v2_seed42/validation_internal_metrics.csv` and `oof_predictions.npz`. For A0–A7 the log line `OOF L2 macro AUROC=` is the pooled out-of-fold score on the seven training cohorts (5,627,071 cells, 896 sections). A8, the [A2 new1066 refit](#2026-11-16-a2-on-the-1066-section-index-finished), and [A7](#2026-11-17-ablation-on-the-updated-index) use the later 1,066-section run. It is not an external test set. These `1_demo_train.sh` tokens are the 896-section dispatcher. A7 here is the focal sampler, not the 1,066-section A7. The series to run now is [`run_uni2_new1066.sh`](#2026-11-17-ablation-on-the-updated-index). A file written before 2026-11-15 has labels and probabilities only; the class-order fields in the [OOF archive contract](#2026-11-15-oof-archive-contract) appear after the next train.

### 2. Choose the architecture — [`2_Pred_statistic_visual_pancancer_uni2_ablation.ipynb`](2_Pred_statistic_visual_pancancer_uni2_ablation.ipynb)

Compare A0–A3, then A5/A6/A7, on that pooled OOF L2 macro AUROC, plus the per-cohort and per-class breakdown of the same OOF cells. Those rows are the 896-section runs. A8, the A2 refit, and A7 (balanced L12, no gate) finished on the enlarged 1,066-section set; their numbers are in [A8 balanced L12](#2026-11-16-a8-balanced-l12-finished), [A2 on the 1,066-section index](#2026-11-16-a2-on-the-1066-section-index-finished), and [Ablation on the updated index](#2026-11-17-ablation-on-the-updated-index). They are not rows of this table. This notebook does not call predict, and those cohorts are not a held-out test set.

Numbers below are the **2026-10-27 `paired_v2_seed42` retrain** (named-layer init, split-seed 42, L2→L1 NLL weight 2). A 2026-10-26 morning A3 of 0.8907 used L2→L1 NLL weight 1 and those checkpoints were removed.

On the 896-section set, A3 remains the best finished architecture on pooled OOF L2 (**0.8897**). Its deployment weight is fold 0:

`data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt`

0.8897 is the five-fold pooled OOF of the A3 recipe on the 896-section index. It is not the score of this one file on every section. The series to train on the updated 1,066-section index, and which of those weights to keep, is in [Ablation on the updated index](#2026-11-17-ablation-on-the-updated-index).

To refit A2 and A3 on the current 1,066-section index without opening those directories, [`run_uni2_new1066.sh`](run_uni2_new1066.sh) sets `RUN_SUFFIX=new1066_paired_v2_seed42`. A2 and A7 have finished. A3 still writes `result_uni2_A3_adjacent_spatial_k8mean_gate_new1066_paired_v2_seed42/`.

```bash
CUDA_DEVICE=1 bash code/PanCancer/run_uni2_new1066.sh A3
```

### 2026-11-16 A2 on the 1,066-section index (finished)

Same recipe as the published A2: adjacent loss, L2→L1 NLL weight 2, spatial k=8 mean, **no gate**, L12 class weights `none`. The training index is the enlarged one used by A8: 1,066 sections, 7,461,039 cells. This run does not overwrite `result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/` (896 sections, pooled OOF L2 0.8894, fold 1).

Finished 2026-11-16 23:16 (start 21:45, wall 1:31:22, GPU 0). Five folds, then `he_validate`.

| | index | spatial | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|-------|---------|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| published A2 | 896 sections, 5,627,071 cells | k=8 mean | 0.8894 | 0.8798 | 0.8272 | 0.2941 ± 0.030 | 0.4181 ± 0.042 | 1 |
| A8 `l12_balanced` | 1,066 sections, 7,461,039 cells | k=8 mean + gate, balanced L12 | 0.9002 | 0.8991 ± 0.006 | 0.8482 ± 0.004 | 0.2953 ± 0.008 | 0.4281 ± 0.010 | 0 |
| **A2 new1066** | 1,066 sections, 7,461,039 cells | k=8 mean | **0.9053** | 0.8997 | 0.8547 | 0.2846 ± 0.0175 | 0.4172 ± 0.0124 | 4 |

Fold L2 AUROC is 0.9135, 0.8926, 0.8899, 0.8985, 0.9039. Fold L12 AUROC is 0.8614, 0.8443, 0.8413, 0.8554, 0.8710. Best epochs are 7, 7, 7, 13, 8. Fold 4 is the selected weight (`three_tier_auc_sum` 2.6503, best epoch 8).

0.9053 and A8's 0.9002 are the same 7,461,039 OOF cells. 0.9053 is not a comparison with the published A2's 0.8894, which is the 896-section mix. On the matched index the no-gate A2 is 0.005 above A8 on pooled L2. Its L2 macro-F1 (0.2846) is below A8 (0.2953).

Argmax on those same OOF cells. `B_cell` is 148,662 (1.99%) and `Plasma` is 69,571 (0.93%). L12 `B_Plasma` is 218,233 (2.92%). A8's balanced L12 raises that family's recall and over-calls it (404,270 predicted). This A2 leaves L12 unweighted: recall stays 0.186 and the head predicts 91,654 cells.

| L2 / L12 | A2 new1066 AUROC | A8 AUROC | A2 new1066 recall | A8 recall | A2 new1066 F1 | A8 F1 | A2 new1066 n_pred | A8 n_pred |
|----------|-----------------:|---------:|------------------:|----------:|--------------:|--------:|------------------:|----------:|
| `B_cell` | 0.854 | 0.857 | 0.320 | **0.370** | 0.307 | 0.300 | 161,263 | 219,081 |
| `Plasma` | **0.901** | 0.895 | 0.239 | **0.299** | 0.187 | 0.186 | 108,200 | 154,200 |
| L12 `B_Plasma` | **0.851** | 0.846 | 0.186 | **0.408** | 0.263 | **0.286** | 91,654 | 404,270 |

Macrophage stays the weak common class (AUROC 0.687, recall 0.072, 224,901 predicted against 946,950 true). Endothelial argmax is 262,680 (recall 0.184), in the same range as the published A2 and not the A7 blow-up.

Weight: fold 4 `data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_new1066_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt`. Log: `code/log/pancancer_section_uni2_A2_adjacent_spatial_k8mean_new1066_paired_v2_seed42.log`. OOF archive: `oof_predictions.npz`.

NPC maps: [`Pred_visual_npc_uni2_A2_new1066.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A2_new1066.ipynb), predict `code/External_val/NPC/demo_uni2_A2_new1066.sh`. Prostate Patient_1 H2: [`Pred_visual_prostate_patient1_H2_uni2_A2_new1066.ipynb`](../External_val/Prostate/Pred_visual_prostate_patient1_H2_uni2_A2_new1066.ipynb), predict `code/External_val/Prostate/demo_uni2_A2_new1066.sh`. Both write `pred_uni2_A2_new1066_paired_v2_seed42/` and leave the published A2 prediction trees in place.

### 2026-11-17 Balanced L12 on the A2 trunk (finished; public name A7)

The command token `A2L12` is retired. In the [updated-index ablation](#2026-11-17-ablation-on-the-updated-index) this run is **A7**. The weight directory below is unchanged.

A7 is the A8 loss change on the A2 trunk: adjacent loss, L2→L1 NLL weight 2, spatial k=8 mean, **no gate**, `--l12-class-weight-mode balanced`. It does not open the finished A8 directory (A3 trunk + gate), the published 896-section A2 directory, or the finished new1066 A2 directory.

The first fp16 attempt died on fold 2 epoch 7 (`RuntimeError: No finite optimizer update`). That fold's Neural L12 weight is 448. The finished run is `HIST2PHENO_AMP=bf16 CUDA_DEVICE=2 bash code/PanCancer/run_uni2_new1066.sh A7`. Start 2026-11-16 23:57, wall 1:20:25, end 2026-11-17 01:18, GPU 2. The log has no non-finite AMP warning. Fold 2 early-stopped at epoch 16 with its best checkpoint at epoch 6. The splitter is the default `cross_dataset_section`. Do not rerun that command after the 2026-11-16 code update: `model.py` is in the checkpoint protocol hash, so a rerun would retrain all five folds into the same directory. The numbers below stay the result for this split.

| | index | spatial | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|-------|---------|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| A2 new1066 | 1,066 sections, 7,461,039 cells | k=8 mean | 0.9053 | 0.8997 | 0.8547 | 0.2846 ± 0.0175 | 0.4172 ± 0.0124 | 4 |
| **A7** | same cells | k=8 mean, balanced L12, bf16 | **0.9035** | 0.8974 ± 0.009 | 0.8456 ± 0.011 | 0.2815 ± 0.0181 | 0.4249 ± 0.0145 | 4 |
| A8 `l12_balanced` | same cells | k=8 mean + gate, balanced L12 | 0.9002 | 0.8991 ± 0.006 | 0.8482 ± 0.004 | 0.2953 ± 0.008 | 0.4281 ± 0.010 | 0 |

Fold L2 AUROC is 0.9084, 0.8910, 0.8867, 0.8986, 0.9025. Fold L12 AUROC is 0.8401, 0.8385, 0.8348, 0.8524, 0.8621. Best epochs are 12, 11, 6, 13, 8. Fold 4 is the selected weight (`three_tier_auc_sum` 2.6384, best epoch 8).

0.9035, A2 new1066's 0.9053, and A8's 0.9002 are the same 7,461,039 OOF cells. Balanced L12 on the no-gate trunk lands between them on pooled L2. It moves L12 `B_Plasma` the way A8 does: recall 0.409 and 433,874 predicted cells, against A2 new1066's recall 0.186 and 91,654 predicted cells. `B_cell` is 148,662 true cells and `Plasma` is 69,571.

| L2 / L12 | A7 AUROC | A2 new1066 AUROC | A8 AUROC | A7 recall | A7 F1 | A7 n_pred |
|----------|------------:|-----------------:|---------:|-------------:|---------:|-------------:|
| `B_cell` | 0.842 | 0.854 | 0.857 | 0.373 | 0.274 | 256,124 |
| `Plasma` | 0.898 | 0.901 | 0.895 | 0.306 | 0.177 | 171,277 |
| L12 `B_Plasma` | 0.841 | 0.851 | 0.846 | 0.409 | 0.274 | 433,874 |

Macrophage stays weak (AUROC 0.682, recall 0.048, 152,272 predicted against 946,950 true). NK is over-called (113,346 predicted against 11,975 true).

Weight: fold 4 `data/PanCancer/result_uni2_A2_l12_balanced_new1066_paired_v2_seed42/cross_dataset_cv/A2_l12_balanced/best_mlp_gpu.pt`. Log: `code/log/pancancer_section_uni2_A2_l12_balanced_new1066_paired_v2_seed42.log`. OOF archive: `oof_predictions.npz`.

External maps write `pred_uni2_A7_new1066_paired_v2_seed42/` and leave the A2 and A8 trees in place.

| Cohort | Notebook | Predict |
|--------|----------|---------|
| NPC | [`Pred_visual_npc_uni2_A7.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A7.ipynb) | `CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A7.sh predict` |
| Prostate Patient_1 H2 | [`Pred_visual_prostate_patient1_H2_uni2_A7.ipynb`](../External_val/Prostate/Pred_visual_prostate_patient1_H2_uni2_A7.ipynb) | `SAMPLE=Patient_1__H2 bash code/External_val/Prostate/demo_uni2_A7.sh predict` |
| BRCA | [`Pred_visual_brca_uni2_A7.ipynb`](../External_val/BRCA/Pred_visual_brca_uni2_A7.ipynb) | `bash code/External_val/BRCA/demo_uni2_A7.sh predict` |

### 2026-11-16 L2-balanced section folds (`stratified_section`)

Optional splitter. Each section is still entirely train or validation. Folds are assigned so every L2 class, every cohort, and the total cell count stay close to a 1/5 share. Rare classes are weighted the same as Tumor. `SPLITTER=stratified_section` in [`1_demo_train.sh`](1_demo_train.sh) appends `stratsec_` to the result suffix, so it does not open `cross_dataset_section` directories, including the finished A7 run. A result directory refuses a `fold_plan.csv` written by the other family.

```bash
SPLITTER=stratified_section HIST2PHENO_AMP=bf16 CUDA_DEVICE=2 bash code/PanCancer/run_uni2_new1066.sh A7
```

That writes `result_uni2_A2_l12_balanced_stratsec_new1066_paired_v2_seed42/`. Pass `--patient-map` to keep a patient's sections in one fold; without it, sections of one patient can still land in different folds, as in the default section CV. `fold_plan_l2_balance.csv` records each class's held-out share (ideal 0.2).

Finished 2026-11-17 06:03 (start 04:42, wall 1:21:36, GPU 2, bf16). Same 1,066 sections and 7,461,039 cells as the `cross_dataset_section` A7 run. Pooled OOF L2 is **0.8811**. Fold-mean L2 AUROC is 0.8911. Fold L2 AUROC is 0.8919, 0.9112, 0.8725, 0.8881, 0.8917. L2 macro-F1 is 0.2687 ± 0.0267. L1 macro-F1 is 0.4214 ± 0.0167. Fold 1 is the selected weight (`three_tier_auc_sum` 2.6287, best epoch 8).

0.8811 is the same cells under a different split from the `cross_dataset_section` A7 score of 0.9035. The two weights are not interchangeable.

Weight: fold 1 `data/PanCancer/result_uni2_A2_l12_balanced_stratsec_new1066_paired_v2_seed42/cross_dataset_cv/A2_l12_balanced/best_mlp_gpu.pt`. Log: `code/log/pancancer_section_uni2_A2_l12_balanced_stratsec_new1066_paired_v2_seed42.log`. NPC maps: [`Pred_visual_npc_uni2_A7_stratsec.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A7_stratsec.ipynb), predict `CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A7_stratsec.sh predict`. Output tree: `data/Visium/NPC/pred_uni2_A7_stratsec_new1066_paired_v2_seed42/`.

### 2026-11-16 Training L2 label mix

Ground-truth L2 counts on the 5,627,071 training cells (896 sections, seven cohorts). Counted from `l2_labels` in `data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/oof_predictions.npz`. Each cell is held out once, so this file is the full training label mix, not an A2 prediction. Finished A0–A7 checkpoints used these cells. A8, and any later prepare, also trains on the SPORA sections that are now part of CRC and GBM (140 Schürch cores + 30 Hoefflin sections; 1,066 sections once `section_index.csv` has been appended). Column order is the workbook L2 order (`ontology_class_names`); this archive has no `l2_class_names` key.

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

### 2026-11-16 A8 balanced L12 (finished)

Same UNI2 cache and the **A3** trunk: k=8 mean, learned spatial gate, scalar loss weights unchanged. A7 changed the sampler and turned class weights off; A8 does not. The only change from A3 is `--l12-class-weight-mode balanced`.

| | spatial | CE(L12) weight | L2→L12 / L12→L1 NLL | L12 class weights |
|--|---------|----------------|---------------------|-------------------|
| A3 (reference) | k=8 mean + gate | 1, unweighted classes | 0.5 / 0.5, unweighted | `none` |
| A8 `l12_balanced` | k=8 mean + gate | 1, **balanced** | 0.5 / 0.5, **balanced** | `balanced` |

L2→L1 NLL stays 2. Both JS terms stay 0.1 and stay unweighted. L1 and L2 cross-entropy stay `balanced`, as in A3. The L12 weights are inverse-frequency on the eight L12 classes, so `B_Plasma` is no longer pulled back to its 3.5% prior by an unweighted L12 term. Mechanism and the `none` default that leaves A0–A7 families unchanged: [`code/Hist2Pheno_pkg/README.md`](../Hist2Pheno_pkg/README.md#2026-11-16-a8-class-weighted-l12-opt-in).

Finished 2026-11-16 20:33 on one GPU (start 18:25, wall 2:08:34). Five folds, then `he_validate`. The run is the enlarged index: 1,066 sections and 7,461,039 OOF cells (`validation_internal_metrics.csv`). CRC is 3 Xenium sections plus 140 Schürch cores. GBM is 2 WangLab sections plus 30 Hoefflin sections. A0–A7 above are the 896-section, 5,627,071-cell runs, so 0.9002 is not a gain over A3's 0.8897.

| | spatial | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|---------|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| A8 `l12_balanced` | gate | **0.9002** | 0.8991 ± 0.006 | 0.8482 ± 0.004 | 0.2953 ± 0.008 | 0.4281 ± 0.010 | 0 |

Fold L2 AUROC is 0.9086, 0.8999, 0.8891, 0.8985, 0.8995. Best epochs are 7, 14, 5, 6, 8. Fold 0 is the selected weight (`three_tier_auc_sum` 2.6139, best epoch 7).

The label mix moved with the new sections. Tumor is 2,188,084 cells, **29.3%** of this OOF, against 14.94% on the 896-section mix. `B_cell` is 148,662 (1.99%) and `Plasma` is 69,571 (0.93%); together L12 `B_Plasma` is 218,233 (2.92%), down from 3.50% on the old mix. Macro AUROC gives each class one vote, so the extra Tumor cells do not by themselves raise the 0.9002.

Balanced L12 on this OOF (argmax of the held-out probabilities):

| L2 / L12 | AUROC | precision | recall | F1 | n | n_pred |
|----------|------:|----------:|-------:|---:|--:|-------:|
| `B_cell` | 0.857 | 0.251 | 0.370 | 0.300 | 148,662 | 219,081 |
| `Plasma` | 0.895 | 0.135 | 0.299 | 0.186 | 69,571 | 154,200 |
| L12 `B_Plasma` | 0.846 | 0.220 | 0.408 | 0.286 | 218,233 | 404,270 |

`B_Plasma` recall is 0.408 and the head predicts 404,270 cells against 218,233 true (1.85×). Precision stays 0.220. The same weights over-call other rare L2 classes: NK 140,394 predicted against 11,975 true, Injury_epithelial 88,156 against 14,090, Neural 46,315 against 10,848. Macrophage is the weak common class (AUROC 0.691, recall 0.055, 150,782 predicted against 946,950 true). L12 Myeloid AUROC is 0.705.

Weight: fold 0 `data/PanCancer/result_uni2_A8_l12_balanced_paired_v2_seed42/cross_dataset_cv/A8_l12_balanced/best_mlp_gpu.pt`. This is the selected checkpoint of the 1,066-section run. The published A3 file remains the 896-section deployment weight. Log: `code/log/pancancer_section_uni2_A8_l12_balanced_paired_v2_seed42.log`. OOF archive: `oof_predictions.npz` (class names included). NPC predict for this weight is `code/External_val/NPC/demo_uni2_A8.sh`. Maps: [`Pred_visual_npc_uni2_A8.ipynb`](../External_val/NPC/Pred_visual_npc_uni2_A8.ipynb).

This run writes its own result directory. CRC in this run is the 3 Xenium sections plus 140 Schürch TMA cores. GBM is the 2 WangLab sections plus 30 Hoefflin sections. The seven column names stay Lung, BRCA, CRC, HCC, PDAC, GIST, GBM. Prepare appends any of those sections missing from `data/PanCancer/section_index.csv` and leaves existing section ids in place. Outputs:

- `data/PanCancer/result_uni2_A8_l12_balanced_paired_v2_seed42/`
- checkpoint directory `cross_dataset_cv/A8_l12_balanced/`
- log `code/log/pancancer_section_uni2_A8_l12_balanced_paired_v2_seed42.log`

One GPU trains the five folds one after another:

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
CUDA_DEVICE=0 bash code/PanCancer/1_demo_train.sh uni2 A8
```

Four GPUs train four folds at once, then the first GPU trains fold 4 and writes the OOF report. Wall time is about two folds plus that report. Each process still loads the full cell table. Run `refresh-schurch` first if Schürch `CD8+ T cells` and `plasma cells` should be in this fit. The shared launcher is [`run_uni2_4gpu.sh`](run_uni2_4gpu.sh); [`run_a8_4gpu.sh`](run_a8_4gpu.sh) calls it with `A8`.

```bash
bash code/spora/run_spora_pancancer.sh refresh-schurch
bash code/PanCancer/run_a8_4gpu.sh
```

`GPUS="0 1 2 3"` changes the device list. `--only-folds N` is how one process trains a single fold. Fold 0 starts alone and the others wait until its log contains `context cache ready`. Folds 1–3 then occupy the other three GPUs. Fold 4 starts on the first GPU after fold 0 exits. The last step has no `--only-folds`: it loads the five `hce_group_fold_*.pt` files, writes `best_mlp_gpu.pt`, and runs `he_validate`. A fold file that already exists is not retrained. Logs are `code/log/pancancer_section_uni2_A8_l12_balanced_paired_v2_seed42_{prepare,fold0,fold1,fold2,fold3,fold4,finalize}.log`. The finished run above used the one-GPU command, not this four-GPU launcher.

### 2026-11-16 A9–A11 Hist2Prot auxiliary heads (removed)

Removed from the ablation on 2026-11-17. The neighbor head (A9) and the tissue head (A11) did not improve L2 macro-F1 over the gated model (0.2828 and 0.2796, against 0.2953). A10 died in fold 2 and never wrote a weight. `run_uni2_new1066.sh A9|A10|A11`, `run_a9_4gpu.sh`, `run_a10_4gpu.sh`, `run_a11_4gpu.sh`, and `code/External_val/NPC/demo_uni2_A9.sh` / `A10` / `A11` now exit. The A9 and A11 result directories were deleted. The numbers below are why.

A9, A10, and A11 kept the A8 trunk (A3 spatial gate, balanced L12). Each added a training-time head on the same fused feature. The extra loss weight was 0.2.

| | added head | target | extra loss |
|--|------------|--------|------------|
| A9 `nbr_comp` | neighbour L2 composition | histogram of the k=8 neighbours, self excluded | soft CE, weight 0.2 |
| A10 `nbr_comp_geom` | same as A9 | same, after converting each section to µm | same, plus a 50 µm neighbour cap and log count / distance / density |
| A11 `nbr_tissue` | A9 plus tissue compartment | 7 classes from the GT L12 mix of the 32 nearest cells (self included) | both weights 0.2; compartment CE is class-weighted |

A9 finished 2026-11-16 21:24 (GPU 1, start 18:25, wall 2:59:06). A11 finished 21:23 (wall 2:21:47). A10 stopped in fold 2 and has no pooled OOF.

| | pooled OOF L2 | fold-mean L2 AUROC | fold-mean L12 AUROC | L2 macro-F1 | L1 macro-F1 | `best_mlp` fold |
|--|---------------|--------------------|---------------------|-------------|-------------|-----------------|
| A8 `l12_balanced` | 0.9002 | 0.8991 ± 0.006 | 0.8482 ± 0.004 | **0.2953 ± 0.008** | 0.4281 ± 0.010 | 0 |
| A9 `nbr_comp` | **0.9008** | 0.8972 ± 0.009 | 0.8473 ± 0.009 | 0.2828 ± 0.021 | 0.4262 ± 0.024 | 4 |
| A11 `nbr_tissue` | 0.8959 | 0.8977 ± 0.009 | 0.8474 ± 0.008 | 0.2796 ± 0.022 | 0.4197 ± 0.019 | 4 |

A9 fold L2 AUROC is 0.9106, 0.8925, 0.8831, 0.8979, 0.9018. Best epochs are 12, 12, 3, 13, 8. Fold 4 is the selected weight (`three_tier_auc_sum` 2.6348). On the same cells the neighbour head leaves pooled L2 level with A8 (0.9008 vs 0.9002) and lowers L2 macro-F1 (0.2828 vs 0.2953). `B_Plasma` recall is 0.421 (A8 0.408) with 439,836 predictions against 218,233 true. Macrophage AUROC is 0.684 (A8 0.691) and L12 Myeloid is 0.696. The log has 4,773 non-finite AMP skips, starting in fold 1; each fold still kept its best epoch and the pooled file was written.

A11 fold L2 AUROC is 0.9122, 0.8945, 0.8831, 0.8981, 0.9006. Best epochs are 12, 7, 3, 15, 5. Fold 4 is selected. Pooled L2 is 0.004 below A8. `B_Plasma` recall is 0.417 (450,531 predicted). Macrophage AUROC is 0.682. The tissue head does not lift the cell-type OOF.

A10 saved folds 0 and 1, then fold 2 epoch 13 had no finite batch and the process exited (`No finite optimizer update in this epoch`). Fold 0 best `three_tier_auc_sum` is 2.5985 at epoch 12. Fold 1 is 2.6036 at epoch 7. There is no `best_mlp_gpu.pt` and no `oof_predictions.npz`. The log has 3,570 non-finite AMP skips. Do not deploy this partial run.

The scores above were computed before the weights were deleted. Do not retrain these heads.

Logs, still on disk: `code/log/pancancer_section_uni2_A9_nbr_comp_paired_v2_seed42.log`, `...A10_nbr_comp_geom...`, `...A11_nbr_tissue...`.

Compartment order was Tumor, Epithelium, Lymphoid, Myeloid, Stroma, Vascular, Mixed. A compartment won at share 0.5 (Vascular at 0.35); otherwise the label was Mixed.

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

## 2026-11-17 Ablation on the updated index

> 2026-11-17: finished series, kept. The SPORA tables it used already match the instance-mask axis. New runs use `retrain_all.sh` / `run_ablation.sh` (section above) and write `result_uni2_ablation_*`.

The updated training index is 1,066 sections (Lung 25, BRCA 2, CRC 143, HCC 36, PDAC 278, GIST 550, GBM 32). Do not train these on the old 896-section index. Every command below is `run_uni2_new1066.sh`, which writes a `new1066_` directory and does not open the 896-section results.

**A2** is the model to use: adjacent loss, k=8 mean, no gate. Each later id changes one thing from A2. A0 and A1 are the two steps that build up to A2. k=8 attention was removed because the gather is too expensive. The neighbor-composition head and the tissue-compartment head were removed: pooled L2 moved by +0.0006 and −0.004, and L2 macro-F1 fell (0.2828 and 0.2796, against 0.2953 for the gated model). Their result directories were deleted.

| id | one change | internal recipe | on 1,066 sections |
|--|--|--|--|
| A0 | legacy loss, no neighbors | A0 | not trained — run |
| A1 | adjacent loss, no neighbors | A1 | not trained — run |
| **A2** | k=8 mean | A2 | **finished, 0.9053. Do not rerun.** |
| A3 | A2 + spatial gate | A3 | not trained — run |
| A4 | A2, L12 link weights 1 | A5 | not trained — run |
| A5 | A2, L12 link weights 2 | A6 | not trained — run |
| A6 | A2, balanced sampler + focal | A7 | not trained — run |
| A7 | A2 + balanced L12, no gate | `A2_l12_balanced` | **finished, 0.9035. Do not rerun.** NPC tumor region goes to Fibroblast. |
| **A8** | A3 + balanced L12 | A8 | **finished, 0.9002.** Weight: `result_uni2_A8_l12_balanced_paired_v2_seed42/`. |

Run one at a time. A finished id prints the weight and exits. `1_demo_train.sh uni2 A8` without this launcher writes the 896-style suffix; do not use it for this index.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A0
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A1
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A2
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A3
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A4
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A5
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A6
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A7
CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A8
```

A2, A7, and A8 exit immediately. A0, A1, A3, A4, A5, and A6 train. New weights land in `data/PanCancer/result_uni2_<recipe>_new1066_paired_v2_seed42/`. If a run prints `non-finite loss`, stop it and rerun that id with `HIST2PHENO_AMP=bf16`.

### Worth keeping

Checked every `data/PanCancer/result_*` directory. A run counts as finished only when `best_mlp_gpu.pt` and `oof_predictions.npz` are both present. Pooled L2 is macro OvR AUROC on out-of-fold cells. The 896-section numbers (5,627,071 cells) and the 1,066-section numbers (7,461,039 cells) are different cohorts and are not comparable.

| model | index | recipe | pooled OOF L2 | L2 macro-F1 | why it stays |
|--|--|--|--|--|--|
| **A2** | 1,066 | k=8 mean, no gate | **0.9053** | 0.2846 | The model to use. Highest pooled L2 on the current index. On NPC `20210809-C-AH4199551` the cell2location tumor region stays Tumor (55%). Weight: `result_uni2_A2_adjacent_spatial_k8mean_new1066_paired_v2_seed42/.../best_mlp_gpu.pt` (fold 4). |
| **A8** | 1,066 | k=8 mean + gate, balanced L12 | 0.9002 | **0.2953** | Best L2 macro-F1 on that index. External notebooks and `code/Internal_val/Pred_visual_brca.ipynb` read this weight. On the same NPC tumor region, 63% of nuclei are Tumor. Weight: `result_uni2_A8_l12_balanced_paired_v2_seed42/.../best_mlp_gpu.pt` (fold 0). |
| **published A3** | 896 | k=8 mean + gate | **0.8897** | **0.3013** | Best finished model on the original seven-cohort index. Do not overwrite. Weight: `result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/.../best_mlp_gpu.pt` (fold 0). |
| **published A2** | 896 | k=8 mean, no gate | 0.8894 | 0.2941 | Tied with A3 and still the default of the older external and internal notebooks. Do not overwrite. Weight: `result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/.../best_mlp_gpu.pt` (fold 1). |

Use **A2** for the updated index. Use **A8** when the comparison needs the gated, class-weighted L12 model. The two 896-section weights stay on disk so the old figures still open; new training does not use them.

### Finished, not worth deploying

These directories were left in place. Each is about 1.2–1.7 GB, and they are real trained weights.

| model | pooled OOF L2 | reason to leave it out of deployment |
|--|--|--|
| 896 A5 `l12_links_1` | 0.8894 | L12-link ablation. Tied with published A2, below A3. Public id on the new index is A4, not yet trained. |
| 896 A6 `l12_links_2` | 0.8882 | Same. L1 macro-F1 is higher; L2 is not. Public id on the new index is A5. |
| A0 | 0.8580 | No spatial context. |
| A1 | 0.8601 | Adjacent loss, still no spatial context. |
| 896 focal `A7_bal_focal` | 0.8581 | Sampler + focal. Level with A0 and 0.031 below A2. Endothelial predictions blow up. Public id on the new index is A6. |
| A7 | 0.9035 | Balanced L12 on A2, no gate, bf16. Between A2 and A8 on AUROC, but on the NPC slide it calls the tumor region Fibroblast (46%, Tumor only 11%). Do not use this weight for external figures. |
| UNI A2, Virchow2 A2 | — | Finished encoder controls (`result_uni_A2`, `result_virchow2_A2`), not the UNI2 choice above. |
| `result_all_spatial` | — | Older spatial run, 10 fold files, no `oof_predictions.npz`. Kept because a `best_mlp_gpu.pt` is there. |

A3 has not been refit on the 1,066-section index. That directory does not exist.

### Stratified A7 — do not retrain

`SPLITTER=stratified_section HIST2PHENO_AMP=bf16` A7 finished 2026-11-17 06:03. Weight: `result_uni2_A2_l12_balanced_stratsec_new1066_paired_v2_seed42/cross_dataset_cv/A2_l12_balanced/best_mlp_gpu.pt` (fold 1, pooled OOF L2 0.8811). This is a different split from the finished `cross_dataset_section` A7 weight. Do not start a second one.

### Removed on 2026-11-17

These were not finished models. Scaled cache went from 471 GB to 35 GB, context cache from 749 GB to 43 GB. The two scaled files and the `94c57d90…` neighbour cache were left in place, as was `label_cache/pooled_features/X_91d402cb….npy` (43 GB), because the stratsec job still had them open at cleanup time. That job finished at 06:03.

- Unused files in `data/PanCancer/scaled_cache/` and `data/PanCancer/context_cache/`.
- `result_uni2_A10_nbr_comp_geom_paired_v2_seed42/` — fold 2 died with no finite batch. No `best_mlp_gpu.pt`, no OOF.
- `result_uni2_A9_nbr_comp_paired_v2_seed42/` and `result_uni2_A11_nbr_tissue_paired_v2_seed42/` — finished, then deleted. L2 macro-F1 was 0.2828 and 0.2796, both below the gated model (0.2953).
- `result_uni2_A2_spora9_paired_v2_seed42/` — only a `fold_plan.csv`.
- `result_smoke/` — BRCA/GBM smoke test.

Left on disk on purpose: `data/PanCancer/uni2_aligned/` (161 GB, the UNI2 tokens every later train and predict reads) and the Visium prediction trees. Next spatial train or predict will rebuild the neighbour cache it needs.

