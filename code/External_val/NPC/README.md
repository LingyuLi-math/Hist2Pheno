# NPC external validation (pan-cancer cross-section Hist2Pheno)

2026.10.06. Two HE slides, StarDist nuclei, no 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. The shared steps live in `code/Hist2Pheno_pkg/external_validation.py` (`_SIMPLE_COHORTS["NPC"]`).

| Sample | HE | StarDist nuclei |
|--------|----|-----------------|
| `20210809-C-AH4199551` | `data/Visium/NPC/HE_images/20210809-C-AH4199551.tif` (17319 × 17351) | 142,184 |
| `20210809-D-AJ3037946` | `data/Visium/NPC/HE_images/20210809-D-AJ3037946.tif` (17290 × 17349) | 120,888 |

StarDist tables: `data/Visium/NPC/StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv` with columns `centroid_x`, `centroid_y`, `probability`. Those centroids are full-resolution pixels on the same TIFF.

## Pixel size: 0.465 µm/px, not 96 dpi

Read with `tifffile` from each page's `ImageDescription` (OME-XML). Both files contain:

```text
PhysicalSizeX="0.46507936507936531" PhysicalSizeXUnit="µm"
PhysicalSizeY="0.4650793650793652"  PhysicalSizeYUnit="µm"
```

X and Y differ only in the last digit, so both slides are **0.465 µm/px**. `external_validation.py` parses `PhysicalSizeX` / `PhysicalSizeY` per TIFF (`ome_um_per_px`); it does not hard-code the number.

The same IFD also has TIFF tags `XResolution = 96`, `YResolution = 96`, `ResolutionUnit = inch`. That is a 96 dpi placeholder (~264 µm/px) and does not match a ~17k slide. It is not used.

A native 16 px crop on this grid is `16 × 0.465 = 7.44 µm`. Pan-cancer UNI crops were **8 µm** (16 px at 0.5 µm/px: CODEX-style `scale_image=False`, or Xenium rescaled with `0.2125/0.5`). Using 16 native pixels here would feed a systematically smaller field.

## Crop used for UNI

Rescale the HE and the StarDist centroids by `0.465 / 0.5 ≈ 0.930` so the working grid is **0.5 µm/px**, then take `patch_size=16`. The physical field is **8 µm**, the same as training. No Visium spot diameter is involved.

`Image_feature_extraction.py` treats these `*.tif` files as OME (the XML sits in `ImageDescription` even though the suffix is not `.ome.tif`) and resizes in uint8. A float32 rescale of a ~17k RGB slide would need several GB.

Coordinates stay the original CSV pixels in the `.pth` filename (`{x_scaled}/scale`), so the later h5ad match uses `centroid_x` / `centroid_y` at 1 px tolerance. `obsm['spatial_HE']` is that same native pixel grid. Neighbor identity for k=8 is unchanged by a uniform scale.

## Checkpoint

Cross-section weights, not the cell-stratified run:

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial context k=8, same as training. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root.

```bash
bash code/External_val/NPC/demo.sh dry-run    # mpp, scale, nucleus counts
bash code/External_val/NPC/demo.sh features   # UNI via code/Image_feature_extraction.py
bash code/External_val/NPC/demo.sh h5ad
bash code/External_val/NPC/demo.sh predict    # h5ad + l2_probs.npz
# or
bash code/External_val/NPC/demo.sh all        # features, h5ad, predict
```

One slide: `SAMPLE=20210809-C-AH4199551`. Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0`.

Screen output is appended to `code/External_val/NPC/NPC.log`. Each step and the whole run record elapsed time there (`00:26:14 (1574s)`).

## Outputs

| Path | Role |
|------|------|
| `data/Visium/NPC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/NPC/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/NPC/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
`Pred_visual_npc.ipynb` draws the spatial maps, composition bars, and TLS / SRI / TNI indices. `SAVE_FIGURES = False` only displays them. There is no ROC: these slides have no ground-truth cell type.

UNI2 maps reuse `project_all_UNI2` and a separate checkpoint:

| Weight | Predict | Notebook |
|--------|---------|----------|
| A2 k=8 mean | `demo_uni2.sh` | `Pred_visual_npc_uni2_A2.ipynb` |
| A2 new1066, k=8 mean, 1,066 sections | `demo_uni2_A2_new1066.sh` | `Pred_visual_npc_uni2_A2_new1066.ipynb` |
| A3 k=8 mean+gate | `demo_uni2_A3.sh` | `Pred_visual_npc_uni2_A3.ipynb` |
| A5 L12 links = 1 | `demo_uni2_A5.sh` | `Pred_visual_npc_uni2_A5.ipynb` |
| A6 L12 links = 2 | `demo_uni2_A6.sh` | `Pred_visual_npc_uni2_A6.ipynb` |
| A8 balanced L12, k=8 mean+gate | `demo_uni2_A8.sh` | `Pred_visual_npc_uni2_A8.ipynb` |
| A7 balanced L12, k=8 mean, no gate, 1,066 sections | `demo_uni2_A7.sh` | `Pred_visual_npc_uni2_A7.ipynb` |
| A7 `stratified_section`, same recipe, fold 1 | `demo_uni2_A7_stratsec.sh` | `Pred_visual_npc_uni2_A7_stratsec.ipynb` |

Predict writes `{sample}_stardist_pred.h5ad` and `{sample}_l2_probs.npz`. `obs['conf_L2']` is the max softmax of the argmax class. The notebooks plot `pred_CT`. Confidence maps (below) are extra; they do not replace `pred_CT`.

`Pred_visual_npc_uni2_A8_validation.ipynb` checks one of those 1,066-section weights at a time. Set `MODEL` in the setup cell to `A2_new1066`, `A7`, `A7_stratsec`, or `A8`. The spot table adds a `Hist2Pheno decision rule` column when that result directory contains `decision_rule_l2_only.json` (`posthoc.py fit-decision` in the PanCancer README). The column is the same whole-slide argmax comparison as `Hist2Pheno argmax`. `pred_CT` stays the checkpoint argmax. Predict the chosen tree before opening the notebook:

```bash
SAMPLE=20210809-C-AH4199551 CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A2_new1066.sh predict
SAMPLE=20210809-C-AH4199551 CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A7.sh predict
SAMPLE=20210809-C-AH4199551 CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A7_stratsec.sh predict
SAMPLE=20210809-C-AH4199551 CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_A8.sh predict
```

## 2026-11-17 Patient 1, finished ablation weights

The comparison for the main text uses the ablation names, not the A0–A8 ids above. It is patient 1 only (`20210809-C-AH4199551`). The weights are the finished arms in `result_uni2_ablation_<arm>_new1066axisfix_paired_v2_seed42` (fold 4). `focal_gamma1` is included. `gate_l12_balanced` is still training and is not in this list.

| Arm | What changed | Prediction tree |
|-----|----------------|-----------------|
| `legacy_nospatial` | old loss, no neighbors | `pred_uni2_axisfix_legacy_nospatial` |
| `adjacent_nospatial` | adjacent loss, no neighbors | `pred_uni2_axisfix_adjacent_nospatial` |
| `spatial_k8` | adjacent + k=8 mean (trunk) | `pred_uni2_axisfix_spatial_k8` |
| `spatial_k8_gate` | trunk + spatial gate | `pred_uni2_axisfix_spatial_k8_gate` |
| `l12_links_1` | trunk, L12 link weights = 1 | `pred_uni2_axisfix_l12_links_1` |
| `l12_links_2` | trunk, L12 link weights = 2 | `pred_uni2_axisfix_l12_links_2` |
| `l12_balanced` | trunk + balanced L12 | `pred_uni2_axisfix_l12_balanced` |
| `focal_gamma1` | trunk + focal γ=1, class weights kept | `pred_uni2_axisfix_focal_gamma1` |
| `balanced_sampler` | trunk + sampler, class weight off | `pred_uni2_axisfix_balanced_sampler` |
| `l12_balanced_nospatial` | balanced L12, no neighbors | `pred_uni2_axisfix_l12_balanced_nospatial` |

```bash
CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh predict
CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh predict spatial_k8
CUDA_DEVICE=2 bash code/External_val/NPC/demo_uni2_axisfix.sh predict-ivsafe
```

`predict-ivsafe` writes the cohort_safe `l12_links_1` seeds 42 and 44 (selected fold 4; pooled OOF L2 0.9071 and 0.9065) to `pred_uni2_axisfix_l12_links_1_ivsafe_seed42` and `pred_uni2_axisfix_l12_links_1_ivsafe_seed44`. Those trees are not the earlier random-split `pred_uni2_axisfix_l12_links_1`. The notebook lists both.

2026-11-18. The same notebook also lists the five-fold means from [`predict_fold_ensemble.sh`](../predict_fold_ensemble.sh): `ens5_l12_links_1_ivsafe_seed42` and `ens5_l12_links_1_ivsafe_seed44`. Those panels read `pred_uni2_<name>/`, not `pred_uni2_axisfix_`, and use that seed's decision rule and OOF prior. The ten-model mean is not shown. Seed 42 is already written; seed 44 is not, until `RUNS` is set to the seed-44 result directory and the script is run. Details are in the External_val README, section "2026-11-17 Released model = fold (and seed) ensemble".

```bash
CUDA_DEVICE=2 bash code/External_val/predict_fold_ensemble.sh NPC
```

[`Pred_visual_npc_uni2_ens5_ivsafe_seed42.ipynb`](Pred_visual_npc_uni2_ens5_ivsafe_seed42.ipynb) repeats the figure set of `Pred_visual_npc_uni2_A2.ipynb` on patient1, using the five-fold mean in `pred_uni2_ens5_l12_links_1_ivsafe_seed42/`. It does not re-predict.

[`Pred_visual_npc_patient1_axisfix.ipynb`](Pred_visual_npc_patient1_axisfix.ipynb) scores B cell (`B_cell` + `Plasma`) inside the expert TLS outline against the rest of the slide, then draws the TLS crops on one color scale. Adding `l12_links_2` and `l12_balanced` does not put B cells in that outline: argmax B inside the 5,262 nuclei is 0 for both, and the foci are still DC and DCIS. cell2location on the same 26 spots calls B. UNI2 tokens stay in `project_all_UNI2`.

2026.11.17. The notebook also scores B inside the TLS under four label modes (`posthoc.label_modes`: argmax, `decision_rule_l2_only.json`, EM prior shift, kNN smoothing). `STAR_PROB_MIN` other than 0 now writes `pred_uni2_axisfix_<arm>_probX.XX/`; set `PRED_TAG` in the notebook to match.

## 2026-11-15 Per-type abundance vs cell2location

Yes. A per-type abundance map is the right way to put Hist2Pheno next to cell2location on these two slides. The nuclei / pseudo-spot / cell2location cell in `Pred_visual_npc_uni2_A2.ipynb` draws one winner per spot: Hist2Pheno `pred_CT`, the majority L2 inside the Visium disc, and `argmax` of the cell2location counts. A type appears only where it is the largest. On patient1 the seven cell2location counts in `Annotation/sp.obs.csv` sum to about 7–55 per spot (median about 21). Tumor’s median is about 4.2 and B’s is about 2.2, so B is often present and still loses the argmax. That is the scale problem: one shared winner hides every type whose mass is smaller than the winner.

Hist2Cell does the same kind of display. Zhao, Liang, Huang, Huang, and Yu plot predicted cell-type abundance from H&E, one map per type, rather than a single label map (Hist2Cell: Deciphering fine-grained cellular architectures from histology images. *Cell Genomics* 6, 101137, 2026. [doi:10.1016/j.xgen.2025.101137](https://doi.org/10.1016/j.xgen.2025.101137)).

What Hist2Pheno can actually put on that map is the mean L2 softmax of the StarDist nuclei that fall inside the Visium spot (`nuclei_to_spots(..., values=probs)`), then summed with `L2_TO_C2L` into the seven shared types (`tumor`, `normal`, `fibroblast`, `T`, `Treg`, `B`, `Myeloid`). `B` is `B_cell` + `Plasma`. `T` is `CD4_T` + `CD8_T` + `NK`. Endothelial and Neural stay out, so a spot’s seven numbers need not sum to 1. The hard `fraction_*` columns written by the previous cell are argmax counts; they are not this abundance.

The following cell of `Pred_visual_npc_uni2_A2.ipynb` draws that mean-softmax row against the cell2location counts. Each panel is colored from 0 to its own 99th percentile (gray = no nuclei, or no cell2location row). A large type and a small type are both readable. The number on each column is Spearman on the spots that have both values. `npc_figures.plot_abundance_maps` builds the figure; the table is `{sample}_abundance_spearman.csv`.

Read the colors as spatial patterns. Hist2Pheno is a probability in [0, 1]. cell2location is an expected cell count. Rank correlation is unchanged by scaling one whole map, and it does change if each spot is divided by its own total. The Spearman numbers in the next paragraph come from that older cell: softmax against raw counts, a 65 µm disc, and `DCIS` mapped to `normal`. These slides have no nucleus ground truth, so the figure is a pattern check and stops short of a ROC.

2026-11-16. `npc_figures.L2_TO_C2L["DCIS"]` is `tumor`. DCIS is neoplastic, so a DCIS call now raises `tumor`. The replacement comparison is [`Pred_visual_npc_uni2_A8_validation.ipynb`](Pred_visual_npc_uni2_A8_validation.ipynb) and [`visium_validation.py`](../../Hist2Pheno_pkg/visium_validation.py). Proportion is compared with proportion (`proportions` of the cell2location counts) and count with count (`count_*` = sum of the grouped softmax inside the spot). The capture radius is `spot_diameter_fullres × 55/65 / 2`: Space Ranger's diameter is a 65 µm spot, and the capture area is 55 µm. Slide-level cell fractions sum the abundances; they are not the fraction of spots where a type wins the argmax. When `filtered_feature_bc_matrix.h5` is present, `marker_scores` adds an RNA reference (MS4A1/CD79A for B, MZB1/JCHAIN for plasma, CD3D/CD3E for T) that does not use cell2location.

On `20210809-C-AH4199551` every one of the 1,331 in-tissue spots has both a captured nucleus and a cell2location row. Spearman is tumor 0.25, normal 0.07, fibroblast −0.32, T 0.04, Treg −0.24, B −0.06, Myeloid −0.22. The maps can be compared type by type. On this slide they do not line up. Re-running the cell writes the same numbers to `{sample}_abundance_spearman.csv` for both patients.

## 2026-11-11 Per-nucleus confidence (`prediction_confidence.py`)

Code lives in [`code/Hist2Pheno_pkg/prediction_confidence.py`](../../Hist2Pheno_pkg/prediction_confidence.py). Tests: [`test_prediction_confidence.py`](../../Hist2Pheno_pkg/test_prediction_confidence.py). Package overview: [`Hist2Pheno_pkg/README.md`](../../Hist2Pheno_pkg/README.md#2026-11-11-per-nucleus-cell-type-confidence). Method: Jin et al., bioRxiv 2026.03.01.708850 ([UTOPIA](https://github.com/kaitianjin/UTOPIA)).

NPC has B cells on cell2location maps, but Hist2Pheno **argmax** `pred_CT` can look empty for `B_cell`. The L2 softmax is a cell2location-style vector; `argmax` hides any type whose peak is below Tumor / DC. B vs Plasma vs T are also close at 7–8 µm. These slides have **no nucleus GT** — calibrate on pan-cancer OOF (`oof_predictions.npz`), not on NPC.

The module does not retrain the MLP. It scores `{sample}_l2_probs.npz`:

| Piece | What it is |
|-------|------------|
| `annotate_nuclei` | `conf_max` / `conf_margin` / `conf_entropy`; `label_plot` is sharp L2, a family (`B_Plasma`, `T_cell`), or `low_confidence` |
| `conformal_from_oof` | per-class conformal **p** vs OOF cells that are not class `c` |
| `confidence` | UTOPIA log-stretch of the **raw** p (p=0.01 → 1, p=0.5 → 0) |
| `present` | FPR gate: `P(c)` ≥ OOF-null `(1-α)` quantile (`--fdr 0.10` = 90th percentile) |
| `p_adj` | whole-slide BH; diagnostic only. Empty on ~1e5 NPC nuclei (UTOPIA BH is per histology cluster) |
| `gated = present × confidence` | display: zero out nuclei that fail the FPR gate. **Not** a second independent score; both come from the same `P(c)` ranking |

Do not `argmax` gated scores across classes to rebuild L1/L12/L2. Confidence is calibrated **per class**; Tumor vs DCIS are not on one scale (A2 gated-argmax inflated DCIS). Use softmax argmax for the all-type map; use `present × conf` as a **single-class** overlay.

### CLI (sidecar, no GPU, does not rewrite pred h5ad)

Match `--oof` to the same UNI2 run as the pred tree. `conda activate SeededNTM`, repo root.

```bash
SAMPLE=20210809-C-AH4199551
PRED=data/Visium/NPC/pred_uni2_A2_bestmlp_paired_v2_seed42
OOF=data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/oof_predictions.npz

/nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python \
  code/Hist2Pheno_pkg/prediction_confidence.py \
  "${PRED}/${SAMPLE}/${SAMPLE}_l2_probs.npz" \
  --oof "${OOF}" \
  --classes B_cell Plasma B_Plasma \
  --fdr 0.10
```

`--classes` also accepts family names `B_Plasma`, `T_cell`, `lymphoid`, `myeloid`, `epithelial`. A3 / A5 / A6: swap `pred_uni2_A{3,5,6}_bestmlp_paired_v2_seed42` and the matching `result_uni2_A{3,5,6}_*_paired_v2_seed42/oof_predictions.npz`.

### A2 notebook (`Pred_visual_npc_uni2_A2.ipynb`)

Plot helpers are in the package, not copied in the notebook:

| Function | Figure |
|----------|--------|
| `plot_bcell_presence_figure` | 2×4: argmax B/Plasma, FPR presence, confidence, `present×conf` (both slides) |
| `plot_class_argmax_vs_gated` | 2×2: argmax vs `present×conf` for one class (patient1: `Tregs`, `Tumor`) |
| `plot_trusted_spatial` | 2×3 gated-argmax L1/L12/L2 (contrast only; do not replace `plot_spatial`) |

Setup imports those three plus `OOF_NPZ`. Restart kernel, run setup, then the three plot cells. Sidecars are created on first run if missing (`ensure_conformal_npz` / `ensure_trusted_labels`).

PDF/PNG export, the cell2location grouping, and the expert-TLS outline are in [`npc_figures.py`](npc_figures.py) (this folder), not in the notebook. Setup imports them and calls `configure_figure_dpi(save_dpi=SAVE_DPI, plot_dpi=DPI)`.

| Function | Used for |
|----------|----------|
| `save_figure` | PDF (editable text) and PNG |
| `map_l2`, `group_fraction`, `legend_handles`, `draw_spot_circles` | nuclei / pseudo-spot / cell2location |
| `l2_group_means`, `load_aligned_l2_probs`, `plot_abundance_maps` | per-type Hist2Pheno softmax vs cell2location counts |
| `load_tls_xy`, `keep_leftmost_foci`, `tls_smooth_contours`, `draw_tls`, `nuclei_in_contours` | expert TLS outline |
| `tls_score_row` | abundance TLS vs spatial TLS |

| Sidecar (next to pred h5ad) | Role |
|-----------------------------|------|
| `{sample}_l2_probs.npz` | L2 softmax from `cmd_predict` |
| `{sample}_confidence.npz` | B_cell / Plasma / B_Plasma `present` + `confidence` |
| `{sample}_confidence_treg_tumor.npz` | Tregs / Tumor (patient1) |
| `{sample}_confidence_hierarchy.npz` | L1 / L12 / L2 gated-argmax labels |

Figures under `data/Visium/NPC/pred_uni2_A2_bestmlp_paired_v2_seed42/figures/`:

| File | Content |
|------|---------|
| `{sample}_spatial_B_confidence.png` | B 2×4 |
| `20210809-C-AH4199551_spatial_Tregs_confidence.png` | Treg 2×2, patient1 only |
| `20210809-C-AH4199551_spatial_Tumor_confidence.png` | Tumor 2×2, patient1 only |
| `{sample}_spatial_L1_L12_L2_trusted.png` | gated-argmax all-type map |

A2 on `20210809-C-AH4199551` (n=142,184):

| Call | argmax | FPR≤0.10 `present×conf` |
|------|--------|-------------------------|
| `B_cell` | 225 | 5,484 (3.86%) |
| `Tregs` | 40 | 207 |
| `Tumor` | 64,294 | 83,253 |

Whole-slide BH FDR≤0.10 is 0 for all three. That is expected. This is output-side quantification, not a cell2location abundance correction. Training-side balance (A7 sampler + focal) is a separate lever if the goal is to move softmax mass onto B.
