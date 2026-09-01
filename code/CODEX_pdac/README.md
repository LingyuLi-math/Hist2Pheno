# CODEX PDAC (s1167 Pancreas TMA) notes

本文件记录 `code/CODEX_pdac` 的数据构成与**每一步**处理。
后续每一步（StarDist、匹配、训练等）都写进「Notebook 处理流程」并在 changelog 顶部追加。

运行环境：`conda activate SeededNTM`。

## Metadata: use `raw_metadata_updated.xlsx`

```text
data/CODEX/HCC/Michael_data_transfer/s1167/raw_metadata_updated.xlsx
sheets: Clinical_info (cores), Celltype (hierarchy), s1167_metadata (旧表，不要用)
```

**不要用** `raw_metadata.csv`，也不要用同文件里的 `s1167_metadata` sheet（`tissue_type` 仍有 490 个空，和旧 CSV 一样）。
现行队列以 **`Clinical_info`** 为准：Pancreas TMA **473**、GIST TMA **550**。

| coverslip | CSV（旧） | xlsx `tissue_type`（现行） |
|-----------|-----------|----------------------------|
| c001 | NA（148） | **Pancreas TMA** 148 |
| c003 | NA（149） | **Pancreas TMA** 149（此前一版曾把其中 37 标成 GIST、2 个空；现已全部改为 Pancreas） |
| c005, c007 | Pancreas TMA 176 | 不变（95 + 81） |
| c009, c011 | GIST TMA 357 | 不变（160 + 197） |
| c013 | NA（193） | **GIST TMA** 193 |

xlsx 合计：Pancreas TMA **473**、GIST TMA **550**。1023 行均有 `tissue_type` 和 `SAMPLE_LABEL`。

`cohort` 跟 Excel `tissue_type`。c001–c007 现为 Pancreas，c009–c013 为 GIST。
`load_s1167_metadata(cohort="PDAC")` 与 `cohort="Pancreas TMA"` 等价。

## PDAC subset

| coverslip | n | SAMPLE_LABEL | cell-type CSV |
|-----------|---|--------------|----------------|
| c001 | 148 | TA-237 | 148 / 148（`{acq}.cell_types.csv`） |
| c003 | 149 | TA-256 | 130 / 149 |
| c005 | 95 | TA-257 | 0 |
| c007 | 81 | TA-258 | 0 |
| **All PDAC** | **473** | | **278 / 473** |

c005 / c007 仍可导出 HE TIFF，但不能画 cell-type JPG。

HE 仍在每个 acquisition 文件夹内，与 HCC s4769 的 `MATCHED_HE` 布局不同。TIFF 是 TMA core 分辨率（c001 约 `3632×3632`），1:1 导出，不要 upsample。

## Code

| 文件 | 作用 |
|------|------|
| `s1167_img_cell_mapping.py` | 读 **xlsx**、HE Zarr、细胞 CSV；导出 TIFF / JPG |
| `s1167_plot.py` | coverslip stacked / pooled 组成图；StarDist macro AUROC vs clinical |
| `s1167_histology_derived_niche_index.py` | PDAC/GIST 共用 TLS / SRI / TNI（`configure("codex_pdac"|"codex_gist")`） |
| `pdac_histology_derived_niche_index.py` | PDAC wrapper（`pan_organ="codex_pdac"`） |
| `Data_process_visual_codex_HEcelltype_pdac.ipynb` | PDAC 交互式流程（步骤 0–7） |
| `match_codex_cells_with_pixel.py` | 278 annotated → `*_cells_with_pixel.csv` / StarDist-matched CSV |
| `demo_GT_feature_extraction_Single.sh` | 单核 UNI（GT 或 StarDist 坐标） |
| `demo_UNI_feature_extraction_batch.sh` | 278 annotated 或 `INCOMPLETE=1` 的 195 Incomplete_Cases |
| `transer_embedding_label_h5ad.py` | matched / all-nuclei h5ad |
| `PDAC_train_validate_cv_UNIlabel.py` | per-sample + cross-dataset 训练 / StarDist 推断 |
| `PDAC_train_validate_cv_UNIlabel_single.ipynb` | **单核** train / HE validate / StarDist（demo：`Charvill-94_c001_v001_r001_reg001`） |
| `PDAC_train_validate_cv_UNIlabel_all.ipynb` | 跨核 CV notebook（§1–§6）；默认 **load** CLI 权重，不重训 |
| `PDAC_histology_derived_niche_index.ipynb` | TLS / SRI / TNI 下游（matched StarDist） |
| `Pred_statistic_visual_pdac_all.ipynb` | 278 核 pooled ROC + coverslip / SAMPLE_LABEL AUROC |
| `demo.sh` | 全流程命令索引 |

从 `code/CODEX_pdac` import，不要从 `CODEX_hnscc` import。

## 下游分析（对照 HCC）

先完成 cross-dataset 权重（CLI `demo.sh` 或 notebook `SKIP_POOLED_TRAIN=False`），输出在 `result_all_spatial_pdac/stardist/`。之后再开这两个 notebook。已有权重时用 `_all` notebook 的 `SKIP_POOLED_TRAIN=True` load 即可。环境：`SeededNTM`。

| Notebook | 作用 | 输出 |
|----------|------|------|
| `PDAC_histology_derived_niche_index.ipynb` | TLS / SRI / TNI（matched StarDist softmax + `spatial_HE`） | `s1167/result_all_spatial_pdac/niche_index_pdac/` |
| `Pred_statistic_visual_pdac_all.ipynb` | 278 核 pooled ROC；macro AUROC vs **coverslip** / **SAMPLE_LABEL** | `s1167/result_all_spatial_pdac/clinical_viz_pdac/` |

与 HCC 的差别：没有 **Response**。临床分组只有 coverslip 与 SAMPLE_LABEL。TNI 的髓系项是 **Macrophages**（没有 M2-like）。§6 默认 `QUICK_VALIDATE = True`（先跑 6 个 core）；全队列把该开关改成 `False`。

不要在同一个 kernel 里同时 `import pdac_histology_derived_niche_index` 和 `gist_histology_derived_niche_index`：共用模块的 `configure()` 以后导入为准。

## Notebook 处理流程

文件：`Data_process_visual_codex_HEcelltype_pdac.ipynb`。  
环境：`SeededNTM`。`chdir` 到 `data/CODEX/HCC/`，加载 `code/CODEX_pdac/s1167_img_cell_mapping.py`。

Demo：

```text
Charvill-94_c001_v001_r001_reg001
```

### 0. Setup

- matplotlib PDF 字体；`sys.path` 加入 `code/CODEX_pdac`。
- 确认打印的 metadata 是 `raw_metadata_updated.xlsx`。

### 1. Load metadata（xlsx）

- `meta_all_df = load_s1167_metadata(cohort=None)`：1023 行。
- `pdac_df`：`cohort="PDAC"` → 473；`pdac_ann_df` → 278。
- `gist_df`：`cohort="GIST TMA"` → 550；`gist_ann_df` → 550（全部有 cell-type CSV）。

### 2. Preview HE

- PDAC demo：`PRIMARY_ACQ_ID = Charvill-94_c001_v001_r001_reg001`
- GIST demo：`PRIMARY_GIST_ACQ_ID = Charvill-94_c013_v001_r001_reg002`

### 3. Load CODEX cells

- PDAC：`load_cells_by_acq(PRIMARY_ACQ_ID)` → `cells` / `cells_by_acq`
- GIST：`load_cells_by_acq(PRIMARY_GIST_ACQ_ID)` → `gist_cells` / `gist_cells_by_acq`

### 4. Export HE Zarr → TIFF

- `export_he_zarr_to_tif`：1:1，不降采样。`{acq}/figures/{ACQUISITION_ID}.tif`
- PDAC batch：`pdac_df`（473）；GIST batch：`gist_df`（550）

### 5. Copy TIFF → StarDist

```text
data/CODEX/HCC/StarDist_Segment_pdac/HE_images/{ACQUISITION_ID}.tif
data/CODEX/HCC/StarDist_Segment_gist/HE_images/{ACQUISITION_ID}.tif
```

`copy_he_tifs`，默认 copy 不 move。PDAC 与 GIST 分目录，不要混在一起。

### 6. Cell types on HE → JPG

- `{acq}/figures/{ACQUISITION_ID}_celltype_on_HE.jpg`
- batch：`export_codex_celltypes_to_jpg(pdac_ann_df)`（278）。

### 7. Visualization（notebook 中 `## Visualization` 之下）

CODEX↔HE alignment analog + hierarchy 已挪到这一节（HE TIFF/JPG 导出之后）：

- `load_codex_he_alignment(cohort=..., annotated_only=False)` ≈ HCC `ALIGNED=='Y'`（有 `he_img`）
  - PDAC `alignment_df`：473；GIST `alignment_gist_df`：550
- `annotated_only=True` ≈ HCC `Annotation=='Y'`（有 cell-type CSV）
  - PDAC `alignment_Anno_df`：278；GIST `alignment_gist_Anno_df`：550
- Hierarchy 来源：`raw_metadata_updated.xlsx` sheet `Celltype`
- notebook 显式保存副本：`s1167/s1167_celltype_hierarchy.xlsx`（PDAC+GIST，含 `Other`）
- `Other` 过滤掉后再校验；`PDAC_CELLTYPES` / `GIST_CELLTYPES` 须与 Level2 及磁盘标签一致
- 细胞组成图：`load_cells_from_mapping(alignment_df, skip_missing_annotation=False)` 读 **全部 473** PDAC cores（无 cell_types 的标为 `Unannotated`）；GIST 550；stacked 按 **coverslip** 分组

### 8. Match CODEX cells → HE pixels + StarDist

- `match_codex_cells_with_pixel.py`：默认 **278 annotated** cores。
- 输出在 `s1167/{ACQUISITION_ID}/`（没有 HCC 的 `HE/{MATCHED_HE}/` 层）：
  - `{acq}_cells_with_pixel.csv`
  - `{acq}_cells_matched_by_stardist.csv`
- StarDist CSV：`StarDist_Segment_pdac/pdac_result/{ACQUISITION_ID}/{acq}_Float_prob0.01_nms_0.3.csv`

### 9. UNI embeddings

- GT：`s1167/{acq}/project_all_UNI/ImgEmbeddings_all/`
- StarDist：`.../ImgEmbeddings_all_stardist/`
- 单核：`demo_GT_feature_extraction_Single.sh [gt|stardist]`
- 批量 278：`demo_UNI_feature_extraction_batch.sh gt|stardist`
- Incomplete_Cases 195（仅 StarDist）：`INCOMPLETE=1 bash .../demo_UNI_feature_extraction_batch.sh stardist`

### 10. h5ad transfer

- annotated：`--steps he_h5ad stardist_csv stardist_h5ad stardist_all_h5ad`
- Incomplete_Cases：`--incomplete --steps stardist_all_h5ad`（无 GT label）

### 11. Train / validate / StarDist inference

- CLI：`PDAC_train_validate_cv_UNIlabel.py`（`--mode per-sample` 或 `cross-dataset`）
- Notebook：`PDAC_train_validate_cv_UNIlabel_all.ipynb`
  - **默认 `SKIP_POOLED_TRAIN = True`**：不重训，只 load `demo.sh` 写出的 checkpoint
  - §1 prepare → §2 load `best_mlp_gpu.pt` → §3 打印已有 OOF 图（不重画）
  - §4 / §5 / §6 StarDist 重预测默认注释掉；已有 JPG/PDF 在 `stardist/` 与 `stardist_Incomplete_Cases/`
  - 只有要重新训练时才把 `SKIP_POOLED_TRAIN` 改成 `False`

CLI 不传 `--ablation-tag` 时目录是 **`D_emph_L2`**（即使开了 `--use-spatial-context`）。Notebook 必须用同一个 tag，否则会写到 `D_emph_L2_spatial_bs4096/` 并再训一遍。

```text
s1167/result_all_spatial_pdac/cross_dataset_cv/D_emph_L2/best_mlp_gpu.pt
s1167/result_all_spatial_pdac/roc_internal_level2_oof.pdf
s1167/result_all_spatial_pdac/stardist/{ACQUISITION_ID}/
```

CLI 训完后看结果：Restart Kernel，再跑 config → §1 → §2。

命令索引：`code/CODEX_pdac/demo.sh`。

## Main API

```python
load_s1167_metadata(cohort="PDAC", annotated_only=False)  # Clinical_info in raw_metadata_updated.xlsx
summarize_s1167_metadata(meta_df)
plot_he_overview(acq_id)
load_cells_by_acq(primary_acq_id, also_view=None)
export_he_zarr_to_tif(acq_id)
copy_he_tifs(tif_paths, dest_dir=DEFAULT_STARDIST_HE_DIR)
export_codex_celltypes_on_he_jpg(acq_id, show=True)
save_s1167_celltype_hierarchy()  # writes s1167/s1167_celltype_hierarchy.xlsx
load_cells_from_mapping(alignment_Anno_df)
plot_celltype_proportions_stacked(...)  # group by coverslip
```

## Changelog

### 2026-08-28 — `_all` notebook load 已有权重（不重训）

`PDAC_train_validate_cv_UNIlabel_all.ipynb` 默认 `SKIP_POOLED_TRAIN=True`，`POOLED_ABLATION_TAG=D_emph_L2`（与 `demo.sh` / CLI 默认一致）。§2 调 `_ensure_pooled_inference_ready` load `best_mlp_gpu.pt`；§3–§6 不自动重训或重预测。不要用 `D_emph_L2_spatial_bs4096`，那会另开一份 CV。

### 2026-08-28 — split pooled outputs from GIST

PDAC and GIST share `s1167/` but must not share `result_all_spatial/`. Cross-dataset weights now go to `result_all_spatial_pdac/`. Matched StarDist h5ad (`*_matched_features_stardist.h5ad`) is required for pooled StarDist validation; build it with `transer_embedding_label_h5ad.py --steps stardist_h5ad` (not only `stardist_all_h5ad`).

### 2026-08-20 — PDAC 下游：niche index + Pred statistic

- `PDAC_histology_derived_niche_index.ipynb` + `pdac_histology_derived_niche_index.py`：对照 HCC TLS / SRI / TNI；临床分组 coverslip / SAMPLE_LABEL。
- `Pred_statistic_visual_pdac_all.ipynb`：278 annotated pooled ROC + macro AUROC vs clinical（`s1167_plot.analyze_pdac_stardist_macro_auroc_by_clinical`）。

### 2026-08-20 — Hist2Pheno pipeline on PDAC (s1167)

Port of the CODEX HCC train/validate path onto Pancreas TMA:

- Sample key is **`ACQUISITION_ID`** (no `MATCHED_HE`).
- Training pool: **278** cores with cell-type CSV (c001 148 + c003 130).
- **Incomplete_Cases**: **195** PDAC cores without cell-type CSV
  (c003 19 + c005 95 + c007 81). Inference outputs go to
  `stardist_Incomplete_Cases/` (lung naming). Inputs stay under `s1167/{acq}/`.
- GIST is not in this training track.
- StarDist root: `data/CODEX/HCC/StarDist_Segment_pdac/pdac_result/`.
- New files: `match_codex_cells_with_pixel.py`, UNI extraction shells,
  `transer_embedding_label_h5ad.py`, `PDAC_train_validate_cv_UNIlabel.py`,
  `PDAC_train_validate_cv_UNIlabel_all.ipynb`, `demo.sh`.
- `plotting_palettes.py`: `pan_organ="codex_pdac"` (coverslip grouping).

### 2026-08-18 — switch to `raw_metadata_updated.xlsx`

Updated:

- 默认读 `raw_metadata_updated.xlsx` / **`Clinical_info`**（473 PDAC + 550 GIST）。
- Hierarchy 写在同一文件的 **`Celltype`** sheet（原先为空，已填入 L2→L1→L0）。
- 不要用 sheet `s1167_metadata`（旧 CSV 缺口）。`raw_metadata.xlsx` 已不在目录中。

### 2026-08-18 — PDAC/GIST hierarchy analog of HCC alignment cell

Updated:

- 新增 `s1167_celltype_hierarchy.xlsx` 与 `load_codex_he_alignment` / `load_s1167_celltype_hierarchy` / `load_s1167_celltypes`。
- notebook 增加与 HCC `ALIGNED` + `Annotation` + hierarchy 对应的一节；校验 Excel Level2 与磁盘标签一致。

### 2026-08-18 — notebook 并行 GIST TMA 流程

Updated:

- 与 Pancreas TMA 并列：读 `gist_df`、demo `PRIMARY_GIST_ACQ_ID`、导 TIFF、copy 到 `StarDist_Segment_gist/HE_images`。
- GIST 550 cores 全部有 cell-type CSV。

### 2026-08-18 — xlsx 再更新：c003 全部为 Pancreas TMA

Updated xlsx:

- 1023 行均有 `tissue_type` + `SAMPLE_LABEL`（无空行）。
- **Pancreas TMA 473**（c001 148 + c003 149 + c005 95 + c007 81），cell types **278**。
- **GIST TMA 550**（c009 160 + c011 197 + c013 193）。
- 上一版 xlsx 把 c003 的 37 个标成 GIST、2 个为空；现 c003 全部是 Pancreas（TA-256）。

Notebook `Data_process_visual_codex_HEcelltype_pdac.ipynb` 计数与 batch 注释已同步。

### 2026-08-18 — switch to `raw_metadata.xlsx` for PDAC

Updated:

- `s1167_img_cell_mapping.py` 只读 `raw_metadata.xlsx`（拒绝 `.csv`）。
- `cohort` 使用 Excel `tissue_type`，不再用 prefix 把 c001/c003/c013 标成 HNSCC。
- 默认队列：Pancreas TMA / PDAC。
- notebook 从 `code/CODEX_pdac` 加载；StarDist 目录改为 `StarDist_Segment_pdac/HE_images`。

Finding:

- CSV 空 `tissue_type` 的 cores 在 xlsx 中是 Pancreas / GIST，**不是 HNSCC**。
