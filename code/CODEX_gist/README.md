# CODEX GIST (s1167 GIST TMA) notes

本文件记录 `code/CODEX_gist`：在 **550 个全部有 cell-type annotation 的 GIST TMA cores** 上跑 Hist2Pheno（对照 `code/CODEX_pdac` 的 278 annotated PDAC 流程）。

运行环境：`conda activate SeededNTM`。

## 与 PDAC 的差别

| | PDAC (`CODEX_pdac`) | GIST (`CODEX_gist`) |
|--|--|--|
| Cores | Pancreas TMA 473 | GIST TMA **550** |
| Annotated | **278** | **550 / 550** |
| Incomplete_Cases | 195（无 cell-type CSV） | **无** |
| Coverslips | c001, c003, c005, c007 | **c009, c011, c013** |
| StarDist | `StarDist_Segment_pdac/pdac_result/` | `StarDist_Segment_gist/gist_result/` |
| Demo ACQ | `Charvill-94_c001_v001_r001_reg001` | `Charvill-94_c013_v001_r001_reg002` |
| `pan_organ` | `codex_pdac` | `codex_gist` |

数据仍在 `data/CODEX/HCC/Michael_data_transfer/s1167/{ACQUISITION_ID}/`（与 PDAC 同一 s1167 根目录，按 coverslip 分开）。Metadata 仍读 `raw_metadata_updated.xlsx` sheet `Clinical_info`，`cohort="GIST TMA"`。

HE 布局与 PDAC 相同：每个 acquisition 文件夹内 `he_img/`，没有 HCC 的 `HE/{MATCHED_HE}/`。

Cell-type CSV：GIST 使用 `{acq}.{numeric_id}.cell_types.csv`（见 `CODEX_pdac/s1167_img_cell_mapping.py` 的 `cell_types_path`）。

## Code

| 文件 | 作用 |
|------|------|
| `match_codex_cells_with_pixel.py` | 550 annotated → `*_cells_with_pixel.csv` / StarDist-matched CSV |
| `demo_GT_feature_extraction_Single.sh` | 单核 UNI（GT 或 StarDist 坐标） |
| `demo_UNI_feature_extraction_batch.sh` | 550 annotated 批量 UNI |
| `transer_embedding_label_h5ad.py` | matched / all-nuclei h5ad |
| `GIST_train_validate_cv_UNIlabel.py` | per-sample + cross-dataset 训练 / StarDist 推断 |
| `GIST_train_validate_cv_UNIlabel_all.ipynb` | 跨核 CV notebook（§1–§5） |
| `s1167_plot.py` | GIST StarDist-all 空间图（overview 默认每 coverslip 2 个 core） |
| `demo.sh` | 全流程命令索引 |

HE / metadata / cell CSV 读写复用 `code/CODEX_pdac/s1167_img_cell_mapping.py`（不要复制第二份 mapping）。

可视化预处理（TIFF / cell-type JPG）仍用 PDAC 的 `Data_process_visual_codex_HEcelltype_pdac.ipynb` 里 GIST 各节。

## Changelog

### 2026-08-20 — GIST Hist2Pheno 训练轨（550 annotated）

- 从 `code/CODEX_pdac` 分出 GIST 专用 CLI / notebook / UNI shells。
- 全部 550 cores 有 annotation，**没有** Incomplete_Cases §6。
- StarDist 根目录：`data/CODEX/HCC/StarDist_Segment_gist/gist_result/`。
- `plotting_palettes.py` 增加 `pan_organ="codex_gist"`。
