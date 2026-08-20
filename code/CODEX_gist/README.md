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

## GPU：指定物理卡（例如 cuda:2）

`demo.sh` 是命令索引，不是可执行脚本。真正吃 GPU 的是 **UNI 特征提取** 和 **训练 / 推断**。  
代码一律把选中的物理卡藏成逻辑 `cuda:0`（`Image_feature_extraction.py` 和 `GIST_train_validate_cv_UNIlabel.py` 都用可见的 `cuda:0`），不要在 Python 里写 `torch.device("cuda:2")`。

| 你想用的物理卡 | 终端 | CLI | Notebook |
|--|--|--|--|
| GPU 2 | `export CUDA_VISIBLE_DEVICES=2` | `--cuda-device 2` | `NCRT_CUDA_DEVICE = "2"` |

日志里会看到 `device: cuda:0`，这是正常的：进程只看得到一张卡。确认：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

应打出 **编号 2** 那张卡的名字。

### 推荐：整段会话绑到 GPU 2

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
export CUDA_VISIBLE_DEVICES=2
```

之后 `demo.sh` 里的命令可以原样跑。`conda run` 会继承这个环境变量。

### 分步（不 export、单条命令指定）

**1. `match_codex_cells_with_pixel.py` / 3. `transer_embedding_label_h5ad.py`**  
CPU 为主，不用指定 GPU。

**2. UNI 特征提取**

```bash
CUDA_VISIBLE_DEVICES=2 bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh gt
CUDA_VISIBLE_DEVICES=2 bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh stardist
```

**4. 训练 CLI**（`--cuda-device` 只在 **还没设置** `CUDA_VISIBLE_DEVICES` 时生效）

```bash
conda run --no-capture-output -n SeededNTM python -u \
  code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
  --cuda-device 2 \
  --sample Charvill-94_c013_v001_r001_reg002

conda run --no-capture-output -n SeededNTM python -u \
  code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
  --cuda-device 2 \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial
```

已经 `export CUDA_VISIBLE_DEVICES=2` 时，不必再加 `--cuda-device 2`。  
**不要**同时设 `CUDA_VISIBLE_DEVICES=0` 和 `--cuda-device 2`：环境变量优先，会仍跑在 GPU 0 上。

**5. Notebook** `GIST_train_validate_cv_UNIlabel_all.ipynb`  
第一个 runtime cell 改成下面这样，然后 **Restart Kernel** 再从上到下跑（必须在 `import torch` 之前设好）：

```python
os.environ.setdefault("NCRT_CUDA_DEVICE", "2")
```

`configure_notebook_runtime()` 会在 `CUDA_VISIBLE_DEVICES` 未设置时把它设为 `2`。若 kernel 启动时已经带了别的 `CUDA_VISIBLE_DEVICES`，需要先清掉或重启。

## Changelog

### 2026-08-20 — 指定 GPU（cuda:2）

- README 补充 `CUDA_VISIBLE_DEVICES` / `--cuda-device` / `NCRT_CUDA_DEVICE` 的用法。

### 2026-08-20 — GIST Hist2Pheno 训练轨（550 annotated）

- 从 `code/CODEX_pdac` 分出 GIST 专用 CLI / notebook / UNI shells。
- 全部 550 cores 有 annotation，**没有** Incomplete_Cases §6。
- StarDist 根目录：`data/CODEX/HCC/StarDist_Segment_gist/gist_result/`。
- `plotting_palettes.py` 增加 `pan_organ="codex_gist"`。
