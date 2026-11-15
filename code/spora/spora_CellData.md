# SPORA CODEX：哪些数据能进 Hist2Pheno 训练集

2026-11-16。数据目录：`data/CODEX/spora`。筛选笔记本是 `spora_celltype_training_screen.ipynb`，打印为中文。本节只读原始图像和 parquet；训练导出写到 `data/CODEX/spora/prepared/`。

Hist2Pheno 现有监督样本来自 CODEX HCC、PDAC、GIST、GBM、ESCC 和 Xenium 肺。一条训练样本要同时有三样东西：配对 H&E、细胞在 H&E 上的质心、能落到 `celltype_pancancer` 且属于可训练 L2 的标签。`Unknown`、杂类、杂交类、以及 H&E 上和现有类别形态不一致的标签不进训练集。下面五个数据集按这个标准筛过。

## 结论

| 数据集 | 疾病 | 细胞类型表 | 建议 |
| --- | --- | --- | --- |
| schurch2020coordinated | 结直肠癌 TMA（Schürch 2020, *Cell*） | 有，257,702 个细胞 / 140 张组织 | **优先纳入** |
| hoefflin2026spatial | IDH 突变胶质瘤（Hoefflin 2026） | 有，2,448,212 个细胞 / 30 张组织 | **纳入**，恶性状态先并成 Tumor |
| phillips2021immune | 皮肤 T 细胞淋巴瘤 MF/SS（Phillips 2021） | 有，117,170 个细胞 / 69 张组织 | **先不进主训练集** |
| haist2026lymph | 头颈鳞癌淋巴结转移（Haist 2026, *Cancer Cell*） | 没有 | 现在不能做监督训练 |
| hickey2023organization | 人小肠图谱，非肿瘤（Hickey 2023） | 没有 | 不能纳入 |

优先做 **schurch2020coordinated**。它是实体瘤，140 张有标注的组织都有配准 H&E，细胞编号和 1 µm 实例分割完全一致，保留下来的 205,264 个细胞里 Tumor、巨噬、基质、内皮、CD4、CD8、B、浆细胞都有一定数量。结直肠癌目前不在已训练队列里，标签空间里已经有 CRC 的 Tumor I–V 等别名，补上这批原生名即可。

**hoefflin2026spatial** 可以作为现有 GBM 旁边的 IDH 突变胶质瘤。30 张组织全部配准，2,448,212 个 `cell_id` 与 1 µm mask 完全一致。可映射细胞 1,633,273 个（66.7%），其中 Tumor 1,301,883 个，大约八成。B 细胞只有 72 个，CD8 只有 4,664 个，稀有免疫类不要单独当可靠类别。

## 纳入时用的门槛

1. `metadata/cells.parquet` 里有细胞类型。没有这张表就当成标注缺失，不能当成 0 个细胞。
2. 同一 `tissue_id` 有 H&E，而且 `alignment` 为 `exact`。
3. 能得到 H&E 像素上的质心。hoefflin 和 schurch 的细胞表没有 X/Y，质心从 `segmentations/1_0mpp/cell_masks/instances/{tissue_id}.npz` 的 `mask` 来，编号就是 `cell_id`。
4. 原生标签映射到现有泛癌 L2，并且属于现在训练会保留的类别。这些名字多数还不在 `NATIVE_TO_V1` 里，不能直接调用 `native_to_pancancer`，要先加别名。
5. 划分按 `patient_id`。官方 `split` 只在下面写明的有标注组织上按病人分开。

图和表：

- `result/figures/spora_trainable_L2_counts.png`
- `result/spora_training_gate.csv`
- `result/spora_label_to_pancancer_L2.csv`
- `result/spora_trainable_L2_counts.csv`

图里的数量是“标签能映射到泛癌 L2”的细胞，phillips 也画进去了，便于比较。phillips 仍然不进主训练集，原因见下文。

## schurch2020coordinated：优先纳入

结直肠癌 TMA。有细胞标注的 H&E 是 140 张，35 个病人。`original_patient_identifier` 全部以 `CRC_TMA_A` 开头。`Full_Histology` 为腺癌 112 张、黏液腺癌 28 张。另外还有 110 张 H&E 没有细胞表，不能做监督。

`diagnosis`、`tissue`、`normal / neoplasia` 这几列和 CRC TMA 对不上：同一批 `CRC_TMA_A_*` 病人被填成骨髓、淋巴瘤、黑色素瘤、正常肝等。训练时的癌种用 `CRC_TMA` 和 `Full_Histology`。不要用 `diagnosis`。

坐标：140/140 张组织的 `cell_id` 与 1 µm 实例 mask 完全一致，257,702/257,702 个细胞都在 mask 里。质心取 mask，分辨率是 1 µm/像素。

标签用细的 `cell_type`，不用更粗的 `cell_type_atlas`。atlas 会把平滑肌、脂肪并进基质，也会把浆细胞并进 B。可保留 205,264 / 257,702（79.7%）：

| 泛癌 L2 | 细胞数 | 主要原生名 |
| --- | ---: | --- |
| Tumor | 46,917 | tumor cells |
| Macrophage | 44,004 | CD68+CD163+ macrophages 等，含 CD11b+ monocytes |
| Neutrophil | 22,015 | granulocytes |
| Stromal | 20,023 | stroma |
| CD4_T | 18,889 | CD4+ T cells，含 CD45RO+、GATA3+ |
| CD8_T | 16,586 | CD8+ T cells |
| B_cell | 12,890 | B cells |
| Endothelial | 11,979 | vasculature、lymphatics |
| Plasma | 8,456 | plasma cells |
| Tregs | 2,786 | Tregs |
| DC | 396 | CD11c+ DCs |
| NK | 323 | NK cells |

丢弃 52,438 个：平滑肌 27,680（现有规则把平滑肌排除在 Stromal 之外）、dirt、undefined、Unknown、未分型 immune cells、免疫/血管混合、肿瘤/免疫混合、脂肪、神经、未分型 CD3+ T cells。

`granulocytes` 按 Neutrophil 保留。原文没有把肥大细胞拆出去，这一条的置信度低于 CD4/CD8/肿瘤。CD45RO+ 和 GATA3+ 的 CD4 并进 CD4_T，因为现有 L2 不按激活状态再拆，H&E 也分不开。

有标注的 140 张上，保留细胞的病人是 train 27 / test 8，没有病人同时出现在两边。全部 250 张 H&E（含没有细胞表的 110 张）里，有 14 个病人的不同视野被分到了 train 和 test。以后如果把无标注图像加进来，按 `patient_id` 重划，不要沿用那 110 张上的 `split`。

## hoefflin2026spatial：纳入

IDH 突变胶质瘤，25 个病人、30 张组织，H&E 与 CODEX 全部 `exact`。1 µm 和 0.5 µm 的 H&E、CODEX、实例分割都在。30/30 张的 `cell_id` 与 1 µm mask 完全一致。细胞表没有坐标列，质心用 mask。

`cell_type_atlas` 已把 Oligo、Astro、Neuron、low、excluded 收成 Unknown。这里跟随原表，不把 Oligo 另提成 Neural。可保留 1,633,273 / 2,448,212（66.7%）：

| 泛癌 L2 | 细胞数 | 原生名 |
| --- | ---: | --- |
| Tumor | 1,301,883 | cAC1、cAC2、cOPC、cUndiff、cMES、cMESHyp |
| Macrophage | 217,526 | Mac（含小胶质，按现有队列把髓系并进巨噬） |
| Endothelial | 100,687 | Vasc |
| CD4_T | 8,131 | TcellCD4 |
| CD8_T | 4,664 | TcellCD8 |
| Neutrophil | 310 | Neutro |
| B_cell | 72 | Bcell |

丢弃 814,939 个，与 atlas 的 Unknown 数量一致：low 308,700、excluded 299,018、Oligo 103,796、Neuron 57,523、Astro 45,902。

恶性状态并成一个 Tumor。cAC / cOPC / cMES 在 H&E 的 16 像素块上分不开，也和现有 GBM 状态标签最后并到 Tumor 的做法一致。保留细胞的病人是 train 20 / test 5，没有重叠。

这批对 Tumor、巨噬、内皮有用。CD4/CD8 数量少，B 和中性粒可以留下，但不要指望它们撑起一个类别。

## phillips2021immune：先不进主训练集

蕈样肉芽肿（MF）和 Sézary 综合征（SS），14 个病人、70 张组织。69 张有细胞表且配准为 exact；`phillips2021immune_bsrmsvtc_0003` 的 alignment 是空的，也没有细胞，排除。

细胞表自带 X、Y。图像高宽恒为 1440×1920，`he_mpp = 0.37744`，所以 X/Y 在 0.37744 µm/像素上。换到 1 µm/像素时乘以 0.37744。1 µm mask 盖住 107,331/117,170（91.6%）。mask 里没有的细胞仍可以用 X/Y。

互相排斥的标签是 `old_cluster`。名字能映射的有 75,946 个（巨噬、表皮上皮、基质、Treg、内皮、CD8、B、树突、CD4、浆细胞、中性粒）。另外 39,815 个 `tumor cells` 和 `tumor cells, intraepithelial` 是皮肤淋巴瘤的恶性淋巴细胞。泛癌 Tumor 是 H&E 上的上皮性/实质肿瘤，这两类不能映射成 Tumor。肥大细胞、神经、CLA+ 白细胞也丢弃。

皮肤淋巴瘤和现有实体瘤的域差很大。免疫和上皮标签可以以后单独做皮肤队列。在那之前不放进泛癌主训练集。

## haist2026lymph：现在不能监督

头颈鳞癌淋巴结，79 个病人、389 对 H&E/CODEX，配准全部 exact，1 µm H&E 389 张都在。没有 `metadata/cells.parquet`。`segmentations/1_0mpp/cell_masks/instances` 不存在；同级的 `instances_virtues` 有 389 个 npz，只有实例编号，没有细胞类型。有分割也不能训练。若原文细胞类型以后补进 SPORA，这批配对最完整，值得再看。

## hickey2023organization：不能纳入

人小肠，不是现有癌种。没有细胞类型表。CODEX 64 张，H&E 只有 19 张，而且这 19 张才是 exact；另外 45 张 CODEX 是 no alignment。有实例分割，没有类型标签。

## 训练

这些别名已经写进 `code/Hist2Pheno_pkg/dataset/build_pancancer_celltype_v1.py`。schurch2020coordinated 的切片算作 **CRC**，hoefflin2026spatial 算作 **GBM**，队列名仍然是原来的七个。`1_demo_train.sh uni2 A8` 和下面的 train 都会在读 `section_index.csv` 时把缺的切片补进去。已经训完的 A0–A7 权重对应补进去之前的 896 张切片。

GT 质心来自 1 µm 实例分割。训练用这些质心上的 UNI2 token。当前索引用 1,066 张切片：CRC 143（原来 3 张 Xenium + 140 张 Schürch），GBM 32（原来 2 张 + 30 张 Hoefflin）。

先把 Schürch 里漏掉的 `CD8+ T cells` 和 `plasma cells` 写回细胞表，并在 GPU 0–3 上重提 UNI2。普通 `prepare` 和 `uni2` 会跳过已有文件，所以用这一条。Hoefflin 不重写。

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
bash code/spora/run_spora_pancancer.sh refresh-schurch
```

然后再训练。A8–A11 各用四张 GPU：fold 0 先建空间缓存，fold 1–3 并行，fold 4 接在 fold 0 后面，最后一张卡写 `best_mlp_gpu.pt`。一次只跑一个。A10 会先写 `data/PanCancer/mpp_map.csv`，Schürch 和 Hoefflin 是 1.0 µm/px。

```bash
bash code/PanCancer/run_a8_4gpu.sh
bash code/PanCancer/run_a9_4gpu.sh
bash code/PanCancer/run_a10_4gpu.sh
bash code/PanCancer/run_a11_4gpu.sh
```

`run_spora_pancancer.sh train` 仍是单独的 A2，结果在 `data/PanCancer/result_uni2_A2_spora9_paired_v2_seed42/`。A8–A11 分别写到 `result_uni2_A8_l12_balanced_paired_v2_seed42/`、`result_uni2_A9_nbr_comp_paired_v2_seed42/`、`result_uni2_A10_nbr_comp_geom_paired_v2_seed42/`、`result_uni2_A11_nbr_tissue_paired_v2_seed42/`。说明在 `code/PanCancer/README.md`。

只想先看 CRC 和 GBM（含这两批新切片）能不能跑通：

```bash
CUDA_DEVICE=0 bash code/spora/run_spora_pancancer.sh smoke
```

## 加进 `NATIVE_TO_V1` 的名字

下面是已经保留的原生名。丢弃类没有加进可训练别名。

schurch：`tumor cells`→Tumor；`CD4+ T cells`、`CD4+ T cells CD45RO+`、`CD4+ T cells GATA3+`→CD4_T；`CD8+ T cells`→CD8_T；`Tregs`→Tregs；`NK cells`→NK；`B cells`→B_cell；`plasma cells`→Plasma；`CD11c+ DCs`→DC；各型 macrophage 与 `CD11b+ monocytes`→Macrophage；`granulocytes`→Neutrophil；`stroma`→Stromal；`vasculature`、`lymphatics`→Endothelial。

`CD8+ T cells` 和 `plasma cells` 原先不在 `NATIVE_TO_V1` 里，`prepare_spora_sections.py` 把它们滤掉了。现在的 140 份 `*_cells_with_pixel.csv` 里这两类都是 0。映射已经补上。要让它们进入 CRC，运行 `bash code/spora/run_spora_pancancer.sh refresh-schurch`。它会强制重导 140 张 Schürch，删掉旧的 UNI2，再用 GPU 0–3 重新提取。Hoefflin 不动。

hoefflin：`cAC1`、`cAC2`、`cOPC`、`cUndiff`、`cMES`、`cMESHyp`→Tumor；`Mac`→Macrophage；`Neutro`→Neutrophil；`Vasc`→Endothelial；`TcellCD4`→CD4_T；`TcellCD8`→CD8_T；`Bcell`→B_cell。

phillips 若以后做皮肤队列，再用 `old_cluster`，并且继续把 `tumor cells` 排除在泛癌 Tumor 之外。
