# PDAC Incomplete_Cases — pan-cancer Hist2Pheno

195 Pancreas TMA cores have HE + StarDist but **no cell-type CSV** (c003 19 +
c005 95 + c007 81). UNI `_all_features_stardist.h5ad` files already exist under
`s1167/{ACQUISITION_ID}/`.

Outputs: `data/PanCancer/pred_unlabel/PDAC_unlabel/`

Default `predict` still uses the **5 largest** cores. The notebook shows **3
cores per coverslip** (c003 / c005 / c007) plus a mixed-class HE / Hist2Pheno
zoom. No TLS / SRI / TNI.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
bash code/External_val/PDAC_unlabel/demo.sh status
CUDA_DEVICE=0 bash code/External_val/PDAC_unlabel/demo.sh predict
```

`ALL=1` scores all 195 (long). Open `Pred_visual_pdac_unlabel.ipynb`.

## UNI2 paired_v2 best weights

A2 is fold 1. A3 is fold 0. `demo_uni2.sh` forwards to A3. A core is skipped when `project_all_UNI2/.../sc_pth_14_14` is missing.

```bash
CUDA_DEVICE=1 bash code/External_val/PDAC_unlabel/demo_uni2_A2.sh predict
CUDA_DEVICE=0 bash code/External_val/PDAC_unlabel/demo_uni2_A3.sh predict
```

Outputs: `data/PanCancer/pred_uni2_A2_bestmlp_paired_v2_seed42/unlabel/PDAC_unlabel/` and `pred_uni2_A3_bestmlp_paired_v2_seed42/unlabel/PDAC_unlabel/`. Notebooks: `Pred_visual_pdac_unlabel_uni2_A2.ipynb`, `Pred_visual_pdac_unlabel_uni2_A3.ipynb`.
