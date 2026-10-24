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
