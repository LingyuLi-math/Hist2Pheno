# HCC unlabeled rest — pan-cancer Hist2Pheno

Eight CODEX HCC StarDist regions have **no CODEX cell-type CSV** (the “rest”
set, not the 36 annotated cores). Distinct from `External_val/HCC`, which is
the Visium JPEG cohort.

Inputs: `data/CODEX/HCC/Michael_data_transfer/s4769/HE/{sample}/{sample}_all_features_stardist.h5ad`  
Outputs: `data/PanCancer/pred_unlabel/HCC_unlabel/`

Default `predict` / the notebook use the **5 largest** rest regions. No TLS /
SRI / TNI.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
bash code/External_val/HCC_unlabel/demo.sh status
CUDA_DEVICE=0 bash code/External_val/HCC_unlabel/demo.sh predict
```

`ALL=1` scores all 8. Open `Pred_visual_hcc_unlabel.ipynb`.
