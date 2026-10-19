# Internal validation (fold 0, pretrained pan-cancer model)

`code/Xenium_lung/Data_process_visual_xenium_all.ipynb` and
`code/CODEX_hcc/Data_process_visual_*.ipynb` / `CODEX_pdac/...` show
**single-dataset** Hist2Pheno models.

This folder is the other column: the joint **cross-section** checkpoint
`D_emph_L2_spatial_pancancer` applied to the **fold-0 hold-out** of the
same seven training cohorts (Lung, BRCA, CRC, HCC, PDAC, GIST, GBM).
A section that is in fold 0 was not used to fit `best_mlp_gpu.pt`.

Predict is already done. `data/PanCancer/pred_all_spatial/` has all 182
fold-0 sections (`matched_pred.h5ad`, `panel_4x3.png`, StarDist ROC).
`demo.sh predict` only fills gaps (`--skip-existing`).

| dataset | fold 0 | notebook draws |
|---------|--------|----------------|
| Lung | 5 | all 5 |
| BRCA | 1 (`rep1`) | 1 |
| CRC | 1 (`P2CRC`) | 1 |
| HCC | 8 | 4 (includes Fig. 2c `kqr-83230`) |
| PDAC | 56 | top 5 by L2 AUROC |
| GIST | 110 | 4 largest |
| GBM | 1 (`P174511_Initial`) | 1 |

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

bash code/Internal_val/demo.sh status
bash code/Internal_val/demo.sh fold0

# only if a fold-0 panel is missing
CUDA_DEVICE=1 bash code/Internal_val/demo.sh predict

# notebooks (optional; they do not rewrite the .ipynb)
ONLY=HCC bash code/Internal_val/demo.sh notebooks
```

Open `Pred_visual_{lung,brca,crc,hcc,pdac,gist,gbm}.ipynb`. Each shows
the existing 4×3 panel (GT / pred / ROC / StarDist-all) plus GT vs
predicted composition. `Pred_visual_fold0.ipynb` is the inventory.
