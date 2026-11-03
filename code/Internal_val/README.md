# Internal validation (UNI2 A2 best model, fold 0)

`code/Xenium_lung/Data_process_visual_xenium_all.ipynb` and
`code/CODEX_hcc/Data_process_visual_*.ipynb` / `CODEX_pdac/...` show
**single-dataset** Hist2Pheno models.

This folder is the joint **UNI2 A2** weight
`result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/.../best_mlp_gpu.pt`
on the seven training cohorts. That file is the fold-1 checkpoint, so the
notebooks do not show fold 1. They show fold 0. Fold 0 is the only other
split that still contains every cohort (BRCA `rep1`, CRC `P1CRC`, GBM
`P174511_Initial`).

Predictions for all 896 sections are in
`data/PanCancer/pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment/`
(`matched_pred.h5ad`, StarDist-all, per-section ROC). The run skipped panel
PNGs. Notebooks draw GT versus prediction from the matched h5ad.
`demo.sh predict` only fills gaps (`--skip-existing`) and does not draw panels.

| dataset | fold 0 | notebook draws |
|---------|--------|----------------|
| Lung | 5 (`VUILD107MA` first) | all 5 |
| BRCA | 1 (`rep1`) | 1 |
| CRC | 1 (`P1CRC`) | 1 |
| HCC | 8 | 4 largest |
| PDAC | 56 | top 5 by L2 AUROC |
| GIST | 110 | 4 largest |
| GBM | 1 (`P174511_Initial`) | 1 |

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

bash code/Internal_val/demo.sh status
bash code/Internal_val/demo.sh fold0

# only if a fold-0 matched_pred.h5ad is missing
CUDA_DEVICE=1 bash code/Internal_val/demo.sh predict
```

Open `Pred_visual_{lung,brca,crc,hcc,pdac,gist,gbm}.ipynb`.
`Pred_visual_fold0.ipynb` is the inventory of this fold.
The lung notebook crops VUILD107MA for Extended Data Fig. 7.
That section is in this fold-0 set.
