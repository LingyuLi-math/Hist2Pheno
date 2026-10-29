# Lung Incomplete_Cases — pan-cancer Hist2Pheno

20 Xenium lung slides under `Incomplete_Cases` have StarDist nuclei but **no
usable cell-type GT** for ROC. UNI `_all_features_stardist.h5ad` files already
exist. The pan-cancer **cross-section** checkpoint
`D_emph_L2_spatial_pancancer` was already applied:

`data/PanCancer/pred_all_spatial/incomplete_lung/{sample}/{sample}_all_stardist_pred.h5ad`

This folder lists those preds and plots **all 20** slides. The first five are
the best matched-cell **L2 AUROC** among the 17 Incomplete_Cases that still
have a `*_matched_features.h5ad`. TILD028LA / TILD049MA / VUHD038 have no
matched GT and are drawn last. There is **no TLS / SRI / TNI** here.

```bash
bash code/External_val/Lung_Incomplete/demo.sh rank
```

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
bash code/External_val/Lung_Incomplete/demo.sh status
# predict is a no-op unless a slide is missing
CUDA_DEVICE=0 bash code/External_val/Lung_Incomplete/demo.sh predict
```

Open `Pred_visual_lung_incomplete.ipynb`. HE is `{sample}-HE.tif` next to the
feature h5ad.

## UNI2 A2

`demo_uni2.sh` scores the paired_v2 A2 weight (fold 1, spatial k=8 mean, no gate):

`data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt`

Predictions go to `data/PanCancer/pred_uni2_A2_bestmlp_paired_v2_seed42/unlabel/Lung_Incomplete/`.
This does not rewrite `pred_all_spatial/incomplete_lung/`.

```bash
CUDA_DEVICE=0 bash code/External_val/Lung_Incomplete/demo_uni2.sh predict
```

Open `Pred_visual_lung_incomplete_uni2_A2.ipynb`. It uses the same 20-slide
order as the UNI notebook (top 5 matched L2 AUROC first) and reads the A2
prediction directory.
