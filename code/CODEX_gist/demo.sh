## 2026.08.20 Hist2Pheno on CODEX GIST (s1167 GIST TMA)
## Analog of code/CODEX_pdac/demo.sh
## All 550 GIST cores are annotated (no Incomplete_Cases).



1. match_codex_cells_with_pixel.py / 3. transer_embedding_label_h5ad.py
CPU 为主，不用指定 GPU。
2. UNI 特征提取（Image_feature_extraction.py 用可见的 cuda:0）
CUDA_VISIBLE_DEVICES=2 bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh gt
CUDA_VISIBLE_DEVICES=2 bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh stardist

#########################################################################


# Run from Hist2Pheno repo root (not ~/ssd2/Python). %cd is IPython-only.
cd /home/lingyu/ssd2/Python/Hist2Pheno/
  conda activate SeededNTM
  export CUDA_VISIBLE_DEVICES=2    # 换一张卡

## Counts:
##   GIST TMA: 550 cores (c009 160 + c011 197 + c013 193)
##   annotated (cell-type CSV): 550 / 550
## Pancreas TMA / PDAC is a separate pipeline: code/CODEX_pdac

1_match_codex_cells_with_pixel.py
  # 550 annotated cores → {acq}_cells_with_pixel.csv + {acq}_cells_matched_by_stardist.csv
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/match_codex_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/match_codex_cells_with_pixel.py \
    --acq-id Charvill-94_c013_v001_r001_reg002

2_demo_GT_feature_extraction_Single.sh / demo_UNI_feature_extraction_batch.sh
  ## GT UNI (550):
  bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh gt
  ## StarDist UNI (550):
  bash code/CODEX_gist/demo_UNI_feature_extraction_batch.sh stardist

3_transer_embedding_label_h5ad.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/transer_embedding_label_h5ad.py \
    --sample Charvill-94_c013_v001_r001_reg002 \
    --steps he_h5ad stardist_csv stardist_h5ad stardist_all_h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/transer_embedding_label_h5ad.py \
    --steps stardist_h5ad stardist_all_h5ad

4_GIST_train_validate_cv_UNIlabel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
    --sample Charvill-94_c013_v001_r001_reg002
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
    --mode cross-dataset \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --pooled-save-result result_all_spatial_gist

4b_GIST_train_validate_cv_UNIlabel_single.ipynb
  ## One annotated core: HE h5ad → K-fold train → HE validate → StarDist
  ## Demo: Charvill-94_c013_v001_r001_reg002

5_GIST_train_validate_cv_UNIlabel_all.ipynb
  ## §1–§5: 550 annotated; StarDist-all spatial maps (overview: 2 cores / coverslip)

6_GIST_histology_derived_niche_index.ipynb
  ## TLS / SRI / TNI on matched StarDist (demo: Charvill-94_c013_v001_r001_reg002)
  ## §6 QUICK_VALIDATE=True → first 6 cores; set False for all 550
  ## out: s1167/result_all_spatial_gist/niche_index_gist/

7_Pred_statistic_visual_gist_all.ipynb
  ## 550 annotated: pooled ROC + macro AUROC vs coverslip / SAMPLE_LABEL
  ## helpers: CODEX_pdac/s1167_plot.py (not gist/s1167_plot.py)
  ## out: s1167/result_all_spatial_gist/clinical_viz_gist/
