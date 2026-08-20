## 2026.08.20 Hist2Pheno on CODEX GIST (s1167 GIST TMA)
## Analog of code/CODEX_pdac/demo.sh
## All 550 GIST cores are annotated (no Incomplete_Cases).

%cd /home/lingyu/ssd2/Python/Hist2Pheno/
  conda activate SeededNTM

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
    --steps stardist_all_h5ad

4_GIST_train_validate_cv_UNIlabel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
    --sample Charvill-94_c013_v001_r001_reg002
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gist/GIST_train_validate_cv_UNIlabel.py \
    --mode cross-dataset \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --pooled-save-result result_all_spatial

5_GIST_train_validate_cv_UNIlabel_all.ipynb
  ## §1–§5: 550 annotated; StarDist-all spatial maps (overview: 2 cores / coverslip)
