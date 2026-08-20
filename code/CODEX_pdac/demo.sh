## 2026.08.20 Hist2Pheno on CODEX PDAC (s1167 Pancreas TMA)
## Analog of code/CODEX_hcc/demo.sh

%cd /home/lingyu/ssd2/Python/Hist2Pheno/
  conda activate SeededNTM

## Counts (on disk, not swapped):
##   Pancreas TMA: 473 cores
##   annotated (cell-type CSV): 278  → training pool (HCC 36 analog)
##   Incomplete_Cases (no cell-type CSV): 195  (c003 19 + c005 95 + c007 81)
## GIST TMA is a separate cohort (550, all annotated) and is not in this pipeline.

0_Data_process_visual_codex_HEcelltype_pdac.ipynb
  ## HE Zarr → TIFF, StarDist HE copy, cell-type JPG, coverslip composition

1_match_codex_cells_with_pixel.py
  # 278 annotated cores → {acq}_cells_with_pixel.csv + {acq}_cells_matched_by_stardist.csv
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/match_codex_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/match_codex_cells_with_pixel.py \
    --acq-id Charvill-94_c001_v001_r001_reg001

2_demo_GT_feature_extraction_Single.sh / demo_UNI_feature_extraction_batch.sh
  ## GT UNI (278):
  bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh gt
  ## StarDist UNI (278 annotated):
  bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh stardist
  ## StarDist UNI Incomplete_Cases (195):
  INCOMPLETE=1 bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh stardist

3_transer_embedding_label_h5ad.py
  ## annotated matched + all-nuclei h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/transer_embedding_label_h5ad.py \
    --sample Charvill-94_c001_v001_r001_reg001 \
    --steps he_h5ad stardist_csv stardist_h5ad stardist_all_h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/transer_embedding_label_h5ad.py \
    --steps stardist_all_h5ad
  ## Incomplete_Cases all-nuclei h5ad only
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/transer_embedding_label_h5ad.py \
    --incomplete --steps stardist_all_h5ad

4_PDAC_train_validate_cv_UNIlabel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \
    --sample Charvill-94_c001_v001_r001_reg001
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \
    --mode cross-dataset \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --pooled-save-result result_all_spatial

5_PDAC_train_validate_cv_UNIlabel_all.ipynb
  ## §1–§5: 278 annotated; §6: 195 Incomplete_Cases → stardist_Incomplete_Cases/
