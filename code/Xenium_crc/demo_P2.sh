## P2-only shortcut. Full P1/P2/P5 index: code/Xenium_crc/demo.sh
## 2026.09.14 LLY: Xenium CRC P2 Hist2Pheno command index (twin of Xenium_brca/demo.sh)
## Steps 0–3 belong here. Train / validate is CRC_train_validate_cv_UNIlabel_single.ipynb.

cd /home/lingyu/ssd2/Python/Hist2Pheno/
# conda activate SeededNTM

SAMPLE="${SAMPLE:-P2CRC}"
CASES=data/Xemium/CRC/Cases

## Labels are transferred Visium HD RCTD (singlet Label1), not native Xenium GT.
## HE: data/Xemium/CRC/HE_images/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_image.ome.tif
##     (~0.274 µm/px; UNI --scale 0.548). StarDist ~1.5M nuclei — long GPU jobs.


## 0. Preprocess + transfer Visium HD RCTD onto P2 Xenium cells
0_Data_process_HEcelltype_CRC.ipynb
  ## writes data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx (P2_CRC)
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC

## 1. Join hierarchy + HE pixels (visium_scale) + match StarDist
1_match_xenium_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_pixel.py --sample P2CRC
  ## writes:
  ##   Cases/P2CRC/P2CRC_cells_with_pixel.csv
  ##   Cases/P2CRC/P2CRC_cells_matched_by_stardist.csv

## 2. UNI embeddings (HE OME ≈ 0.274 µm/px → --scale ~0.548)
2_demo_UNI_feature_extraction_batch.sh
  SAMPLE=P2CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
  SAMPLE=P2CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
  SAMPLE=P2CRC bash code/Xenium_crc/demo_GT_feature_extraction_Single.sh gt

## 3. matched / StarDist h5ad (optional CLI; the _single notebook can also build these)
3_transer_embedding_label_h5ad.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --sample P2CRC
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --sample P2CRC --steps he_h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --sample P2CRC --steps stardist_all_h5ad

## 4. Train / validate — P2 notebook (not the CLI unless you want a terminal twin)
4_CRC_train_validate_cv_UNIlabel_single.ipynb
  ## default SAMPLE=P2CRC; three-head L2/L12/L1, spatial k=8 mean
  ## CLI twin (optional):
  # conda run --no-capture-output -n SeededNTM python -u \
  #   code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py --sample P2CRC \
  #     --use-spatial-context --spatial-k 8 --spatial-mode mean \
  #     --ablation-tag D_emph_L2_spatial_crc


## Optional clean of P2 UNI / h5ad / train outputs (keeps match CSVs + StarDist CSV)
# /bin/rm -rf -- "$CASES/$SAMPLE/project_all_UNI/ImgEmbeddings_all"
# /bin/rm -f -- "$CASES/$SAMPLE/${SAMPLE}_matched_features.h5ad"
# /bin/rm -rf -- "$CASES/$SAMPLE/project_all_UNI/result" \
#                "$CASES/$SAMPLE/project_all_UNI/ablation_kfold_ckpts"
# /bin/rm -f -- "$CASES/$SAMPLE/project_all_UNI/best_mlp_gpu.pt"
# /bin/rm -rf -- data/Xemium/CRC/Results/result_all_spatial
