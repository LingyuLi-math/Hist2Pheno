## 2026.09.15 LLY: Xenium CRC Hist2Pheno command index (twin of Xenium_brca/demo.sh)
## Samples: P1CRC / P2CRC / P5CRC  (Excel / folders P1_CRC, P2_CRC, P5_CRC)
## Labels: transferred Visium HD RCTD singlet, not native Xenium GT.
## P2-only shortcut remains: demo_P2.sh
##
## HE OME ≈ 0.274 µm/px → UNI --scale 0.548. StarDist is whole-WSI.
## P1/P5 UNI skips sc_pth_16_16_image (--no_save_patch_images).
## P5 StarDist CSV may still be missing — GT UNI / per-sample train still run.

cd /home/lingyu/ssd2/Python/Hist2Pheno/
# conda activate SeededNTM

CASES=data/Xemium/CRC/Cases
SAMPLES="${SAMPLE:-P1CRC P2CRC P5CRC}"

## Optional clean of UNI / h5ad / train outputs (keeps match CSVs + StarDist CSV)
# for s in $SAMPLES; do
#   /bin/rm -rf -- "$CASES/$s/project_all_UNI/ImgEmbeddings_all"
#   /bin/rm -f -- "$CASES/$s/${s}_matched_features.h5ad"
#   /bin/rm -rf -- "$CASES/$s/project_all_UNI/result" \
#                  "$CASES/$s/project_all_UNI/ablation_kfold_ckpts"
#   /bin/rm -f -- "$CASES/$s/project_all_UNI/best_mlp_gpu.pt"
# done
# /bin/rm -rf -- data/Xemium/CRC/Results/result_all_spatial



## 0. Transfer Visium HD RCTD onto Xenium cells (writes Excel P{1,2,5}_CRC)
0_Data_process_HEcelltype_CRC.ipynb
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_visiumhd.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC

## 1. Join hierarchy + HE pixels (visium_scale) + match StarDist
1_match_xenium_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/match_xenium_cells_with_pixel.py --sample P2CRC
  ## writes Cases/{P1,P2,P5}CRC/{sample}_cells_with_pixel.csv
  ##         Cases/{sample}/{sample}_cells_matched_by_stardist.csv (if StarDist CSV exists)

## 2. UNI embeddings (HE OME ≈ 0.274 µm/px → --scale ~0.548)
2_demo_UNI_feature_extraction_batch.sh
  bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
  bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
  SAMPLE=P2CRC bash code/Xenium_crc/demo_GT_feature_extraction_Single.sh gt
  SAMPLE=P1CRC SAVE_PATCH_IMAGES=0 bash code/Xenium_crc/demo_GT_feature_extraction_Single.sh gt

  SAMPLE=P1CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
  SAMPLE=P1CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist

## 3. matched / StarDist h5ad
3_transer_embedding_label_h5ad.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --sample P2CRC
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --steps he_h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/transer_embedding_label_h5ad.py --steps stardist_all_h5ad

## 4. Train / validate (three-head L2/L12/L1, spatial k=8 mean)
4_CRC_train_validate_cv_UNIlabel.py
  # one sample
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py --sample P2CRC
  # all samples with Cases CSVs, per-sample
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py
  # cross-dataset pool (P1+P2+P5 when h5ads exist)
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py \
      --mode cross-dataset \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --pooled-save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_crc

  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py \
      --mode cross-dataset \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --pooled-save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_crc \
      --no-resume-from-checkpoints

## 5. Notebooks
5_CRC_train_validate_cv_UNIlabel_single.ipynb
  ## debug one sample (default SAMPLE=P2CRC)
5_CRC_train_validate_cv_UNIlabel_all.ipynb
  ## pool P1+P2+P5; set SKIP_POOLED_TRAIN=True to load CLI weights
