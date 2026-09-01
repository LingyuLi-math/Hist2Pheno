## 20260706 copy from Xenium_lung/demo.sh

cd /home/lingyu/ssd2/Python/Hist2Pheno/
  conda activate SeededNTM

## 2026.07.06 LLY: process the CODEX cell type annotation on the HE image

0_Data_process_visual_codex_HEcelltype.ipynb
  ## Transfer .zarray into .tif

1_s4769_img_cell_mapping.py
  using command: 
  conda run --no-capture-output -n SeededNTM python -u code/CODEX_hcc/1_s4769_img_cell_mapping.py
2_match_codex_cells_with_pixel.py 
  using command: 
  cd /home/lingyu/ssd2/Python/Hist2Pheno
  conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py \
    --he-key awy-98938_aligned_0d535a74
  conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py
3_demo_GT_feature_extraction_Single.sh
  ## GT: using command: conda run --no-capture-output -n SeededNTM bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh
  ## StarDist: using command: conda run --no-capture-output -n SeededNTM bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh stardist
4_HCC_train_validate_cv_UNIlabel.py
  # Must use SeededNTM (base anaconda has NumPy 2.x vs pandas mismatch).
  # 单样本全流程（默认开 spatial）
  conda run --no-capture-output -n SeededNTM python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py \
    --sample awy-98938_aligned_0d535a74
  # 全部 36 个样本（per-sample）
  conda run --no-capture-output -n SeededNTM python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py
  # 跨数据集 + lung 同款 spatial
  conda run --no-capture-output -n SeededNTM python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py \
    --mode cross-dataset \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --pooled-save-result result_all_spatial

5_transer_embedding_label_h5ad.py
  # transfer the all stardist embedding and label to h5ad file for downstream analysis
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_hcc/transer_embedding_label_h5ad.py \
    --steps stardist_all_h5ad

##############################################################################
## 2026.08.20 LLY: add stardist_all_h5ad for StarDist predict on all nuclei of 36 StarDist_Segment
## Call the other six datasets that have no annotations

# cd /home/lingyu/ssd2/Python/Hist2Pheno

# # Extract embeddings first, 
# for MATCHED_HE in \
#   awm-10421_aligned_to_FinalLiv-27_c001_v001_r001_reg029 \
#   awm-10421_aligned_to_FinalLiv-27_c001_v001_r001_reg030 \
#   mwc-11504_aligned_to_FinalLiv-27_c001_v001_r001_reg031 \
#   mwc-11504_aligned_to_FinalLiv-27_c001_v001_r001_reg032 \
#   clk-37422_aligned_to_FinalLiv-27_c002_v001_r001_reg001 \
#   clk-37422_aligned_to_FinalLiv-27_c002_v001_r001_reg002
# do
#   COORD=stardist HE_KEY="$MATCHED_HE" \
#     bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh stardist && \
#   conda run --no-capture-output -n SeededNTM python -u \
#     code/CODEX_hcc/transer_embedding_label_h5ad.py \
#     --sample "$MATCHED_HE" \
#     --steps stardist_all_h5ad \
#     --stardist-root data/CODEX/HCC/StarDist_Segment
# done


# # then rebuild h5ad:
# for MATCHED_HE in \
#   dpn-56105_aligned_93258926 \
#   xid-87175_aligned_f82fb393 \
#   awm-10421_aligned_to_FinalLiv-27_c001_v001_r001_reg029 \
#   awm-10421_aligned_to_FinalLiv-27_c001_v001_r001_reg030 \
#   mwc-11504_aligned_to_FinalLiv-27_c001_v001_r001_reg031 \
#   mwc-11504_aligned_to_FinalLiv-27_c001_v001_r001_reg032 \
#   clk-37422_aligned_to_FinalLiv-27_c002_v001_r001_reg001 \
#   clk-37422_aligned_to_FinalLiv-27_c002_v001_r001_reg002
# do
#   conda run --no-capture-output -n SeededNTM python -u \
#     code/CODEX_hcc/transer_embedding_label_h5ad.py \
#     --sample "$MATCHED_HE" \
#     --steps stardist_all_h5ad \
#     --stardist-root data/CODEX/HCC/StarDist_Segment
# done

## Only test one sample (concluded in StarDist_Segment)
# conda run --no-capture-output -n SeededNTM python -u \
#   code/CODEX_hcc/transer_embedding_label_h5ad.py \
#   --sample dpn-56105_aligned_93258926 \
#   --steps stardist_all_h5ad \
#   --stardist-root data/CODEX/HCC/StarDist_Segment
##############################################################################

6_HCC_train_validate_cv_UNIlabel_all.ipynb
  # obtain the StarDist macro AUROC by clinical groups

7_Pred_statistic_visual_hcc_all.ipynb 
  # See the prediction difference between clinical groups

