## 20260706 copy from Xenium_lung/demo.sh

%cd /home/lingyu/ssd2/Python/Hist2Pheno/
  conda activate SeededNTM

## 2026.07.06 LLY: process the CODEX cell type annotation on the HE image

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

5_HCC_train_validate_cv_UNIlabel_all.ipynb
  # Cross-dataset notebook (36 regions); shows L2 / L12 / L1 prediction results
  # Twin of code/Xenium_lung/Lung_train_validate_cv_UNIlabel_all.ipynb
  # Open in Jupyter (SeededNTM): code/CODEX_hcc/HCC_train_validate_cv_UNIlabel_all.ipynb
  # Outputs: data/CODEX/HCC/Michael_data_transfer/s4769/result_all_spatial/
  #   - internal: conf_matrix_level{2,1,12}.pdf
  #   - stardist/{MATCHED_HE}/: celltype_pred_stardist_level{2,1,12}.jpg + ROC