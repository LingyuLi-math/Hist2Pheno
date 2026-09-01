## 20260520 copy from demo.sh

cd /home/lingyu/ssd2/Python/Hist2Pheno/

## 2026.06.22: LLY, Define the order of .py
1_extract_cell_spatial_coords.py
2_match_HEanno_with_sample_pix.py
3_check_select_consistency.py
4_plot_HEanno_spatial_labels.py
5_copy_stardist_to_cases.py
## 2026.06.24 LLY Transer embedding+label to h5ad for Xenium lung
6_transer_embedding_label_h5ad.py
## 2026.06.27 LLY: Train and validate the model
7_Lung_train_validate_cv_UNIlabel.py
8_histology_derived_niche_index.py


## 2026.07.05 LLY: predict the StarDist all h5ad on Incomplete_Cases
##                 see Active Proliferative Niche Burden (APNB) difference with clinical group
9_transer_embedding_label_h5ad.py
  using command: conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/transer_embedding_label_h5ad.py --cases-set incomplete --steps stardist_all_h5ad
10_Lung_train_validate_cv_UNIlabel_all.py
  using command: conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --mode cross-dataset --cases-set incomplete --pooled-save-result result_all_spatial --use-spatial-context --spatial-k 8 --spatial-mode mean
11_histology_derived_niche_index.py
  using command: conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/histology_derived_niche_index.py

