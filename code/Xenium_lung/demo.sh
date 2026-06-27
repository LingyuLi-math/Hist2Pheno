## 20260520 copy from demo.sh

%cd /home/lingyu/ssd2/Python/Collaborate/esccAI/

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




########################################################
## Pro-prescription visualization
########################################################
# !python code/data_process_visualize.py \
#    --dataset NCRT \
#    --segment_project_dir "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/NCRT/NCRT_project - seg 1-14" \
#    --celltype_anno_csv "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/codex_meta_celltype_final.csv" \
#    --coords_csvs "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/NCRT/NCRT_project - seg 1-14/export/NCRT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Collaborate/esccAI/data/NCRT/NCRT_project - seg 15-28/export/NCRT-measurements-tumor-15-28.csv" \
#    --coords_csvs_all "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/NCRT/NCRT_project - seg 1-14/export/NCRT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Collaborate/esccAI/data/NCRT/NCRT_project - seg 15-28/export/NCRT-measurements-tumor-15-28.csv" \
#    --parent_value tumor1 \
#    --save_spatial_jpg

########################################################
## Feature extraction - CODEX (all ROIs)
########################################################
# !python code/Image_feature_extraction.py \
#    --dataset sc_NCRT \
#    --position_path data/he_cell_coords/NCRT_CellPixCoords4ViT_all.csv \
#    --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
#    --scale_image False \
#    --method HIPT \
#    --patch_size 16 \
#    --output_img data/NCRT/NCRT_project_all/ImgEmbeddings_all/sc_pth_16_16_image \
#    --output_pth data/NCRT/NCRT_project_all/ImgEmbeddings_all/sc_pth_16_16 \
#    --logging data/NCRT/NCRT_project_all/ImgEmbeddings_all/ \
#    --scale 0.5  

## 2026.04.14 LLY update using UNI method
## 2026.05.20 Xenium Select4 — VUILD96LA (coords: x_centroid, y_centroid)
mkdir -p data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/sc_VUILD96LA
!python code/Image_feature_extraction.py \
   --dataset sc_VUILD96LA \
   --position_path data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_cells_partitioned_by_annotation.csv \
   --rawimage_path 'data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA-HE.tif' \
   --scale_image False \
   --method UNI \
   --patch_size 16 \
   --output_img data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16_image \
   --output_pth data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16 \
   --logging data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/ \
   --scale 0.5  


########################################################
## Feature extraction - CODEX (tumor1 ROI)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NCRT \
  --position_path data/he_cell_coords/NCRT_CellPixCoords4ViT_tumor1.csv \
  --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1/sc_pth_16_16_image \
  --output_pth data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1/sc_pth_16_16 \
  --logging data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1/ \
  --scale 0.5

## 2026.04.14 LLY update using UNI method
! CUDA_VISIBLE_DEVICES=1 python code/Image_feature_extraction.py \
  --dataset sc_NCRT \
  --position_path data/he_cell_coords/NCRT_CellPixCoords4ViT_tumor1.csv \
  --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
  --scale_image False \
  --method UNI \
  --patch_size 16 \
  --output_img data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1/sc_pth_16_16_image \
  --output_pth data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1/sc_pth_16_16 \
  --logging data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1/ \
  --scale 0.5

########################################################
## StarDist coordinate detection
########################################################
# Yu lu

########################################################
## PCF2HE2StarDist alignment
########################################################
# http://localhost:8888/notebooks/Collaborate/esccAI/code/NCRT_valid.ipynb

python code/PCF2HE2StarDist_alignment.py \
  --therapy_data NCRT \
  --parent_value tumor1 \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/codex_meta_celltype_final.csv


## 2026.05.11 LLY update using codex_meta_celltype_final012.csv - using level 1-1-2 cell type annotation
  python code/PCF2HE2StarDist_alignment.py \
  --therapy_data NCRT \
  --parent_value tumor1 \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords012 \
  --codex_meta_celltype_final data/codex_meta_celltype_final012.csv



########################################################
## Feature extraction - StarDist - tumor1
########################################################
!python code/Image_feature_extraction.py \
   --dataset sc_NCRT \
   --position_path data/he_cell_coords/NCRT_CellPixCoords_tumor1_StarDist_ViT.csv \
   --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
   --scale_image False \
   --method HIPT \
   --patch_size 16 \
   --output_img data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1_stardist/sc_pth_16_16_image \
   --output_pth data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1_stardist/sc_pth_16_16 \
   --logging data/NCRT/NCRT_project_tumor1/ImgEmbeddings_tumor1_stardist/ \
   --scale 0.5  

## 2026.04.14 LLY update using UNI method
!python code/Image_feature_extraction.py \
   --dataset sc_NCRT \
   --position_path data/he_cell_coords/NCRT_CellPixCoords_tumor1_StarDist_ViT.csv \
   --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
   --scale_image False \
   --method UNI \
   --patch_size 16 \
   --output_img data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1_stardist/sc_pth_16_16_image \
   --output_pth data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1_stardist/sc_pth_16_16 \
   --logging data/NCRT/NCRT_project_tumor1_UNI/ImgEmbeddings_tumor1_stardist/ \
   --scale 0.5  

########################################################
## Feature extraction - StarDist - all ROIs
########################################################
!python code/Image_feature_extraction.py \
   --dataset sc_NCRT \
   --position_path data/he_cell_coords/NCRT_CellPixCoords_all_StarDist_ViT.csv \
   --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
   --scale_image False \
   --method HIPT \
   --patch_size 16 \
   --output_img data/NCRT/NCRT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
   --output_pth data/NCRT/NCRT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16 \
   --logging data/NCRT/NCRT_project_all/ImgEmbeddings_all_stardist/ \
   --scale 0.5  

## 2026.04.14 LLY update using UNI method
!python code/Image_feature_extraction.py \
   --dataset sc_NCRT \
   --position_path data/he_cell_coords/NCRT_CellPixCoords_all_StarDist_ViT.csv \
   --rawimage_path 'data/NCRT/NCRT HE.qptiff' \
   --scale_image False \
   --method UNI \
   --patch_size 16 \
   --output_img data/NCRT/NCRT_project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
   --output_pth data/NCRT/NCRT_project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16 \
   --logging data/NCRT/NCRT_project_all_UNI/ImgEmbeddings_all_stardist/ \
   --scale 0.5 

########################################################
## model train and validate for StarDist prediction
########################################################
# Default: train and save. Load-only (no training): --checkpoint_exists true
conda activate SeededNTM

# using tumor1 to train and validate
python code/model_train_validate_cv.py \
  --therapy_data NCRT \
  --parent_value tumor1 \
  --cuda_device 2 \
  --cv_mode logo \
  --logo_spatial_nx 2 \
  --logo_spatial_ny 2 \
  --logo_resume true \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --parent_value_stardist tumor1

# For UNI method
python code/model_train_validate_cv_f1.py \
  --therapy_model NCRT_project_tumor1_UNI \
  --therapy_data NCRT \
  --parent_value tumor1 \
  --cuda_device 2 \
  --hce_w2 2.0 \
  --cv_mode logo \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --parent_value_stardist tumor1


# using all ROIs to train and validate
python code/model_train_validate_cv.py \
  --therapy_data NCRT \
  --parent_value all \
  --cuda_device 2 \
  --cv_mode logo \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --parent_value_stardist tumor1 \
  --spatial_fig_w 64 \
  --spatial_fig_h 48 \
  --spatial_point_size 0.5


# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value all \
#   --cuda_device 2 \
#   --hidden_dims 256 256 256 \
#   --cv_mode logo \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1 \
#   --spatial_fig_w 64 \
#   --spatial_fig_h 48 \
#   --spatial_point_size 0.5

