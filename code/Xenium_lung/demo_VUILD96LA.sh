## 20260520 copy from demo.sh
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# bash code/Xenium_lung/demo_VUILD96LA.sh

%cd /home/lingyu/ssd2/Python/Collaborate/esccAI/

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
## Feature extraction - Xenium Select4 — VUILD96LA
########################################################
## 2026.04.14 LLY update using UNI method
## 2026.05.20 Xenium Select4 — VUILD96LA (CSV: X_pix_HE/Y_pix_HE from 0.2125 um/px)
## Target 0.5 um/px so patch_size=16 -> ~8 um: scale_image=True, scale=0.2125/0.5=0.425
# Use SeededNTM env (base anaconda python has numpy 2.x vs skimage ABI mismatch)
/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python code/Image_feature_extraction.py \
   --dataset sc_VUILD96LA \
   --position_path data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_cells_partitioned_by_annotation.csv \
   --rawimage_path 'data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA-HE.tif' \
   --scale_image True \
   --method UNI \
   --patch_size 16 \
   --output_img data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16_image \
   --output_pth data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16 \
   --logging data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all/ \
   --scale 0.425

########################################################
## StarDist coordinate detection
########################################################
# Yu lu

########################################################
## Feature extraction - StarDist (HE nuclei detections)
########################################################
## centroid_x/centroid_y = full-res HE pixels; same scale 0.425 -> 0.5 um/px, patch 16 ~ 8 um
/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python code/Image_feature_extraction.py \
   --dataset sc_VUILD96LA \
   --position_path data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_Float_prob001_nms_03.csv \
   --rawimage_path 'data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA-HE.tif' \
   --scale_image True \
   --method UNI \
   --patch_size 16 \
   --output_img data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
   --output_pth data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16 \
   --logging data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUILD96LA/VUILD96LA_project_all_UNI/ImgEmbeddings_all_stardist/ \
   --scale 0.425

# ########################################################
# ## model train and validate for StarDist prediction
# ########################################################
# # Default: train and save. Load-only (no training): --checkpoint_exists true
# conda activate SeededNTM
#
# # using tumor1 to train and validate
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --cv_mode logo \
#   --logo_spatial_nx 2 \
#   --logo_spatial_ny 2 \
#   --logo_resume true \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1
#
# # For UNI method
# python code/model_train_validate_cv_f1.py \
#   --therapy_model NCRT_project_tumor1_UNI \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode logo \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1
#
#
# # using all ROIs to train and validate
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value all \
#   --cuda_device 2 \
#   --cv_mode logo \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1 \
#   --spatial_fig_w 64 \
#   --spatial_fig_h 48 \
#   --spatial_point_size 0.5
#
#
# # python code/model_train_validate_cv.py \
# #   --therapy_data NCRT \
# #   --parent_value all \
# #   --cuda_device 2 \
# #   --hidden_dims 256 256 256 \
# #   --cv_mode logo \
# #   --save_result result \
# #   --run_all_data_eval \
# #   --run_stardist_eval \
# #   --parent_value_stardist tumor1 \
# #   --spatial_fig_w 64 \
# #   --spatial_fig_h 48 \
# #   --spatial_point_size 0.5

