%cd /home/lingyu/ssd2/Python/Hist2Pheno/

########################################################
## Pro-prescription visualization (NCT, tumor1)
########################################################
!python code/CODEX_escc/data_process_visualize.py \
  --dataset NCT \
  --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14" \
  --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
  --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14/export/NCT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_15-28/export/NCT-measurements-tumor-15-28.csv" \
  --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14/export/NCT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_15-28/export/NCT-measurements-tumor-15-28.csv" \
  --parent_value tumor1 \
  --save_spatial_jpg

########################################################
## Feature extraction - CODEX (all ROIs)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NCT \
  --position_path data/CODEX/ESCC/he_cell_coords/NCT_CellPixCoords4ViT_all.csv \
  --rawimage_path 'data/CODEX/ESCC/NCT/NCT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all/ \
  --scale 0.5

########################################################
## Feature extraction - CODEX (tumor1 ROI)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NCT \
  --position_path data/CODEX/ESCC/he_cell_coords/NCT_CellPixCoords4ViT_tumor1.csv \
  --rawimage_path 'data/CODEX/ESCC/NCT/NCT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1/ \
  --scale 0.5


########################################################
## StarDist coordinate detection
########################################################
# Yu lu

########################################################
## PCF2HE2StarDist alignment
########################################################
# http://localhost:8888/notebooks/Collaborate/esccAI/code/NCRT_valid.ipynb

## For tumor1 ROI
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data NCT \
  --parent_value tumor1 \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/CODEX/ESCC/codex_meta_celltype_final.csv \
  --save_plots


## For all ROIs
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data NCT \
  --parent_value all \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/CODEX/ESCC/codex_meta_celltype_final.csv \
  --save_plots


########################################################
## Feature extraction - StarDist - tumor1
########################################################
!python code/Image_feature_extraction.py \
   --dataset sc_NCT \
   --position_path data/CODEX/ESCC/he_cell_coords/NCT_CellPixCoords_tumor1_StarDist_ViT.csv \
   --rawimage_path 'data/CODEX/ESCC/NCT/NCT HE.qptiff' \
   --scale_image False \
   --method HIPT \
   --patch_size 16 \
   --output_img data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1_stardist/sc_pth_16_16_image \
   --output_pth data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1_stardist/sc_pth_16_16 \
   --logging data/CODEX/ESCC/NCT/NCT_project_tumor1/ImgEmbeddings_tumor1_stardist/ \
   --scale 0.5

########################################################
## Feature extraction - StarDist - all ROIs
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NCT \
  --position_path data/CODEX/ESCC/he_cell_coords/NCT_CellPixCoords_all_StarDist_ViT.csv \
  --rawimage_path 'data/CODEX/ESCC/NCT/NCT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NCT/NCT_project_all/ImgEmbeddings_all_stardist/ \
  --scale 0.5


########################################################
## Train/validate (CODEX + StarDist eval)
########################################################
# Default: train and save. Load-only: --checkpoint_exists true
conda activate SeededNTM

# using tumor1 to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data NCT \
  --parent_value tumor1 \
  --parent_value_stardist tumor1 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval

# using all ROIs to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data NCT \
  --parent_value all \
  --parent_value_stardist tumor1 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --spatial_fig_w 64 \
  --spatial_fig_h 48 \
  --spatial_point_size 0.5
# exit 0

