%cd /home/lingyu/ssd2/Python/Hist2Pheno/

########################################################
## Pro-prescription visualization (NICT, tumor6)
## - coords_csvs: tumor6 segment only (spatial plots + ROI CSVs)
## - coords_csvs_all: three exports concatenated (global NICT_CellPixCoords_all.csv)
########################################################
!python code/CODEX_escc/data_process_visualize.py \
  --dataset NICT \
  --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6" \
  --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
  --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6/export/NICT-measurements-tumor6.csv" \
  --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_1-5&7-14/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6/export/NICT-measurements-tumor6.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_15-28/export/measurements.csv" \
  --parent_value tumor6 \
  --save_spatial_jpg

########################################################
## Feature extraction - CODEX (all ROIs)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NICT \
  --position_path data/CODEX/ESCC/he_cell_coords/NICT_CellPixCoords4ViT_all.csv \
  --rawimage_path 'data/CODEX/ESCC/NICT/NICT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all/ \
  --scale 0.5

########################################################
## Feature extraction - CODEX (tumor6 ROI)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NICT \
  --position_path data/CODEX/ESCC/he_cell_coords/NICT_CellPixCoords4ViT_tumor6.csv \
  --rawimage_path 'data/CODEX/ESCC/NICT/NICT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6/ \
  --scale 0.5


########################################################
## StarDist coordinate detection
########################################################
# Yu lu

########################################################
## PCF2HE2StarDist alignment
########################################################
# http://localhost:8888/notebooks/Collaborate/esccAI/code/NCRT_valid.ipynb

## For tumor6 ROI
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data NICT \
  --parent_value tumor6 \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/CODEX/ESCC/codex_meta_celltype_final.csv \
  --save_plots


## For all ROIs
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data NICT \
  --parent_value all \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/CODEX/ESCC/codex_meta_celltype_final.csv \
  --save_plots


########################################################
## Feature extraction - StarDist - tumor6
########################################################
!python code/Image_feature_extraction.py \
   --dataset sc_NICT \
   --position_path data/CODEX/ESCC/he_cell_coords/NICT_CellPixCoords_tumor6_StarDist_ViT.csv \
   --rawimage_path 'data/CODEX/ESCC/NICT/NICT HE.qptiff' \
   --scale_image False \
   --method HIPT \
   --patch_size 16 \
   --output_img data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6_stardist/sc_pth_16_16_image \
   --output_pth data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6_stardist/sc_pth_16_16 \
   --logging data/CODEX/ESCC/NICT/NICT_project_tumor6/ImgEmbeddings_tumor6_stardist/ \
   --scale 0.5

########################################################
## Feature extraction - StarDist - all ROIs
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_NICT \
  --position_path data/CODEX/ESCC/he_cell_coords/NICT_CellPixCoords_all_StarDist_ViT.csv \
  --rawimage_path 'data/CODEX/ESCC/NICT/NICT HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16 \
  --logging data/CODEX/ESCC/NICT/NICT_project_all/ImgEmbeddings_all_stardist/ \
  --scale 0.5


########################################################
## Train/validate (CODEX + StarDist eval)
########################################################
# Default: train and save. Load-only: --checkpoint_exists true
conda activate SeededNTM

# using tumor6 to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data NICT \
  --parent_value tumor6 \
  --parent_value_stardist tumor6 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval

# using all ROIs to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data NICT \
  --parent_value all \
  --parent_value_stardist tumor6 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --spatial_fig_w 64 \
  --spatial_fig_h 48 \
  --spatial_point_size 0.5
# exit 0

