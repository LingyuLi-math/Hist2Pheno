%cd /home/lingyu/ssd2/Python/Hist2Pheno/

########################################################
## Pro-prescription visualization (SA, tumor3)
########################################################
!python code/CODEX_escc/data_process_visualize.py \
  --dataset SA \
  --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3" \
  --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
  --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3/export/measurements-3sample.csv" \
  --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3/export/measurements-3sample.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor1-2&4-10/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor11-20/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor21-28/export/measurements.csv" \
  --parent_value tumor3 \
  --save_spatial_jpg

########################################################
## Feature extraction - CODEX (all ROIs)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_SA \
  --position_path data/CODEX/ESCC/he_cell_coords/SA_CellPixCoords4ViT_all.csv \
  --rawimage_path 'data/CODEX/ESCC/SA/SA HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all/sc_pth_16_16 \
  --logging data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all/ \
  --scale 0.5

########################################################
## Feature extraction - CODEX (tumor3 ROI)
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_SA \
  --position_path data/CODEX/ESCC/he_cell_coords/SA_CellPixCoords4ViT_tumor3.csv \
  --rawimage_path 'data/CODEX/ESCC/SA/SA HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3/sc_pth_16_16 \
  --logging data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3/ \
  --scale 0.5


########################################################
## StarDist coordinate detection
########################################################
# Yu lu

########################################################
## PCF2HE2StarDist alignment
########################################################
# http://localhost:8888/notebooks/Collaborate/esccAI/code/NCRT_valid.ipynb

## For tumor3 ROI
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data SA \
  --parent_value tumor3 \
  --stardist_data StarDist_Segment \
  --qupath_corner QupathCorners \
  --transfer_data he_cell_coords \
  --codex_meta_celltype_final data/CODEX/ESCC/codex_meta_celltype_final.csv \
  --save_plots


## For all ROIs
python code/CODEX_escc/PCF2HE2StarDist_alignment.py \
  --therapy_data SA \
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
   --dataset sc_SA \
   --position_path data/CODEX/ESCC/he_cell_coords/SA_CellPixCoords_tumor3_StarDist_ViT.csv \
   --rawimage_path 'data/CODEX/ESCC/SA/SA HE.qptiff' \
   --scale_image False \
   --method HIPT \
   --patch_size 16 \
   --output_img data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3_stardist/sc_pth_16_16_image \
   --output_pth data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3_stardist/sc_pth_16_16 \
   --logging data/CODEX/ESCC/SA/SA_project_tumor3/ImgEmbeddings_tumor3_stardist/ \
   --scale 0.5  

########################################################
## Feature extraction - StarDist - all ROIs
########################################################
!python code/Image_feature_extraction.py \
  --dataset sc_SA \
  --position_path data/CODEX/ESCC/he_cell_coords/SA_CellPixCoords_all_StarDist_ViT.csv \
  --rawimage_path 'data/CODEX/ESCC/SA/SA HE.qptiff' \
  --scale_image False \
  --method HIPT \
  --patch_size 16 \
  --output_img data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16_image \
  --output_pth data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all_stardist/sc_pth_16_16 \
  --logging data/CODEX/ESCC/SA/SA_project_all/ImgEmbeddings_all_stardist/ \
  --scale 0.5  


########################################################
## Train/validate (CODEX only; StarDist not used)
########################################################
# Default: train and save. Load-only: --checkpoint_exists true (same as notebook checkpoint_exists=True).
conda activate SeededNTM

# using tumor3 to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data SA \
  --parent_value tumor3 \
  --parent_value_stardist tumor3 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval 

# using all ROIs to train and validate
python code/CODEX_escc/model_train_validate.py \
  --therapy_data SA \
  --parent_value all \
  --parent_value_stardist tumor3 \
  --save_result result \
  --run_all_data_eval \
  --run_stardist_eval \
  --spatial_fig_w 64 \
  --spatial_fig_h 48 \
  --spatial_point_size 0.5
# exit 0


