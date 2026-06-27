## 20260520 copy from demo.sh
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# bash code/Xenium_lung/demo_VUHD113.sh


%cd /home/lingyu/ssd2/Python/Collaborate/esccAI/


########################################################
## Feature extraction - Xenium Select4 — VUHD113
########################################################
## 2026.04.14 LLY update using UNI method
## 2026.05.20 Xenium Select4 — VUHD113 (CSV: X_pix_HE/Y_pix_HE from 0.2125 um/px)
## Target 0.5 um/px so patch_size=16 -> ~8 um: scale_image=True, scale=0.2125/0.5=0.425
# Use SeededNTM env (base anaconda python has numpy 2.x vs skimage ABI mismatch)
# python code/Image_feature_extraction.py \

/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python code/Image_feature_extraction.py \
   --dataset sc_VUHD113 \
   --position_path data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUHD113/VUHD113_cells_partitioned_by_annotation.csv \
   --rawimage_path 'data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUHD113/VUHD113-HE.tif' \
   --scale_image True \
   --method UNI \
   --patch_size 16 \
   --output_img data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUHD113/VUHD113_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16_image \
   --output_pth data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUHD113/VUHD113_project_all_UNI/ImgEmbeddings_all/sc_pth_16_16 \
   --logging data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4/VUHD113/VUHD113_project_all_UNI/ImgEmbeddings_all/ \
   --scale 0.425


########################################################
## Feature extraction - Complete_Cases - 25 samples
########################################################
SAMPLE=VUHD113 bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt complete    # one sample
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt complete    # all 25 Complete_Cases
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt incomplete  # all 20 Incomplete_Cases
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist complete



