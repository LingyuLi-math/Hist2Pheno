## 2026.09.03 LLY: CODEX GBM (WangLab Visium HD) Hist2Pheno command index

cd /home/lingyu/ssd2/Python/Hist2Pheno/
# conda activate SeededNTM

## 2026.09.07, for gbm, add function to build nuclei-level annotation table: cell type + primary SN (+ Rec GD)
conda run --no-capture-output -n SeededNTM python -u \
  code/CODEX_gbm/build_nuclei_sn_gd_annotation.py



## 0. Preprocess single-nuclei annotation + microscope HE pixels (Ini / Rec)
0_Data_process_HEcelltype_GBM.ipynb
  ## writes data/CODEX/GBM/Results/gbm_{P174511_Initial,P179161_Recurrent}_*
  ## drops SN LowQ/missing and Unknown/LowQ cell labels so GT triples are trainable

## 1. Join Celltype hierarchy + match StarDist
1_match_codex_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/match_codex_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/match_codex_cells_with_pixel.py --sample P174511_Initial

## 2. UNI embeddings (microscope HE: scale_image=False, patch_size=16 — HCC style)
2_demo_UNI_feature_extraction_batch.sh
  bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh gt
  bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh stardist
  SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_GT_feature_extraction_Single.sh gt

## 3. matched / StarDist h5ad
3_transer_embedding_label_h5ad.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/transer_embedding_label_h5ad.py --sample P174511_Initial
  # Optional / expensive on Rec (~1.16M nuclei):
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/transer_embedding_label_h5ad.py \
    --sample P179161_Recurrent --steps stardist_all_h5ad

## 4. Train / validate (three-head L2/L12/L1, spatial k=8 mean)
4_GBM_train_validate_cv_UNIlabel.py
  # one sample
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/GBM_train_validate_cv_UNIlabel.py --sample P174511_Initial
  # both samples, per-sample
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/GBM_train_validate_cv_UNIlabel.py
  # cross-dataset leave-one-replicate-out (recommended)
  conda run --no-capture-output -n SeededNTM python -u \
    code/CODEX_gbm/GBM_train_validate_cv_UNIlabel.py \
      --mode cross-dataset \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --pooled-save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_gbm

## 5. Notebooks
5_GBM_train_validate_cv_UNIlabel_single.ipynb
  ## debug one sample (default P174511_Initial)
5_GBM_train_validate_cv_UNIlabel_all.ipynb
  ## pool Ini+Rec; set SKIP_POOLED_TRAIN=True to load CLI weights
5_Pred_statistic_visual_gbm_all.ipynb
  ## Initial vs Recurrent StarDist macro AUROC
