## 2026.09.03 LLY: Xenium BRCA Hist2Pheno command index (twin of Xenium_lung/demo.sh)

cd /home/lingyu/ssd2/Python/Hist2Pheno/
# conda activate SeededNTM

## Clean after HE-coord fix (crop align). Keeps:
##   *_cells_*.csv, ImgEmbeddings_all_stardist, *_stardist*.h5ad
## Removes: GT ImgEmbeddings_all (pth names = old HE px), broken HE matched h5ad,
##          train outputs. MUST re-extract GT UNI then rebuild HE h5ad.
# SAMPLE=rep1   # optional: only one replicate; default both
CASES=data/Xemium/BRCA/Cases
SAMPLES="${SAMPLE:-rep1 rep2}"
for s in $SAMPLES; do
  /bin/rm -rf -- "$CASES/$s/project_all_UNI/ImgEmbeddings_all"
  /bin/rm -f -- "$CASES/$s/${s}_matched_features.h5ad"
  /bin/rm -rf -- "$CASES/$s/project_all_UNI/result" \
                 "$CASES/$s/project_all_UNI/ablation_kfold_ckpts"
  /bin/rm -f -- "$CASES/$s/project_all_UNI/best_mlp_gpu.pt"
done
/bin/rm -rf -- data/Xemium/BRCA/Results/result_all_spatial



## 0. Preprocess Xenium matrix + supervised labels (rep1 / rep2)
0_Data_process_HEcelltype_BRCA.ipynb
  ## writes data/Xemium/BRCA/Results/xenium_brca_rep{1,2}_*

## 1. Join LY celltype hierarchy + match StarDist
1_match_xenium_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/match_xenium_cells_with_pixel.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/match_xenium_cells_with_pixel.py --sample rep1

## 2. UNI embeddings (HE tif ≈ 0.42 µm/px after alignment → --scale ~0.84)
2_demo_UNI_feature_extraction_batch.sh
  bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh gt
  bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh stardist
  SAMPLE=rep1 bash code/Xenium_brca/demo_GT_feature_extraction_Single.sh gt

## 3. matched / StarDist h5ad
3_transer_embedding_label_h5ad.py
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/transer_embedding_label_h5ad.py --sample rep1
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/transer_embedding_label_h5ad.py --steps he_h5ad
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/transer_embedding_label_h5ad.py --steps stardist_all_h5ad

## 4. Train / validate (three-head L2/L12/L1, spatial k=8 mean)
4_BRCA_train_validate_cv_UNIlabel.py
  # one replicate
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/BRCA_train_validate_cv_UNIlabel.py --sample rep1
  # both replicates, per-sample
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/BRCA_train_validate_cv_UNIlabel.py
  # cross-dataset (recommended for “how well does Hist2Pheno work on BRCA”)
  conda run --no-capture-output -n SeededNTM python -u \
    code/Xenium_brca/BRCA_train_validate_cv_UNIlabel.py \
      --mode cross-dataset \
      --use-spatial-context --spatial-k 8 --spatial-mode mean \
      --pooled-save-result result_all_spatial \
      --ablation-tag D_emph_L2_spatial_brca

## 5. Notebooks
5_BRCA_train_validate_cv_UNIlabel_single.ipynb
  ## debug one replicate (default SAMPLE=rep1)
5_BRCA_train_validate_cv_UNIlabel_all.ipynb
  ## pool rep1+rep2; set SKIP_POOLED_TRAIN=True to load CLI weights
