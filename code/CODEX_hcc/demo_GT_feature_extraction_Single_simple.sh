#!/usr/bin/env bash
## 20260706 CODEX HCC s4769 — single-sample UNI feature extraction (reg001 example)

## 2026.07.06 LLY: RUN this script to extract the HE cell feature for selected samples
# cd /home/lingyu/ssd2/Python/Hist2Pheno
# bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh

# position: {acq_id}.cell_data.csv (X,Y = aligned HE full-res pixels; see Image_feature_extraction.py)
# HE: ~0.5 um/px → scale=1.0 so patch_size=16 -> ~8 um (not Xenium 0.425)




########################################################
# 2026.07.06 LLY: RUN this script to extract the HE cell feature from CODEX for selected samples
########################################################
set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
PYTHON="/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python"
ACQ_ID="${ACQ_ID:-FinalLiv-27_c001_v001_r001_reg001}"
HE_KEY="${HE_KEY:-awy-98938_aligned_0d535a74}"
S4769="${REPO}/data/CODEX/HCC/Michael_data_transfer/s4769"
STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment}"

cd "${REPO}"

"${PYTHON}" code/Image_feature_extraction.py \
  --dataset "sc_${ACQ_ID}" \
  --position_path "${S4769}/${ACQ_ID}/${ACQ_ID}.cell_data.csv" \
  --rawimage_path "${S4769}/HE/${HE_KEY}/figures/${HE_KEY}.tif" \
  --scale_image False \
  --method UNI \
  --patch_size 16 \
  --output_img "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all/sc_pth_16_16_image" \
  --output_pth "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all/sc_pth_16_16" \
  --logging "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all/" \



########################################################
# 2026.07.06 LLY: StarDist coords on aligned HE (one CSV per HE_KEY, not per ACQ_ID)
#   {STARDIST_ROOT}/{HE_KEY}/{HE_KEY}_Float_prob0.01_nms_0.3.csv  (centroid_x, centroid_y)
########################################################
STARDIST_CSV="${STARDIST_ROOT}/${HE_KEY}/${HE_KEY}_Float_prob0.01_nms_0.3.csv"
if [[ ! -f "${STARDIST_CSV}" ]]; then
  echo "ERROR: StarDist CSV not found: ${STARDIST_CSV}" >&2
  exit 1
fi

"${PYTHON}" code/Image_feature_extraction.py \
  --dataset "sc_${HE_KEY}" \
  --position_path "${STARDIST_CSV}" \
  --rawimage_path "${S4769}/HE/${HE_KEY}/figures/${HE_KEY}.tif" \
  --scale_image False \
  --method UNI \
  --patch_size 16 \
  --output_img "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16_image" \
  --output_pth "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all_stardist/sc_pth_16_16" \
  --logging "${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all_stardist/" \

########################################################
## Feature extraction - all samples 
########################################################
# SAMPLE=VUHD113 bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh   # one sample
# bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh
# bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist
