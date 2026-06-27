## 2026.06.23 LLY, adjust for StarDist feature extraction, not only for Ground truth
##                 there are six scripts for UNI feature extraction, now only one script is needed
##                 demo_UNI_feature_extraction_batch.sh

#!/usr/bin/env bash
## UNI feature extraction — GT or StarDist coords, Complete or Incomplete cases
#
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
#
# Usage:
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh [COORD] [CASES]
#
#   COORD:  gt (default) | stardist
#   CASES:  complete (default) | incomplete
#
# Replaces the former 6-script setup:
#   demo_Complete_GT_feature_extraction.sh          ->  gt complete
#   demo_Complete_StarDist_feature_extraction.sh    ->  stardist complete
#   demo_Incomplete_GT_feature_extraction.sh        ->  gt incomplete
#   demo_Incomplete_StarDist_feature_extraction.sh  ->  stardist incomplete
#
# Examples:
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt complete            # Complete + GT (25 samples)
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt incomplete          # Incomplete + GT (20 samples)
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist complete      # Complete + StarDist (25 samples)
#   bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist incomplete    # Incomplete + StarDist (20 samples)
#   SAMPLE=VUHD113 bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist complete    # single sample (Complete + StarDist)
#
# Env overrides: SAMPLE, CASES_ROOT, COORD_SOURCE, CASES_SET, CLEAN_UNI_OUTPUT
#
# GT input:     {SAMPLE}/{SAMPLE}_cells_partitioned_by_annotation_sample_match_with_pixel.csv
# StarDist input (copy_stardist_to_cases.py):
#               {SAMPLE}/{SAMPLE}_Float_prob0.01_nms_0.3.csv
# HE image:     {SAMPLE}/{SAMPLE}-HE.tif
#
# Output:
#   gt       -> .../{SAMPLE}_project_all_UNI/ImgEmbeddings_all/
#   stardist -> .../{SAMPLE}_project_all_UNI/ImgEmbeddings_all_stardist/
#
# When CASES is passed on the command line, CLEAN_UNI_OUTPUT defaults to 1
# (same as the old Complete/Incomplete wrapper scripts).

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Collaborate/esccAI"
PYTHON="/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python"
DATA="${REPO}/data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data"

PATCH_SIZE=16
SCALE=0.425
METHOD=UNI

COORD_SOURCE="${COORD_SOURCE:-gt}"
CASES_SET="${CASES_SET:-complete}"
CASES_FROM_ARG=0

if [[ $# -ge 1 && "${1}" =~ ^(gt|stardist)$ ]]; then
  COORD_SOURCE="${1}"
  shift
fi
if [[ $# -ge 1 && "${1}" =~ ^(complete|incomplete)$ ]]; then
  CASES_SET="${1}"
  CASES_FROM_ARG=1
  shift
fi
if [[ $# -ge 1 ]]; then
  echo "ERROR: unknown argument(s): $*" >&2
  echo "Usage: $0 [gt|stardist] [complete|incomplete]" >&2
  exit 1
fi

case "${COORD_SOURCE}" in
  gt)
    POSITION_SUFFIX="_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
    OUT_SUBDIR="ImgEmbeddings_all"
    ;;
  stardist)
    POSITION_SUFFIX="_Float_prob0.01_nms_0.3.csv"
    OUT_SUBDIR="ImgEmbeddings_all_stardist"
    ;;
  *)
    echo "ERROR: COORD_SOURCE must be 'gt' or 'stardist', got: ${COORD_SOURCE}" >&2
    exit 1
    ;;
esac

case "${CASES_SET}" in
  complete)
    CASES_ROOT="${CASES_ROOT:-${DATA}/Complete_Cases}"
    ;;
  incomplete)
    CASES_ROOT="${CASES_ROOT:-${DATA}/Incomplete_Cases}"
    ;;
  *)
    echo "ERROR: CASES_SET must be 'complete' or 'incomplete', got: ${CASES_SET}" >&2
    exit 1
    ;;
esac

if [[ "${CASES_FROM_ARG}" -eq 1 ]]; then
  CLEAN_UNI_OUTPUT="${CLEAN_UNI_OUTPUT:-1}"
else
  CLEAN_UNI_OUTPUT="${CLEAN_UNI_OUTPUT:-0}"
fi

cd "${REPO}"

echo "=== UNI feature extraction ==="
echo "Coord source: ${COORD_SOURCE}"
echo "Cases set:    ${CASES_SET}"
echo "Cases root:   ${CASES_ROOT#${REPO}/}"
echo "Position CSV: *${POSITION_SUFFIX}"
echo "Output subdir: ${OUT_SUBDIR}"
echo "Clean output: ${CLEAN_UNI_OUTPUT}"
echo

shopt -s nullglob
for case_dir in "${CASES_ROOT}"/*/; do
  sample="$(basename "${case_dir}")"

  if [[ -n "${SAMPLE:-}" && "${sample}" != "${SAMPLE}" ]]; then
    continue
  fi

  he_tifs=("${case_dir}"/*-HE.tif)
  if (( ${#he_tifs[@]} == 0 )); then
    echo "SKIP ${sample}: no *-HE.tif"
    continue
  fi
  he_tif="${he_tifs[0]}"

  position_csv="${case_dir}/${sample}${POSITION_SUFFIX}"
  if [[ ! -f "${position_csv}" ]]; then
    echo "SKIP ${sample}: missing ${position_csv#${REPO}/}"
    continue
  fi

  uni_project="${case_dir}/${sample}_project_all_UNI"
  out_root="${uni_project}/${OUT_SUBDIR}"
  if [[ "${CLEAN_UNI_OUTPUT}" == "1" && -d "${out_root}" ]]; then
    echo "  Removing previous: ${out_root#${REPO}/}"
    rm -rf "${out_root}"
  fi
  mkdir -p "${out_root}"

  echo "--------------------------------------------------"
  echo "Sample: ${sample}"
  echo "  HE:       ${he_tif#${REPO}/}"
  echo "  position: ${position_csv#${REPO}/}"
  echo "  output:   ${out_root#${REPO}/}"

  "${PYTHON}" code/Image_feature_extraction.py \
    --dataset "${sample}" \
    --position_path "${position_csv#${REPO}/}" \
    --rawimage_path "${he_tif#${REPO}/}" \
    --scale_image True \
    --method "${METHOD}" \
    --patch_size "${PATCH_SIZE}" \
    --output_img "${out_root}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image" \
    --output_pth "${out_root}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}" \
    --logging "${out_root}/" \
    --scale "${SCALE}"
done

echo
echo "=== Done (${COORD_SOURCE}, ${CASES_SET}) ==="
