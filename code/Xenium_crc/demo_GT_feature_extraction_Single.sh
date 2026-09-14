#!/usr/bin/env bash
## Xenium CRC — single-sample UNI feature extraction (P1CRC / P2CRC / P5CRC)
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#   SAMPLE=P2CRC bash code/Xenium_crc/demo_GT_feature_extraction_Single.sh gt
#   SAMPLE=P1CRC bash code/Xenium_crc/demo_GT_feature_extraction_Single.sh stardist
#
# Add-on HE OME ≈ 0.274 µm/px (same as Visium HD fullres).
# Target 0.5 µm/px so patch_size=16 → ~8 µm: scale ≈ 0.548.
#
# P1/P5 skip PNG dumps (--no_save_patch_images). P2 keeps sc_pth_16_16_image.
# StarDist is whole-WSI (~1M+ nuclei). Pin a GPU. P5 StarDist CSV may be absent.

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
if [[ -z "${PYTHON:-}" ]]; then
  for cand in \
    /nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python \
    /ssd2/users/lingyu/conda_envs/SeededNTM/bin/python
  do
    if [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi
PYTHON="${PYTHON:-$(command -v python || true)}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: SeededNTM python not found." >&2
  exit 1
fi

SAMPLE="${SAMPLE:-P2CRC}"
CRC="${REPO}/data/Xemium/CRC"
CASES="${CASES_ROOT:-${CRC}/Cases}"
PATCH_SIZE=16
SCALE="${SCALE:-0.548}"    # HE_UM_PER_PX≈0.2738
METHOD=UNI

COORD="${COORD:-gt}"
if [[ $# -ge 1 ]]; then
  if [[ "${1}" =~ ^(gt|stardist)$ ]]; then
    COORD="${1}"
    shift
  else
    echo "Usage: $0 [gt|stardist]" >&2
    exit 1
  fi
fi

case "${SAMPLE}" in
  P1CRC|P2CRC|P5CRC)
    FOLDER="${SAMPLE/CRC/_CRC}"
    PREFIX="Xenium_V1_Human_Colon_Cancer_${FOLDER}_Add_on_FFPE"
    HE_TIF="${CRC}/HE_images/${PREFIX}_he_image.ome.tif"
    STARDIST_KEY="${PREFIX}_he_image.ome"
    ;;
  *) echo "ERROR: SAMPLE must be P1CRC, P2CRC, or P5CRC (got ${SAMPLE})" >&2; exit 1 ;;
esac

SAMPLE_DIR="${CASES}/${SAMPLE}"
if [[ "${COORD}" == "gt" ]]; then
  POSITION="${SAMPLE_DIR}/${SAMPLE}_cells_with_pixel.csv"
  OUT_SUB="ImgEmbeddings_all"
else
  POSITION="${CRC}/StarDist_Segment/${STARDIST_KEY}/${STARDIST_KEY}_Float_prob0.01_nms_0.3.csv"
  OUT_SUB="ImgEmbeddings_all_stardist"
fi

if [[ ! -f "${HE_TIF}" ]]; then
  echo "ERROR: HE image not found: ${HE_TIF}" >&2
  exit 1
fi
if [[ ! -f "${POSITION}" ]]; then
  echo "SKIP: position CSV not found for ${SAMPLE} ${COORD}: ${POSITION}" >&2
  if [[ "${COORD}" == "stardist" ]]; then
    echo "StarDist CSV missing (P5 may not be segmented yet). Continuing." >&2
    exit 0
  fi
  echo "Run: python code/Xenium_crc/match_xenium_cells_with_pixel.py --sample ${SAMPLE}" >&2
  exit 1
fi

# P2 already ran with PNG dumps; P1/P5 skip sc_pth_16_16_image (override with SAVE_PATCH_IMAGES=0|1).
if [[ -z "${SAVE_PATCH_IMAGES:-}" ]]; then
  if [[ "${SAMPLE}" == "P2CRC" ]]; then
    SAVE_PATCH_IMAGES=1
  else
    SAVE_PATCH_IMAGES=0
  fi
fi

UNI_ROOT="${SAMPLE_DIR}/project_all_UNI/${OUT_SUB}"
mkdir -p "${UNI_ROOT}/sc_pth_16_16"
EXTRA_FLAGS=()
if [[ "${SAVE_PATCH_IMAGES}" == "1" ]]; then
  mkdir -p "${UNI_ROOT}/sc_pth_16_16_image"
else
  EXTRA_FLAGS+=(--no_save_patch_images)
fi

cd "${REPO}"
echo "=== Xenium CRC UNI (${COORD}) sample=${SAMPLE} ==="
echo "HE:       ${HE_TIF}"
echo "Position: ${POSITION}"
echo "Output:   ${UNI_ROOT}"
echo "SCALE:    ${SCALE}"
echo "PNG dumps: ${SAVE_PATCH_IMAGES}"

"${PYTHON}" code/Image_feature_extraction.py \
  --dataset "sc_${SAMPLE}" \
  --position_path "${POSITION}" \
  --rawimage_path "${HE_TIF}" \
  --scale_image True \
  --method "${METHOD}" \
  --patch_size "${PATCH_SIZE}" \
  --output_img "${UNI_ROOT}/sc_pth_16_16_image" \
  --output_pth "${UNI_ROOT}/sc_pth_16_16" \
  --logging "${UNI_ROOT}/" \
  --scale "${SCALE}" \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
