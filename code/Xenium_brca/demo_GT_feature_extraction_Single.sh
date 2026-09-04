#!/usr/bin/env bash
## Xenium BRCA — single-sample UNI feature extraction (rep1 / rep2)
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#   bash code/Xenium_brca/demo_GT_feature_extraction_Single.sh gt
#   bash code/Xenium_brca/demo_GT_feature_extraction_Single.sh stardist
#   SAMPLE=rep2 bash code/Xenium_brca/demo_GT_feature_extraction_Single.sh gt
#
# Working HE *.tif after Explorer alignment + OME→tif resize ≈ 0.42 µm/px.
# Target 0.5 µm/px so patch_size=16 → ~8 µm: scale ≈ 0.84.

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

SAMPLE="${SAMPLE:-rep1}"
BRCA="${REPO}/data/Xemium/BRCA"
CASES="${CASES_ROOT:-${BRCA}/Cases}"
PATCH_SIZE=16
SCALE="${SCALE:-0.84}"    # he_um_per_px≈0.42
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
  rep1) PREFIX="Xenium_FFPE_Human_Breast_Cancer_Rep1" ;;
  rep2) PREFIX="Xenium_FFPE_Human_Breast_Cancer_Rep2" ;;
  *) echo "ERROR: SAMPLE must be rep1 or rep2 (got ${SAMPLE})" >&2; exit 1 ;;
esac

HE_TIF="${BRCA}/human_breast_Xenium_${SAMPLE}/${PREFIX}_he_image.tif"
SAMPLE_DIR="${CASES}/${SAMPLE}"
if [[ "${COORD}" == "gt" ]]; then
  POSITION="${SAMPLE_DIR}/${SAMPLE}_cells_with_pixel.csv"
  OUT_SUB="ImgEmbeddings_all"
else
  STARDIST_KEY="${PREFIX}_he_image"
  POSITION="${BRCA}/StarDist_Segment/${STARDIST_KEY}/${STARDIST_KEY}_Float_prob0.01_nms_0.3.csv"
  OUT_SUB="ImgEmbeddings_all_stardist"
fi

if [[ ! -f "${HE_TIF}" ]]; then
  echo "ERROR: HE image not found: ${HE_TIF}" >&2
  exit 1
fi
if [[ ! -f "${POSITION}" ]]; then
  echo "ERROR: position CSV not found: ${POSITION}" >&2
  echo "Run: python code/Xenium_brca/match_xenium_cells_with_pixel.py --sample ${SAMPLE}" >&2
  exit 1
fi

UNI_ROOT="${SAMPLE_DIR}/project_all_UNI/${OUT_SUB}"
mkdir -p "${UNI_ROOT}/sc_pth_16_16" "${UNI_ROOT}/sc_pth_16_16_image"

cd "${REPO}"
echo "=== Xenium BRCA UNI (${COORD}) sample=${SAMPLE} ==="
echo "HE:       ${HE_TIF}"
echo "Position: ${POSITION}"
echo "Output:   ${UNI_ROOT}"

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
  --scale "${SCALE}"
