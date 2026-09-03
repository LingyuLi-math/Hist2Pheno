#!/usr/bin/env bash
## CODEX GBM (WangLab) — single-sample UNI feature extraction
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#   SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_GT_feature_extraction_Single.sh gt
#   SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_GT_feature_extraction_Single.sh stardist
#   SAMPLE=P179161_Recurrent bash code/CODEX_gbm/demo_GT_feature_extraction_Single.sh gt
#
# Microscope HE has no reliable µm/px tag; use HCC-style scale_image=False,
# patch_size=16 (do NOT use CytAssist 0.883 µm/px on the microscope TIFF).

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

SAMPLE="${SAMPLE:-P174511_Initial}"
GBM="${REPO}/data/CODEX/GBM"
CASES="${CASES_ROOT:-${GBM}/Cases}"
WANGLAB="${GBM}/WangLab"
STARDIST_ROOT="${STARDIST_ROOT:-${WANGLAB}/StarDist_Segment}"
PATCH_SIZE=16
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
  P174511_Initial) STARDIST_KEY="174511-2-3"; HE_TIF="${WANGLAB}/4_Images/Microscope_Image/174511-2-3.tif" ;;
  P179161_Recurrent) STARDIST_KEY="179161-4-3"; HE_TIF="${WANGLAB}/4_Images/Microscope_Image/179161-4-3.tif" ;;
  *) echo "ERROR: SAMPLE must be P174511_Initial or P179161_Recurrent (got ${SAMPLE})" >&2; exit 1 ;;
esac

SAMPLE_DIR="${CASES}/${SAMPLE}"
if [[ "${COORD}" == "gt" ]]; then
  POSITION="${SAMPLE_DIR}/${SAMPLE}_cells_with_pixel.csv"
  OUT_SUB="ImgEmbeddings_all"
else
  POSITION="${STARDIST_ROOT}/${STARDIST_KEY}/${STARDIST_KEY}_Float_prob0.01_nms_0.3.csv"
  OUT_SUB="ImgEmbeddings_all_stardist"
fi

if [[ ! -f "${HE_TIF}" ]]; then
  echo "ERROR: HE image not found: ${HE_TIF}" >&2
  exit 1
fi
if [[ ! -f "${POSITION}" ]]; then
  echo "ERROR: position CSV not found: ${POSITION}" >&2
  echo "Run: python code/CODEX_gbm/match_codex_cells_with_pixel.py --sample ${SAMPLE}" >&2
  exit 1
fi

UNI_ROOT="${SAMPLE_DIR}/project_all_UNI/${OUT_SUB}"
mkdir -p "${UNI_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}" \
         "${UNI_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image"

cd "${REPO}"
echo "=== CODEX GBM UNI (${COORD}) sample=${SAMPLE} ==="
echo "HE:       ${HE_TIF}"
echo "Position: ${POSITION}"
echo "Output:   ${UNI_ROOT}"

"${PYTHON}" code/Image_feature_extraction.py \
  --dataset "sc_${SAMPLE}" \
  --position_path "${POSITION}" \
  --rawimage_path "${HE_TIF}" \
  --scale_image False \
  --method "${METHOD}" \
  --patch_size "${PATCH_SIZE}" \
  --output_img "${UNI_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image" \
  --output_pth "${UNI_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}" \
  --logging "${UNI_ROOT}/"

echo
echo "=== Done (${COORD}) ==="
