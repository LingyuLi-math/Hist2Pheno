#!/usr/bin/env bash
## 2026.08.20 CODEX GIST s1167 — single-core UNI feature extraction

# cd /home/lingyu/ssd2/Python/Hist2Pheno
#
# Usage:
#   bash code/CODEX_gist/demo_GT_feature_extraction_Single.sh [gt|stardist]
#
# Env overrides: COORD, ACQ_ID, HE_KEY, STARDIST_ROOT, PYTHON
#
# GT:
#   position: {S1167}/{ACQ_ID}/{ACQ_ID}.cell_data.csv
#   output:   {S1167}/{ACQ_ID}/project_all_UNI/ImgEmbeddings_all/
#
# StarDist:
#   position: {STARDIST_ROOT}/{ACQ_ID}/{ACQ_ID}_Float_prob0.01_nms_0.3.csv
#   output:   {S1167}/{ACQ_ID}/project_all_UNI/ImgEmbeddings_all_stardist/
#
# Examples:
#   ACQ_ID=Charvill-94_c013_v001_r001_reg002 bash code/CODEX_gist/demo_GT_feature_extraction_Single.sh
#   ACQ_ID=Charvill-94_c013_v001_r001_reg002 bash code/CODEX_gist/demo_GT_feature_extraction_Single.sh stardist

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
  echo "ERROR: SeededNTM python not found. Activate the env or set PYTHON=..." >&2
  exit 1
fi

ACQ_ID="${ACQ_ID:-${HE_KEY:-Charvill-94_c013_v001_r001_reg002}}"
HE_KEY="${HE_KEY:-${ACQ_ID}}"
S1167="${REPO}/data/CODEX/HCC/Michael_data_transfer/s1167"
STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment_gist/gist_result}"
PATCH_SIZE=16
METHOD=UNI

COORD="${COORD:-gt}"
if [[ $# -ge 1 ]]; then
  if [[ "${1}" =~ ^(gt|stardist)$ ]]; then
    COORD="${1}"
    shift
  else
    echo "ERROR: unknown argument: ${1}" >&2
    echo "Usage: $0 [gt|stardist]" >&2
    exit 1
  fi
fi
if [[ $# -ge 1 ]]; then
  echo "ERROR: unknown argument(s): $*" >&2
  echo "Usage: $0 [gt|stardist]" >&2
  exit 1
fi

HE_TIF="${S1167}/${ACQ_ID}/figures/${ACQ_ID}.tif"
if [[ ! -f "${HE_TIF}" ]]; then
  echo "ERROR: HE image not found: ${HE_TIF}" >&2
  exit 1
fi

cd "${REPO}"

case "${COORD}" in
  gt)
    POSITION_CSV="${S1167}/${ACQ_ID}/${ACQ_ID}.cell_data.csv"
    DATASET="sc_${ACQ_ID}"
    OUT_ROOT="${S1167}/${ACQ_ID}/project_all_UNI/ImgEmbeddings_all"
    ;;
  stardist)
    POSITION_CSV="${STARDIST_ROOT}/${ACQ_ID}/${ACQ_ID}_Float_prob0.01_nms_0.3.csv"
    DATASET="sc_${ACQ_ID}"
    OUT_ROOT="${S1167}/${ACQ_ID}/project_all_UNI/ImgEmbeddings_all_stardist"
    ;;
  *)
    echo "ERROR: COORD must be 'gt' or 'stardist', got: ${COORD}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${POSITION_CSV}" ]]; then
  echo "ERROR: position CSV not found: ${POSITION_CSV}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "=== CODEX GIST UNI feature extraction (single) ==="
echo "Python:       ${PYTHON}"
echo "Coord source: ${COORD}"
echo "ACQ_ID:       ${ACQ_ID}"
echo "HE:           ${HE_TIF}"
echo "Position:     ${POSITION_CSV}"
echo "Output:       ${OUT_ROOT}"
echo

"${PYTHON}" code/Image_feature_extraction.py \
  --dataset "${DATASET}" \
  --position_path "${POSITION_CSV}" \
  --rawimage_path "${HE_TIF}" \
  --scale_image False \
  --method "${METHOD}" \
  --patch_size "${PATCH_SIZE}" \
  --output_img "${OUT_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}_image" \
  --output_pth "${OUT_ROOT}/sc_pth_${PATCH_SIZE}_${PATCH_SIZE}" \
  --logging "${OUT_ROOT}/"

echo
echo "=== Done (${COORD}) ==="
