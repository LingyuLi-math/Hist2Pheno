#!/usr/bin/env bash
## 20260706 CODEX HCC s4769 — single-sample UNI feature extraction

# cd /home/lingyu/ssd2/Python/Hist2Pheno

## Usage:
#   bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh [gt|stardist]
#
#   COORD:  gt (default) | stardist
#
# Env overrides: COORD, ACQ_ID, HE_KEY, STARDIST_ROOT
#
# GT (CODEX cell_data.csv):
#   position: {S4769}/{ACQ_ID}/{ACQ_ID}.cell_data.csv  (X,Y = aligned HE full-res px)
#   output:   .../project_all_UNI/ImgEmbeddings_all/
#
# StarDist (one CSV per HE_KEY):
#   position: {STARDIST_ROOT}/{HE_KEY}/{HE_KEY}_Float_prob0.01_nms_0.3.csv
#   output:   .../project_all_UNI/ImgEmbeddings_all_stardist/
#
# HE: ~0.5 um/px → scale_image=False, patch_size=16 -> ~8 um (not Xenium 0.425)
#
# Examples:
#   bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh
#   bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh gt
#   bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh stardist
#   COORD=stardist HE_KEY=awy-98938_aligned_8ded610e bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh

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
ACQ_ID="${ACQ_ID:-FinalLiv-27_c001_v001_r001_reg001}"
HE_KEY="${HE_KEY:-awy-98938_aligned_0d535a74}"
S4769="${REPO}/data/CODEX/HCC/Michael_data_transfer/s4769"
STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment}"
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

HE_TIF="${S4769}/HE/${HE_KEY}/figures/${HE_KEY}.tif"
if [[ ! -f "${HE_TIF}" ]]; then
  echo "ERROR: HE image not found: ${HE_TIF}" >&2
  exit 1
fi

cd "${REPO}"

case "${COORD}" in
  gt)
    POSITION_CSV="${S4769}/${ACQ_ID}/${ACQ_ID}.cell_data.csv"
    DATASET="sc_${ACQ_ID}"
    OUT_ROOT="${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all"
    ;;
  stardist)
    POSITION_CSV="${STARDIST_ROOT}/${HE_KEY}/${HE_KEY}_Float_prob0.01_nms_0.3.csv"
    DATASET="sc_${HE_KEY}"
    OUT_ROOT="${S4769}/HE/${HE_KEY}/project_all_UNI/ImgEmbeddings_all_stardist"
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

echo "=== CODEX HCC UNI feature extraction (single) ==="
echo "Coord source: ${COORD}"
echo "ACQ_ID:       ${ACQ_ID}"
echo "HE_KEY:       ${HE_KEY}"
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
