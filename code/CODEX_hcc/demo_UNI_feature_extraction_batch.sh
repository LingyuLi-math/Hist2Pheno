#!/usr/bin/env bash
## CODEX HCC s4769 — batch UNI feature extraction (38 ALIGNED regions)
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#
# Usage:
#   bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh [gt|stardist]
#
#   COORD:  gt (default) | stardist
#
# Env overrides: COORD, ACQ_ID, HE_KEY, STARDIST_ROOT, CLEAN_UNI_OUTPUT, SKIP_IF_DONE
#
# GT:       38 rows from s4769_he_mapping_updated_Visium.xlsx (ALIGNED=='Y')
#           position: {S4769}/{ACQ_ID}/{ACQ_ID}.cell_data.csv
# StarDist: 38 unique MATCHED_HE (1:1 with regions)
#           position: {STARDIST_ROOT}/{HE_KEY}/{HE_KEY}_Float_prob0.01_nms_0.3.csv
#
# Output (under each HE_KEY):
#   gt       -> .../project_all_UNI/ImgEmbeddings_all/
#   stardist -> .../project_all_UNI/ImgEmbeddings_all_stardist/
#
# Examples:
#   bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh
#   bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh gt
#   bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh stardist
#   ACQ_ID=FinalLiv-27_c001_v001_r001_reg001 bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh gt
#   HE_KEY=awy-98938_aligned_0d535a74 bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh stardist
#   CLEAN_UNI_OUTPUT=1 bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh gt

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
SINGLE_SCRIPT="${REPO}/code/CODEX_hcc/demo_GT_feature_extraction_Single.sh"
PYTHON="/ssd2/users/lingyu/conda_envs/SeededNTM/bin/python"
S4769="${REPO}/data/CODEX/HCC/Michael_data_transfer/s4769"
MAPPING_XLSX="${S4769}/HE/s4769_he_mapping_updated_Visium.xlsx"
STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment}"

COORD="${COORD:-gt}"
COORD_FROM_ARG=0
if [[ $# -ge 1 ]]; then
  if [[ "${1}" =~ ^(gt|stardist)$ ]]; then
    COORD="${1}"
    COORD_FROM_ARG=1
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

if [[ "${COORD_FROM_ARG}" -eq 1 ]]; then
  CLEAN_UNI_OUTPUT="${CLEAN_UNI_OUTPUT:-1}"
else
  CLEAN_UNI_OUTPUT="${CLEAN_UNI_OUTPUT:-0}"
fi
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

if [[ ! -f "${MAPPING_XLSX}" ]]; then
  echo "ERROR: mapping xlsx not found: ${MAPPING_XLSX}" >&2
  exit 1
fi
if [[ ! -x "${SINGLE_SCRIPT}" ]]; then
  chmod +x "${SINGLE_SCRIPT}"
fi

cd "${REPO}"

echo "=== CODEX HCC UNI feature extraction (batch) ==="
echo "Coord source:    ${COORD}"
echo "Mapping:         ${MAPPING_XLSX#${REPO}/}"
echo "StarDist root:   ${STARDIST_ROOT}"
echo "Clean output:    ${CLEAN_UNI_OUTPUT}"
echo "Skip if done:    ${SKIP_IF_DONE}"
if [[ -n "${ACQ_ID:-}" ]]; then echo "Filter ACQ_ID:   ${ACQ_ID}"; fi
if [[ -n "${HE_KEY:-}" ]]; then echo "Filter HE_KEY:   ${HE_KEY}"; fi
echo

export MAPPING_XLSX
mapfile -t ALIGNMENT_ROWS < <("${PYTHON}" - <<'PY'
import pandas as pd
import os

xlsx = os.environ["MAPPING_XLSX"]
df = pd.read_excel(xlsx)
df = df[df["ALIGNED"].astype(str).str.upper() == "Y"].reset_index(drop=True)
for _, row in df.iterrows():
    acq = str(row["CODEX_ACQUISITION_ID"]).strip()
    he = str(row["MATCHED_HE"]).strip()
    print(f"{acq}\t{he}")
PY
)

TOTAL="${#ALIGNMENT_ROWS[@]}"
if (( TOTAL == 0 )); then
  echo "ERROR: no ALIGNED=='Y' rows in mapping table" >&2
  exit 1
fi

OK=0
SKIP=0
FAIL=0
IDX=0

for row in "${ALIGNMENT_ROWS[@]}"; do
  ACQ="${row%%$'\t'*}"
  HE="${row#*$'\t'}"

  if [[ -n "${ACQ_ID:-}" && "${ACQ}" != "${ACQ_ID}" ]]; then
    continue
  fi
  if [[ -n "${HE_KEY:-}" && "${HE}" != "${HE_KEY}" ]]; then
    continue
  fi

  (( ++IDX )) || true

  case "${COORD}" in
    gt)
      OUT_ROOT="${S4769}/HE/${HE}/project_all_UNI/ImgEmbeddings_all"
      PTH_DIR="${OUT_ROOT}/sc_pth_16_16"
      LABEL="${ACQ}"
      ;;
    stardist)
      OUT_ROOT="${S4769}/HE/${HE}/project_all_UNI/ImgEmbeddings_all_stardist"
      PTH_DIR="${OUT_ROOT}/sc_pth_16_16"
      LABEL="${HE}"
      ;;
  esac

  if [[ "${SKIP_IF_DONE}" == "1" && -d "${PTH_DIR}" ]]; then
    pth_count="$(find "${PTH_DIR}" -maxdepth 1 -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${pth_count}" -gt 0 ]]; then
      echo "--------------------------------------------------"
      echo "[${IDX}/${TOTAL}] SKIP (done, ${pth_count} .pth): ${LABEL}"
      (( ++SKIP )) || true
      continue
    fi
  fi

  if [[ "${CLEAN_UNI_OUTPUT}" == "1" && -d "${OUT_ROOT}" ]]; then
    echo "  Removing previous: ${OUT_ROOT#${REPO}/}"
    rm -rf "${OUT_ROOT}"
  fi

  echo "--------------------------------------------------"
  echo "[${IDX}/${TOTAL}] ${COORD}: ACQ_ID=${ACQ} HE_KEY=${HE}"

  if ! ACQ_ID="${ACQ}" HE_KEY="${HE}" STARDIST_ROOT="${STARDIST_ROOT}" \
      bash "${SINGLE_SCRIPT}" "${COORD}"; then
    echo "FAILED: ${LABEL}" >&2
    (( ++FAIL )) || true
    continue
  fi
  (( ++OK )) || true
done

echo
echo "=== Done (${COORD}) ==="
echo "Succeeded: ${OK}  Skipped: ${SKIP}  Failed: ${FAIL}  Total aligned: ${TOTAL}"
if (( FAIL > 0 )); then
  exit 1
fi
