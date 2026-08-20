#!/usr/bin/env bash
## CODEX PDAC s1167 — batch UNI feature extraction
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#
# Usage:
#   bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh [gt|stardist]
#
# Default: 278 annotated Pancreas TMA cores.
# Incomplete_Cases (195, no cell-type CSV): INCOMPLETE=1 (stardist only)
#
# Env: COORD, ACQ_ID, INCOMPLETE, STARDIST_ROOT, CLEAN_UNI_OUTPUT, SKIP_IF_DONE, PYTHON
#
# Examples:
#   bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh gt
#   bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh stardist
#   INCOMPLETE=1 bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh stardist
#   ACQ_ID=Charvill-94_c001_v001_r001_reg001 bash code/CODEX_pdac/demo_UNI_feature_extraction_batch.sh stardist

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
SINGLE_SCRIPT="${REPO}/code/CODEX_pdac/demo_GT_feature_extraction_Single.sh"
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

S1167="${REPO}/data/CODEX/HCC/Michael_data_transfer/s1167"
STARDIST_ROOT="${STARDIST_ROOT:-${REPO}/data/CODEX/HCC/StarDist_Segment_pdac/pdac_result}"
INCOMPLETE="${INCOMPLETE:-0}"

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

if [[ "${INCOMPLETE}" == "1" && "${COORD}" != "stardist" ]]; then
  echo "ERROR: Incomplete_Cases have no CODEX GT coords; use COORD=stardist" >&2
  exit 1
fi

CLEAN_UNI_OUTPUT="${CLEAN_UNI_OUTPUT:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

if [[ ! -x "${SINGLE_SCRIPT}" ]]; then
  chmod +x "${SINGLE_SCRIPT}"
fi

cd "${REPO}"

echo "=== CODEX PDAC UNI feature extraction (batch) ==="
echo "Coord source:    ${COORD}"
echo "Incomplete:      ${INCOMPLETE}"
echo "StarDist root:   ${STARDIST_ROOT}"
echo "Skip if done:    ${SKIP_IF_DONE}"
if [[ -n "${ACQ_ID:-}" ]]; then echo "Filter ACQ_ID:   ${ACQ_ID}"; fi
echo

export REPO
mapfile -t ACQ_ROWS < <("${PYTHON}" - <<PY
import sys
from pathlib import Path
repo = Path("${REPO}")
sys.path.insert(0, str(repo / "code" / "CODEX_pdac"))
from match_codex_cells_with_pixel import (
    list_aligned_annotated_regions,
    list_incomplete_pdac_regions,
)
df = list_incomplete_pdac_regions() if "${INCOMPLETE}" == "1" else list_aligned_annotated_regions()
for aid in df["ACQUISITION_ID"].astype(str).str.strip():
    print(aid)
PY
)

TOTAL="${#ACQ_ROWS[@]}"
if (( TOTAL == 0 )); then
  echo "ERROR: no PDAC cores to process" >&2
  exit 1
fi

OK=0
SKIP=0
FAIL=0
IDX=0

for ACQ in "${ACQ_ROWS[@]}"; do
  if [[ -n "${ACQ_ID:-}" && "${ACQ}" != "${ACQ_ID}" ]]; then
    continue
  fi
  (( ++IDX )) || true

  case "${COORD}" in
    gt)
      OUT_ROOT="${S1167}/${ACQ}/project_all_UNI/ImgEmbeddings_all"
      ;;
    stardist)
      OUT_ROOT="${S1167}/${ACQ}/project_all_UNI/ImgEmbeddings_all_stardist"
      ;;
  esac
  PTH_DIR="${OUT_ROOT}/sc_pth_16_16"

  if [[ "${SKIP_IF_DONE}" == "1" && -d "${PTH_DIR}" ]]; then
    pth_count="$(find "${PTH_DIR}" -maxdepth 1 -name '*.pth' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${pth_count}" -gt 0 ]]; then
      echo "--------------------------------------------------"
      echo "[${IDX}/${TOTAL}] SKIP (done, ${pth_count} .pth): ${ACQ}"
      (( ++SKIP )) || true
      continue
    fi
  fi

  if [[ "${CLEAN_UNI_OUTPUT}" == "1" && -d "${OUT_ROOT}" ]]; then
    echo "  Removing previous: ${OUT_ROOT#${REPO}/}"
    rm -rf "${OUT_ROOT}"
  fi

  echo "--------------------------------------------------"
  echo "[${IDX}/${TOTAL}] ${COORD}: ACQ_ID=${ACQ}"

  if ! ACQ_ID="${ACQ}" STARDIST_ROOT="${STARDIST_ROOT}" \
      bash "${SINGLE_SCRIPT}" "${COORD}"; then
    echo "FAILED: ${ACQ}" >&2
    (( ++FAIL )) || true
    continue
  fi
  (( ++OK )) || true
done

echo
echo "=== Done (${COORD}) ==="
echo "Succeeded: ${OK}  Skipped: ${SKIP}  Failed: ${FAIL}  Listed: ${TOTAL}"
if (( FAIL > 0 )); then
  exit 1
fi
