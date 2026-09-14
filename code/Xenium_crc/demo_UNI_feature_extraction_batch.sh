#!/usr/bin/env bash
## Xenium CRC — UNI for P1CRC / P2CRC / P5CRC
#
#   bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
#   bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
#   SAMPLE=P2CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
#
# P1/P5 skip PNG dumps. Missing StarDist CSVs are skipped (exit 0 per sample).

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
SINGLE="${REPO}/code/Xenium_crc/demo_GT_feature_extraction_Single.sh"
chmod +x "${SINGLE}"

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

SAMPLES="${SAMPLE:-P1CRC P2CRC P5CRC}"
cd "${REPO}"
for s in ${SAMPLES}; do
  echo
  SAMPLE="${s}" bash "${SINGLE}" "${COORD}"
done
