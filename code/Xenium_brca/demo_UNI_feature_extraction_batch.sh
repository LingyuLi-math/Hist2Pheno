#!/usr/bin/env bash
## Xenium BRCA — UNI for rep1 and/or rep2
#
#   bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh gt
#   bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh stardist
#   SAMPLE=rep1 bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh gt

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
SINGLE="${REPO}/code/Xenium_brca/demo_GT_feature_extraction_Single.sh"
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

SAMPLES="${SAMPLE:-rep1 rep2}"
cd "${REPO}"
for s in ${SAMPLES}; do
  echo
  SAMPLE="${s}" bash "${SINGLE}" "${COORD}"
done
