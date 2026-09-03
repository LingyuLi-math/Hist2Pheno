#!/usr/bin/env bash
## CODEX GBM — UNI for P174511_Initial and/or P179161_Recurrent
#
#   bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh gt
#   bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh stardist
#   SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh gt

set -euo pipefail

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
SINGLE="${REPO}/code/CODEX_gbm/demo_GT_feature_extraction_Single.sh"
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

SAMPLES="${SAMPLE:-P174511_Initial P179161_Recurrent}"
cd "${REPO}"
for s in ${SAMPLES}; do
  echo
  SAMPLE="${s}" bash "${SINGLE}" "${COORD}"
done
