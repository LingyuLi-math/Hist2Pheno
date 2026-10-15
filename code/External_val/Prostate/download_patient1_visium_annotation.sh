#!/usr/bin/env bash
## Download Mendeley svw96g68dv v5 → Patient 1 / Visium_with_annotation
##   https://data.mendeley.com/datasets/svw96g68dv/5
##
## Replaces data/Visium/Prostate/Patient_1/Visium_with_annotation
## (7 sections: H1_2 H1_4 H1_5 H2_1 H2_2 H2_5 V1_2, ~70 MB).
##
##   bash code/External_val/Prostate/download_patient1_visium_annotation.sh
##   bash code/External_val/Prostate/download_patient1_visium_annotation.sh --list
##   bash code/External_val/Prostate/download_patient1_visium_annotation.sh --force

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
python -u code/External_val/Prostate/download_patient1_visium_annotation.py "$@"
