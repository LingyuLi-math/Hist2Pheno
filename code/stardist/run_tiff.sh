#!/bin/bash
# Detect nuclei in one H&E TIFF.
#
#   ./run_tiff.sh /path/to/HE.tiff
#   ./run_tiff.sh /path/to/HE.tiff /path/to/output_dir
#
# Uses the benchmark StarDist env unless STARDIST_ENV is set.
# Coordinates land in nuclei_coordinates.csv: cx_px = column, cy_px = row.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 HE.tiff [out_dir]" >&2
  exit 1
fi
IMAGE=$1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$IMAGE")"
NAME="${NAME%.*}"
OUT="${2:-$ROOT/stardist/results/$NAME}"
E="${STARDIST_ENV:-/nobackup2/shared/benchmark/01segmentation/envs/stardist}"
if [[ ! -x "$E/bin/python" ]]; then
  echo "StarDist env not found: $E" >&2
  echo "Set STARDIST_ENV to a venv with tensorflow and stardist." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(ls -d "$E"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:)"
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$OUT"
exec "$E/bin/python" "$ROOT/stardist/run_tiff.py" --image "$IMAGE" --out-dir "$OUT"
