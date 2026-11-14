#!/usr/bin/env bash
# Usage: bash run_embryo.sh [--check-only | --roi X Y WIDTH HEIGHT | other run_wsi.py options]
# Override IMAGE, OUT_DIR, STARDIST_ENV, TILE, OVERLAP, CUDA_VISIBLE_DEVICES as needed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMAGE="${IMAGE:-$PROJECT_ROOT/Embryo_data/tic-4644/data/HnE/CS17_HnE_SOB26-WEM-0-FFPE-1-S046.ome.tif}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/Embryo_output/embryo_output/stardist/S046}"
STARDIST_ENV="${STARDIST_ENV:-/software/conda/users/ll20/envs/hist2pheno}"
PYTHON="$STARDIST_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON. Set STARDIST_ENV to your StarDist environment." >&2
    exit 1
fi
# Preserve existing library paths; append pip NVIDIA libraries when present.
shopt -s nullglob
for lib_dir in "$STARDIST_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
exec "$PYTHON" -u "$SCRIPT_DIR/run_wsi.py" \
    --image "$IMAGE" --out-dir "$OUT_DIR" \
    --tile "${TILE:-2048}" --overlap "${OVERLAP:-256}" "$@"
