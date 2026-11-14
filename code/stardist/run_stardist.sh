#!/bin/bash
# StarDist launcher. TensorFlow 2.21 only sees the GPU when LD_LIBRARY_PATH
# points at the pip nvidia libs inside the venv.
#
#   export STARDIST_ENV=/path/to/envs/stardist
#   ./run_stardist.sh --image HE.btf --section-dir P1_CRC --section P1 --out-root results
#
# The CascadeHD run used envs/stardist (TF 2.21.0, stardist 0.9.2). See pip-freeze.txt.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E="${STARDIST_ENV:-$ROOT/envs/stardist}"
if [[ ! -x "$E/bin/python" ]]; then
  echo "StarDist env not found: $E" >&2
  echo "Set STARDIST_ENV to a venv with tensorflow and stardist." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(ls -d "$E"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:)"
export TF_CPP_MIN_LOG_LEVEL=2
exec "$E/bin/python" "$ROOT/stardist/run_stardist.py" "$@"
