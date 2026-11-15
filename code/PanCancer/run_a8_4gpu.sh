#!/usr/bin/env bash
## A8 across four GPUs. Same schedule as run_uni2_4gpu.sh.
##   bash code/PanCancer/run_a8_4gpu.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_uni2_4gpu.sh" A8
