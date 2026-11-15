#!/usr/bin/env bash
## A9 (neighbour composition) across four GPUs.
##   bash code/PanCancer/run_a9_4gpu.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_uni2_4gpu.sh" A9
