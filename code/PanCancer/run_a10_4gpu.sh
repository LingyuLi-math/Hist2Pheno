#!/usr/bin/env bash
## A10 (neighbour composition in µm) across four GPUs.
## Writes data/PanCancer/mpp_map.csv before training.
##   bash code/PanCancer/run_a10_4gpu.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_uni2_4gpu.sh" A10
