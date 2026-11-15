#!/usr/bin/env bash
## A11 (neighbour composition plus tissue compartment) across four GPUs.
##   bash code/PanCancer/run_a11_4gpu.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_uni2_4gpu.sh" A11
