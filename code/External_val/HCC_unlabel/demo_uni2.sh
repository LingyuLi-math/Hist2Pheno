#!/usr/bin/env bash
## CODEX HCC unlabeled rest — UNI2 Hist2Pheno (no GT).
##
## 2026.10.26 history: this script scored the morning A3 checkpoint
## result_uni2_A3_adjacent_spatial_k8mean_gate/.../best_mlp_gpu.pt
## into data/PanCancer/pred_uni2_A3_bestmlp/unlabel/HCC_unlabel/.
## That checkpoint and pred directory were removed after the paired_v2 retrain.
##
## 2026.10.28: this command forwards to demo_uni2_A3.sh, which loads
## result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/.../best_mlp_gpu.pt
## (fold 0). A2 is a separate weight:
##   bash code/External_val/HCC_unlabel/demo_uni2_A2.sh predict

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/demo_uni2_A3.sh" "$@"
