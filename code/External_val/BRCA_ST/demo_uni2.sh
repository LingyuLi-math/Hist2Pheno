#!/usr/bin/env bash
## 2026.10.26 history: this script scored the morning UNI2 A3 checkpoint
## result_uni2_A3_adjacent_spatial_k8mean_gate/.../best_mlp_gpu.pt
## into pred_uni2_A3_bestmlp. That checkpoint was removed after the paired_v2 retrain.
##
## 2026.10.29: this command forwards to demo_uni2_A2.sh
## (fold 1, k=8 mean, no gate). A3 is demo_uni2_A3.sh (fold 0, k=8 mean with gate).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/demo_uni2_A2.sh" "$@"
