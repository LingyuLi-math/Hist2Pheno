#!/usr/bin/env bash
# 2026.10.29, default Virchow2 A2 / 默认 A2，独立权重。
## CUDA_DEVICE=1 bash code/External_val/NPC/demo_virchow2.sh all
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/../demo_virchow2.sh" NPC "$@"
