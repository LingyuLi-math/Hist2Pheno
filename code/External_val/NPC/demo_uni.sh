#!/usr/bin/env bash
# 2026.10.29, default UNI A2 / 默认 A2，独立权重。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/../demo_uni.sh" NPC "$@"
