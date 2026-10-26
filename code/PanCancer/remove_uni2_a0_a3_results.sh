#!/usr/bin/env bash
# Remove the finished UNI2 A0–A3 training outputs from 2026-10-26 morning.
# The retrain writes elsewhere:
#   data/PanCancer/result_uni2_<arch>_paired_v2_seed42
# so this script does not touch those directories.
#
# Removed:
#   data/PanCancer/result_uni2_A0_legacy_nospatial
#   data/PanCancer/result_uni2_A1_adjacent_nospatial
#   data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean
#   data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate
#   code/log/pancancer_section_uni2_A{0,1,2,3}_*.log   (train logs only)
#
# Left in place:
#   result_uni2_A4_*                  finished and abandoned attention runs
#   pred_uni2_A3_bestmlp              predictions already written
#   context_cache, uni2_aligned       reusable caches, not training results
#
# The A2 and A3 best_mlp_gpu.pt files live inside the result directories above.
# External-validation and cohort notebooks still point at those files.
# After deletion, predict from the old weights until the paired_v2 retrain finishes.
#
# Uses /bin/rm. The shell alias rm -> trash-put cannot move these trees
# onto a full /nobackup2 volume.
#
# From the repo root. The first command only prints. The second deletes.
#   bash code/PanCancer/remove_uni2_a0_a3_results.sh
#   bash code/PanCancer/remove_uni2_a0_a3_results.sh delete
set -euo pipefail

mode="${1:-list}"
if [[ "$mode" != "list" && "$mode" != "delete" ]]; then
  echo "usage: bash code/PanCancer/remove_uni2_a0_a3_results.sh [list|delete]" >&2
  exit 2
fi

repo="$(cd "$(dirname "$0")/../.." && pwd)"
pan="$repo/data/PanCancer"
logdir="$repo/code/log"

results=(
  "$pan/result_uni2_A0_legacy_nospatial"
  "$pan/result_uni2_A1_adjacent_nospatial"
  "$pan/result_uni2_A2_adjacent_spatial_k8mean"
  "$pan/result_uni2_A3_adjacent_spatial_k8mean_gate"
)
logs=(
  "$logdir/pancancer_section_uni2_A0_legacy_nospatial.log"
  "$logdir/pancancer_section_uni2_A1_adjacent_nospatial.log"
  "$logdir/pancancer_section_uni2_A2_adjacent_spatial_k8mean.log"
  "$logdir/pancancer_section_uni2_A3_adjacent_spatial_k8mean_gate.log"
)

for d in "${results[@]}"; do
  base="$(basename "$d")"
  case "$base" in
    result_uni2_A0_legacy_nospatial|\
    result_uni2_A1_adjacent_nospatial|\
    result_uni2_A2_adjacent_spatial_k8mean|\
    result_uni2_A3_adjacent_spatial_k8mean_gate) ;;
    *)
      echo "refuse $d" >&2
      exit 1
      ;;
  esac
  case "$d" in
    "$pan/$base") ;;
    *)
      echo "refuse $d" >&2
      exit 1
      ;;
  esac
done

n=0
for path in "${results[@]}" "${logs[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "absent ${path#"$repo"/}"
    continue
  fi
  n=$((n + 1))
  if [[ "$mode" == "list" ]]; then
    printf '%s\n' "${path#"$repo"/}"
  else
    printf 'rm %s\n' "${path#"$repo"/}"
    /bin/rm -rf -- "$path"
  fi
done

echo "----"
echo "mode=$mode  present=$n"
if [[ "$mode" == "list" ]]; then
  echo "Nothing deleted. Run again with: delete"
fi
