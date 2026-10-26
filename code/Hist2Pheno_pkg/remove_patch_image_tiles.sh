#!/usr/bin/env bash
# Delete PNG crop caches under data/. Keeps the .pth directories.
#
# Removed names, and only these:
#   sc_pth_16_16_image
#   sc_pth_14_14_image
# Left in place:
#   sc_pth_16_16
#   sc_pth_14_14
#
# Uses /bin/rm. The shell alias rm -> trash-put cannot move these trees
# onto a full /nobackup2 volume.
#
# From the repo root:
#   bash code/Hist2Pheno_pkg/remove_patch_image_tiles.sh
#   bash code/Hist2Pheno_pkg/remove_patch_image_tiles.sh delete
set -euo pipefail

mode="${1:-list}"
if [[ "$mode" != "list" && "$mode" != "delete" ]]; then
  echo "usage: bash code/Hist2Pheno_pkg/remove_patch_image_tiles.sh [list|delete]" >&2
  exit 2
fi

repo="$(cd "$(dirname "$0")/../.." && pwd)"
data="$repo/data"
if [[ ! -d "$data" ]]; then
  echo "data directory not found: $data" >&2
  exit 1
fi

mapfile -d '' dirs < <(
  find "$data" -type d -name 'sc_pth_*' -prune -print0
)

kept=0
n=0
for d in "${dirs[@]}"; do
  [[ -n "$d" ]] || continue
  base="$(basename "$d")"
  case "$base" in
    sc_pth_16_16_image|sc_pth_14_14_image) ;;
    *)
      kept=$((kept + 1))
      continue
      ;;
  esac
  case "$d" in
    */ImgEmbeddings*/"$base") ;;
    *)
      echo "refuse $d" >&2
      exit 1
      ;;
  esac
  n=$((n + 1))
  if [[ "$mode" == "list" ]]; then
    printf '%s\n' "${d#"$repo"/}"
  else
    printf 'rm %s\n' "${d#"$repo"/}"
    /bin/rm -rf -- "$d"
  fi
done

echo "----"
echo "mode=$mode  image_dirs=$n  pth_dirs_kept=$kept"
if [[ "$mode" == "list" ]]; then
  echo "Nothing deleted. Run again with: delete"
fi
