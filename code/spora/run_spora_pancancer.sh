#!/usr/bin/env bash
# SPORA -> pan-cancer UNI2 A2.
#
# schurch2020coordinated is extra CRC sections. hoefflin2026spatial is extra
# GBM sections. Both stay inside the seven cohort names. This script writes
# data/PanCancer/result_uni2_A2_spora9_paired_v2_seed42/. A8 reads the same
# enlarged CRC and GBM through section_index.csv.
#
# GT centroids come from the supplied 1 µm instance masks (prepare, CPU).
# StarDist is a separate all-nuclei pass and is the step that uses 4 GPUs
# at once (segment). UNI2 tokens for the GT cells also shard across those
# 4 GPUs (uni2). Do not run segment and uni2 at the same time: each one
# already occupies every card.
#
#   cd /home/lingyu/ssd2/Python/Hist2Pheno
#   bash code/spora/run_spora_pancancer.sh prepare
#   bash code/spora/run_spora_pancancer.sh refresh-schurch
#   bash code/spora/run_spora_pancancer.sh segment
#   bash code/spora/run_spora_pancancer.sh uni2
#   CUDA_DEVICE=0 bash code/spora/run_spora_pancancer.sh train
#
# Smoke, two epochs, CRC + GBM (Xenium/WangLab plus the SPORA sections), 2k cells/section:
#   CUDA_DEVICE=0 bash code/spora/run_spora_pancancer.sh smoke
#
# GPUS="0 1 2 3" is the default shard. One tissue per card, round-robin.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PREP="${REPO}/data/CODEX/spora/prepared"
OUT="${REPO}/data/PanCancer"
CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
GPUS=(${GPUS:-0 1 2 3})
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BASE_DATASETS=(Lung BRCA CRC HCC PDAC GIST GBM)

cd "${REPO}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u
hash -r
PYTHON="$(command -v python)"

usage() {
  cat <<EOF
Usage:
  bash code/spora/run_spora_pancancer.sh prepare
  bash code/spora/run_spora_pancancer.sh refresh-schurch   # Schurch CD8 + plasma, then UNI2
  bash code/spora/run_spora_pancancer.sh segment     # StarDist, GPUs: ${GPUS[*]}
  bash code/spora/run_spora_pancancer.sh uni2        # GT UNI2 tokens, GPUs: ${GPUS[*]}
  CUDA_DEVICE=${CUDA_DEVICE} bash code/spora/run_spora_pancancer.sh train
  CUDA_DEVICE=${CUDA_DEVICE} bash code/spora/run_spora_pancancer.sh smoke

prepare is CPU. segment and uni2 each fill every listed GPU.
train is the README UNI2 A2 recipe. CRC includes the Schurch TMA and GBM includes Hoefflin.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_prepare() {
  "${PYTHON}" -u code/spora/prepare_spora_sections.py --workers "${PREPARE_WORKERS:-8}"
}

list_tifs() {
  local root="${1:-${PREP}}"
  [[ -d "${root}" ]] || die "missing ${root}. Run prepare first."
  find "${root}" -name '*_he.tif' | sort
}

shard_lists() {
  local kind="$1"
  local root="${2:-${PREP}}"
  local dir="${PREP}/.shards/${kind}"
  rm -rf "${dir}"
  mkdir -p "${dir}"
  local i=0 gpu n=${#GPUS[@]}
  local tif
  while IFS= read -r tif; do
    [[ -n "${tif}" ]] || continue
    gpu="${GPUS[$((i % n))]}"
    echo "${tif}" >> "${dir}/${gpu}.txt"
    i=$((i + 1))
  done < <(list_tifs "${root}")
  echo "${i}" > "${dir}/count"
  echo "${dir}"
}

wait_pids() {
  local fail=0 pid
  for pid in "$@"; do
    wait "${pid}" || fail=1
  done
  [[ "${fail}" == 0 ]] || die "one GPU worker failed"
}

cmd_segment() {
  local dir
  dir="$(shard_lists segment)"
  local n
  n="$(cat "${dir}/count")"
  echo "StarDist ${n} tissues on GPUs: ${GPUS[*]}"
  local pids=() gpu
  for gpu in "${GPUS[@]}"; do
    [[ -s "${dir}/${gpu}.txt" ]] || continue
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u code/spora/spora_stardist_worker.py \
      --list "${dir}/${gpu}.txt" &
    pids+=("$!")
  done
  if ((${#pids[@]})); then
    wait_pids "${pids[@]}"
  fi
}

cmd_refresh_schurch() {
  local ds="schurch2020coordinated"
  local root="${PREP}/${ds}"
  echo "force-prepare ${ds}: rewrite tables so CD8+ T cells and plasma cells are kept"
  "${PYTHON}" -u code/spora/prepare_spora_sections.py \
    --datasets "${ds}" --force --workers "${PREPARE_WORKERS:-8}"
  "${PYTHON}" - "${root}" << 'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
files = sorted(root.glob("*/*_cells_with_pixel.csv"))
n = n_cd8 = n_plasma = 0
for path in files:
    labels = pd.read_csv(path, usecols=["final_CT"])["final_CT"]
    n += len(labels)
    n_cd8 += int((labels == "CD8+ T cells").sum())
    n_plasma += int((labels == "plasma cells").sum())
print(f"schurch files={len(files)} cells={n} CD8+ T cells={n_cd8} plasma cells={n_plasma}", flush=True)
if len(files) != 140 or n_cd8 == 0 or n_plasma == 0:
    raise SystemExit("Schurch export is still missing CD8+ T cells or plasma cells")
PY
  local removed=0 emb
  while IFS= read -r emb; do
    [[ -n "${emb}" ]] || continue
    rm -rf "${emb}"
    removed=$((removed + 1))
  done < <(find "${root}" -mindepth 2 -maxdepth 2 -type d -name '*_project_all_UNI2' | sort)
  echo "removed ${removed} old Schurch UNI2 stores"
  local dir n pids=() gpu
  dir="$(shard_lists schurch-uni2 "${root}")"
  n="$(cat "${dir}/count")"
  echo "UNI2 Schurch ${n} tissues on GPUs: ${GPUS[*]}"
  for gpu in "${GPUS[@]}"; do
    [[ -s "${dir}/${gpu}.txt" ]] || continue
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u code/spora/spora_uni2_worker.py \
      --list "${dir}/${gpu}.txt" &
    pids+=("$!")
  done
  if ((${#pids[@]})); then
    wait_pids "${pids[@]}"
  fi
  echo "Schurch UNI2 refreshed. Hoefflin was not rewritten. Start A8/A9 after this, not before."
}

cmd_uni2() {
  local dir
  dir="$(shard_lists uni2)"
  local n
  n="$(cat "${dir}/count")"
  echo "UNI2 GT ${n} tissues on GPUs: ${GPUS[*]}"
  local pids=() gpu
  for gpu in "${GPUS[@]}"; do
    [[ -s "${dir}/${gpu}.txt" ]] || continue
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u code/spora/spora_uni2_worker.py \
      --list "${dir}/${gpu}.txt" &
    pids+=("$!")
  done
  if ((${#pids[@]})); then
    wait_pids "${pids[@]}"
  fi
}

run_train() {
  local save_result="$1"
  local ablation="$2"
  shift 2
  mkdir -p "${REPO}/code/log" "${OUT}/scaled_cache" "${OUT}/context_cache"
  export HIST2PHENO_ARRAY_CACHE="${OUT}/scaled_cache"
  export HIST2PHENO_CONTEXT_CACHE="${OUT}/context_cache"
  local log="${REPO}/code/log/${ablation}.log"
  echo "log ${log}"
  "${PYTHON}" -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
    --cuda-device "${CUDA_DEVICE}" \
    --embedder uni2 \
    --seed "${TRAIN_SEED:-42}" --split-seed "${SPLIT_SEED:-42}" \
    --group-splitter cross_dataset_section \
    --inner-val-fraction 0.2 \
    --class-weight-mode balanced \
    --hierarchy adjacent --legacy-l2-l1-weight 2 \
    --l2-l12-nll 0.5 --l12-l1-nll 0.5 --l2-l12-js 0.1 --l12-l1-js 0.1 \
    --use-spatial-context --spatial-k 8 --spatial-mode mean \
    --context-cache-dir "${OUT}/context_cache" \
    --steps prepare train he_validate \
    --save-result "${save_result}" \
    --ablation-tag "${ablation}" \
    "$@" \
    2>&1 | tee "${log}"
}

cmd_train() {
  run_train \
    "result_uni2_A2_spora9_paired_v2_seed${TRAIN_SEED:-42}" \
    "A2_adjacent_spatial_k8mean_spora9" \
    --datasets "${BASE_DATASETS[@]}"
}

cmd_smoke() {
  run_train \
    "result_uni2_smoke_spora_paired_v2_seed${TRAIN_SEED:-42}" \
    "smoke_spora" \
    --datasets CRC GBM \
    --max-cells-per-section 2000 \
    --cv-k 2 --max-epochs 2 --patience 2
}

case "${1:-}" in
  -h|--help|help|"") usage ;;
  prepare) cmd_prepare ;;
  refresh-schurch) cmd_refresh_schurch ;;
  segment|stardist) cmd_segment ;;
  uni2) cmd_uni2 ;;
  train) cmd_train ;;
  smoke) cmd_smoke ;;
  *) die "unknown step: ${1}" ;;
esac
