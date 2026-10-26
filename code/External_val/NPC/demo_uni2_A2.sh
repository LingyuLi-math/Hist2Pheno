#!/usr/bin/env bash
## NPC external validation — UNI2 A2 (adjacent loss, spatial k=8 mean, no gate).
## Reuses project_all_UNI2 tokens. Does not rewrite A3 predictions.
##
##   CUDA_DEVICE=0 bash code/External_val/NPC/demo_uni2_A2.sh predict
##   SAMPLE=20210809-C-AH4199551 CUDA_DEVICE=0 bash code/External_val/NPC/demo_uni2_A2.sh predict
## Figures: code/External_val/NPC/Pred_visual_npc_uni2_A2.ipynb

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../log_run.sh"

REPO="/home/lingyu/ssd2/Python/Hist2Pheno"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export CUDA_DEVICE

if [[ -z "${PYTHON:-}" ]]; then
  for cand in \
    /nobackup2/users/lingyu/conda_envs/SeededNTM/bin/python \
    /ssd2/users/lingyu/conda_envs/SeededNTM/bin/python
  do
    if [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi
PYTHON="${PYTHON:-$(command -v python || true)}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: SeededNTM python not found. Activate the env or set PYTHON=..." >&2
  exit 1
fi

STEP="${1:-}"
SAMPLE_ARG=()
if [[ -n "${SAMPLE:-}" ]]; then
  SAMPLE_ARG=(--sample "${SAMPLE}")
fi
FORCE_ARG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARG=(--force)
fi

usage() {
  cat <<EOF
Usage: bash code/External_val/NPC/demo_uni2_A2.sh {dry-run|predict|probs|plot|all}

Checkpoint: data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean/.../best_mlp_gpu.pt
Embeddings (already extracted): data/Visium/NPC/project_all_UNI2/
Outputs: data/Visium/NPC/pred_uni2_A2_bestmlp/
A3 predictions in pred_uni2_A3_bestmlp/ are left unchanged.
Log: code/log/external_val_npc_uni2_A2.log
EOF
}

if [[ -z "${STEP}" ]]; then
  usage
  exit 1
fi

cd "${REPO}"
export PYTHONPATH="${REPO}/code${PYTHONPATH:+:${PYTHONPATH}}"

run_step() {
  extval_time_step "NPC-UNI2-A2" "$1" \
    "${PYTHON}" -u - "$1" "${SAMPLE_ARG[@]}" "${FORCE_ARG[@]}" <<'PY'
import shutil
import sys
from pathlib import Path

from Hist2Pheno_pkg.external_validation import simple_cohort

repo = Path("/home/lingyu/ssd2/Python/Hist2Pheno")
step = sys.argv[1]
argv = sys.argv[2:]
cohort = simple_cohort("NPC", repo, "uni2")
cohort.pred_dirname = "pred_uni2_A2_bestmlp"
cohort.checkpoint = (
    repo
    / "data/PanCancer/result_uni2_A2_adjacent_spatial_k8mean"
    / "cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt"
)
cohort.scaler = Path(str(cohort.checkpoint) + ".artifact.pt")
if not cohort.checkpoint.is_file():
    raise SystemExit(f"missing A2 checkpoint: {cohort.checkpoint}")

# Same UNI2 matrix as A3. Copy it so predict does not rematch 140k .pth files.
src_root = cohort.data_root / "pred_uni2_A3_bestmlp"
for sample in cohort.discover_samples():
    src = src_root / sample / f"{sample}_stardist_features.h5ad"
    dst = cohort.sample_paths(sample)["feat"]
    if src.is_file() and not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[h5ad] copied A3 features -> {dst}", flush=True)

raise SystemExit(cohort.main([step, *argv]))
PY
}

run_steps() {
  case "${STEP}" in
    dry-run|predict|probs|plot)
      run_step "${STEP}" || return $?
      ;;
    all)
      run_step predict || return $?
      ;;
    *)
      usage
      return 1
      ;;
  esac
}

extval_run_logged "${REPO}/code/log/external_val_npc_uni2_A2.log" "NPC-UNI2-A2" "${STEP}" run_steps
