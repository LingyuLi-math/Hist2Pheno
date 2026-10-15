#!/usr/bin/env bash
## Run every Pred_visual_*.ipynb in External_val, one after another.
## Does not overwrite the source notebooks with embedded images.
##
##   bash code/External_val/demo.sh notebooks
##   ONLY=BRCA,CRC bash code/External_val/demo.sh notebooks
##   SAVE_FIGURES=0 bash code/External_val/demo.sh notebooks
##   STOP_ON_ERROR=1 bash code/External_val/demo.sh notebooks

## 2026.10.15 Predict with new model  
# 用批次脚本按顺序跑 20 本 绘图 notebook（IPython + 无界面 Agg），不改源文件。
# 默认 SAVE_FIGURES=1，300 dpi PNG 写到各队列的 pred_pancancer_section/figures/。
# cd /home/lingyu/ssd2/Python/Hist2Pheno
# conda activate SeededNTM
# screen -S extval_nb
# bash code/External_val/demo.sh notebooks


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
usage() {
  cat <<EOF
Usage: bash code/External_val/demo.sh notebooks

Runs Pred_visual_*.ipynb sequentially (IPython + Agg). Source notebooks are
not rewritten. Default SAVE_FIGURES=1 writes 300 dpi PNGs under each cohort
pred_pancancer_section/figures/.

ONLY=BRCA,CRC          subset by path substring
SAVE_FIGURES=0         display-only (no PNG)
STOP_ON_ERROR=1        halt at the first failure
Log: code/External_val/Pred_visual_batch.log
EOF
}

if [[ "${STEP}" != "notebooks" ]]; then
  usage
  exit 1
fi

cd "${REPO}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
exec "${PYTHON}" -u "${SCRIPT_DIR}/run_pred_visual_notebooks.py"
