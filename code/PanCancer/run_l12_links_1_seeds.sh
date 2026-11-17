#!/usr/bin/env bash
## 2026.11.17, l12_links_1 with the cohort-safe inner validation, three seeds
## l12_links_1（层级连接权重 1），inner validation 用 cohort_safe，跑 3 个 seed，只看内部 OOF 性能。
##
## Folds: cross_dataset_section, SPLIT_SEED=42 (the same outer folds as every arm so far;
## BRCA rep1 in fold 0, rep2 in fold 1). Only the early-stopping split changes:
## INNER_VAL=cohort_safe never takes a cohort's last training section or an L2 class
## out of training. Seeds change only the initialisation and batch order.
##
##   cd /home/lingyu/ssd2/Python/Hist2Pheno
##   conda activate SeededNTM
##   nohup bash code/PanCancer/run_l12_links_1_seeds.sh > code/log/l12_links_1_seeds.out 2>&1 &
##   bash code/PanCancer/run_l12_links_1_seeds.sh summary        # table only
##
## Settings (environment, optional):
##   SEEDS="42 43 44"    one training per seed
##   GPUS="0 1 2"        one queue per entry; "0 0 1" puts two seeds on GPU 0
##   DATA_TAG=new1066axisfix   folders: result_uni2_ablation_l12_links_1_<DATA_TAG>_ivsafe_paired_v2_seed<N>
##
## Output per seed: best_mlp_gpu.pt, oof_predictions.npz, decision_rule*.json,
## decision_report.csv (1_demo_train.sh fits the decision layer after training).
## Summary: data/PanCancer/l12_links_1_<DATA_TAG>_ivsafe_seeds.csv (one row per seed + mean ± sd),
## with the earlier random-inner-split l12_links_1 / spatial_k8 seed 42 rows for reference.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
CONDA_ENV="${CONDA_ENV:-SeededNTM}"
export DATA_TAG="${DATA_TAG:-new1066axisfix}"
export INNER_VAL=cohort_safe
export HIST2PHENO_AMP="${HIST2PHENO_AMP:-bf16}"
export HIST2PHENO_STORE_DTYPE="${HIST2PHENO_STORE_DTYPE:-float16}"
export DECISION_OBJECTIVE="${DECISION_OBJECTIVE:-f1_acc}"
export SPLIT_SEED=42
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"
read -r -a GPUS <<< "${GPUS:-0 1 2}"
TAG="${DATA_TAG}_ivsafe"
LOGDIR="$REPO/code/log/l12_links_1_${TAG}"
mkdir -p "$LOGDIR"
ARM=l12_links_1

train() {
  echo "===== l12_links_1 x seeds ${SEEDS[*]} on GPUs ${GPUS[*]}  inner val cohort_safe  $(date '+%F %T') ====="
  local n=${#GPUS[@]} i
  local -a queue=()
  for ((i=0; i<n; i++)); do queue[i]=""; done
  for ((i=0; i<${#SEEDS[@]}; i++)); do
    local q=$(( i % n ))
    queue[q]="${queue[q]} ${SEEDS[i]}"
  done
  # the first seed builds (or finds) the k=8 neighbor cache; the others wait for it
  local first_out="$LOGDIR/seed${SEEDS[0]}.out"
  local -a pids=()
  for ((i=0; i<n; i++)); do
    [[ -n "${queue[i]// /}" ]] || continue
    local gpu="${GPUS[i]}"
    (
      for seed in ${queue[i]}; do
        if [[ "$seed" != "${SEEDS[0]}" ]]; then
          waited=0
          until grep -q "context cache ready" "$first_out" 2>/dev/null || (( waited >= ${CACHE_WAIT_MIN:-360} )); do
            sleep 60; waited=$((waited + 1))
          done
        fi
        echo "[GPU $gpu] $(date '+%F %T') start seed $seed"
        if ! CUDA_DEVICE="$gpu" TRAIN_SEED="$seed" SEEDS="$seed" \
             bash code/PanCancer/run_ablation.sh "$ARM" > "$LOGDIR/seed${seed}.out" 2>&1; then
          echo "[GPU $gpu] FAILED seed $seed (see $LOGDIR/seed${seed}.out)" | tee -a "$LOGDIR/seed${seed}.out"
        fi
        echo "[GPU $gpu] $(date '+%F %T') done  seed $seed"
      done
    ) &
    pids+=($!)
    echo "GPU $gpu seeds:${queue[i]}"
  done
  wait "${pids[@]}"
  ! grep -h FAILED "$LOGDIR"/*.out >/dev/null 2>&1 || echo "some seeds failed; see $LOGDIR"
}

summary() {
  echo "===== summary $(date '+%F %T') ====="
  conda run --no-capture-output -n "$CONDA_ENV" python -u - "$TAG" "$DATA_TAG" <<'EOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code/Hist2Pheno_pkg")
from posthoc import run_fold_metrics

tag, base_tag = sys.argv[1], sys.argv[2]
pan = Path("data/PanCancer")
runs = []
for rd in sorted(pan.glob(f"result_uni2_ablation_l12_links_1_{tag}_paired_v2_seed*")):
    if (rd / "oof_predictions.npz").is_file():
        runs.append(("l12_links_1 (cohort_safe)", int(rd.name.rsplit("seed", 1)[1]), rd))
    else:
        print(f"  not finished: {rd.name}")
for arm in ("l12_links_1", "spatial_k8"):            # earlier runs, random inner split, for reference
    rd = pan / f"result_uni2_ablation_{arm}_{base_tag}_paired_v2_seed42"
    if (rd / "oof_predictions.npz").is_file():
        runs.append((f"{arm} (random, reference)", 42, rd))
if not runs:
    raise SystemExit("no finished run")

cols = ["L2_macro_auroc", "L2_accuracy_argmax", "L2_macro_f1_argmax",
        "L2_accuracy_decision", "L2_macro_f1_decision", "L12_macro_f1_argmax"]
rows, per_fold = [], []
for name, seed, rd in runs:
    fm = run_fold_metrics(rd)
    fm["run"], fm["seed"] = name, seed
    per_fold.append(fm)
    rec = {"run": name, "seed": seed}
    rec.update({c: float(fm[c].mean()) for c in cols if c in fm})   # mean over the 5 folds
    rows.append(rec)
table = pd.DataFrame(rows)
safe = table[table.run.str.contains("cohort_safe")]
if len(safe) > 1:
    agg = {"run": "l12_links_1 (cohort_safe) mean", "seed": -1}
    agg.update({c: safe[c].mean() for c in cols if c in safe})
    sd = {"run": "l12_links_1 (cohort_safe) sd over seeds", "seed": -1}
    sd.update({c: safe[c].std(ddof=1) for c in cols if c in safe})
    table = pd.concat([table, pd.DataFrame([agg, sd])], ignore_index=True)
out = pan / f"l12_links_1_{tag}_seeds.csv"
table.to_csv(out, index=False)
pd.concat(per_fold).to_csv(out.with_name(out.stem + "_per_fold.csv"), index=False)
pd.set_option("display.width", 200)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nfold-mean metrics; per fold: {out.with_name(out.stem + '_per_fold.csv')}")

# inner-validation audit written by model.py (2026.11.17)
for name, seed, rd in runs:
    for f in sorted(rd.rglob("group_cv_folds.csv")):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if "inner_lost_classes" in t:
            lost = t["inner_lost_classes"].fillna("").astype(str)
            brca = t.get("inner_val_sections", pd.Series([""] * len(t))).fillna("").astype(str)
            print(f"{name} seed {seed}: folds whose inner training lost an L2 class "
                  f"{int((lost != '').sum())}/{len(t)}; folds with a BRCA section in inner val "
                  f"{int(brca.str.contains('BRCA').sum())}/{len(t)}")
        else:
            print(f"{name} seed {seed}: {f.name} has no inner-val audit (trained before 2026-11-17 code)")
        break
EOF
}

case "${1:-all}" in
  train) train ;;
  summary) summary ;;
  all) train; summary ;;
  *) echo "usage: bash code/PanCancer/run_l12_links_1_seeds.sh [all|train|summary]" >&2; exit 1 ;;
esac
