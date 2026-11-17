#!/usr/bin/env bash
## BRCA internal val — held-out fold 0 only (Xenium rep1).
##
## Each finished 2026-11-17 arm is applied with the checkpoint named on
## --checkpoint. PanCancer_predict_visualize.py then swaps in
## hce_group_fold_0.pt, the weights that held BRCA::rep1 out.
## best_mlp_gpu.pt for these arms is fold 4, which trained on both BRCA
## slides, so this script does not pass --deployment.
##
## Output:
##   data/PanCancer/pred_uni2_ablation_<arm>_new1066axisfix_paired_v2_seed42_brca_fold0/oof/
##
## gate_l12_balanced is still training fold 0 and is not in this list.
## Do not point CUDA_DEVICE at the GPU that retrain_all.sh is using.
##
##   CUDA_DEVICE=1 bash code/Internal_val/demo_uni2_axisfix_brca.sh predict
##   CUDA_DEVICE=1 bash code/Internal_val/demo_uni2_axisfix_brca.sh predict spatial_k8
##   CUDA_DEVICE=1 bash code/Internal_val/demo_uni2_axisfix_brca.sh predict-ivsafe
##
## predict-ivsafe writes the cohort_safe l12_links_1 seeds 42 and 44, the two
## runs read by Pred_visual_brca.ipynb. Same rule: fold-0 weights, not best_mlp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

ARMS=(
  legacy_nospatial
  adjacent_nospatial
  spatial_k8
  spatial_k8_gate
  l12_links_1
  l12_links_2
  l12_balanced
  focal_gamma1
  balanced_sampler
  l12_balanced_nospatial
)

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
  echo "ERROR: SeededNTM python not found. Set PYTHON=..." >&2
  exit 1
fi

STEP="${1:-}"
WHICH="${2:-all}"

usage() {
  cat <<EOF
Usage: CUDA_DEVICE=1 bash code/Internal_val/demo_uni2_axisfix_brca.sh predict [arm|all]
       CUDA_DEVICE=1 bash code/Internal_val/demo_uni2_axisfix_brca.sh predict-ivsafe

predict: one finished 2026-11-17 arm, fold-0 checkpoint, Xenium BRCA rep1.
predict-ivsafe: cohort_safe l12_links_1 seeds 42 and 44, same held-out fold.
EOF
}

known_arm() {
  local arm="$1" one
  for one in "${ARMS[@]}"; do
    [[ "$one" == "$arm" ]] && return 0
  done
  return 1
}

run_arm() {
  local arm="$1"
  local result="${REPO}/data/PanCancer/result_uni2_ablation_${arm}_new1066axisfix_paired_v2_seed42"
  local ckpt="${result}/cross_dataset_cv/${arm}/best_mlp_gpu.pt"
  local pred="${REPO}/data/PanCancer/pred_uni2_ablation_${arm}_new1066axisfix_paired_v2_seed42_brca_fold0"
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: missing ${ckpt}" >&2
    return 1
  fi
  echo "===== ${arm}  ckpt=${ckpt}"
  echo "===== out=${pred}/oof/matched/BRCA/rep1"
  "${PYTHON}" -u code/PanCancer/PanCancer_predict_visualize.py \
    --embedder uni2 \
    --checkpoint "${ckpt}" \
    --pred-root "${pred}" \
    --datasets BRCA \
    --fold 0 \
    --cuda-device "${CUDA_DEVICE}" \
    --skip-existing \
    --skip-figures
}

if [[ -z "${STEP}" || "${STEP}" == "-h" || "${STEP}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${REPO}"

if [[ "${STEP}" == "predict-ivsafe" ]]; then
  rc=0
  for seed in 42 44; do
    result="${REPO}/data/PanCancer/result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed${seed}"
    ckpt="${result}/cross_dataset_cv/l12_links_1/hce_group_fold_0.pt"
    pred="${REPO}/data/PanCancer/pred_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed${seed}_brca_fold0"
    if [[ ! -f "${ckpt}" ]]; then
      echo "ERROR: missing ${ckpt}" >&2
      rc=1
      continue
    fi
    echo "===== l12_links_1 seed ${seed}  ckpt=${ckpt}"
    echo "===== out=${pred}/oof/matched/BRCA/rep1"
    "${PYTHON}" -u code/PanCancer/PanCancer_predict_visualize.py \
      --embedder uni2 \
      --checkpoint "${ckpt}" \
      --pred-root "${pred}" \
      --datasets BRCA \
      --fold 0 \
      --cuda-device "${CUDA_DEVICE}" \
      --skip-existing \
      --skip-figures || rc=1
  done
  exit "${rc}"
fi

if [[ "${STEP}" != "predict" ]]; then
  usage
  exit 1
fi

chosen=()
if [[ "${WHICH}" == "all" ]]; then
  chosen=("${ARMS[@]}")
else
  known_arm "${WHICH}" || { echo "ERROR: unknown arm ${WHICH}" >&2; usage; exit 1; }
  chosen=("${WHICH}")
fi

rc=0
for arm in "${chosen[@]}"; do
  run_arm "${arm}" || rc=1
done
exit "${rc}"
