#!/usr/bin/env bash
## Ablation on the current 1,066-section index (the updated seven cohorts).
## Does not open the 896-section directories.
##
## A2 is the model to use: adjacent loss, k=8 mean, no gate.
## Attention pooling was removed (the gather is too expensive).
## The neighbor-composition head and the tissue-compartment head were removed:
## neither improved L2 macro-F1 over the gated model.
##
##   A0  legacy loss, no neighbors
##   A1  adjacent loss, no neighbors
##   A2  k=8 mean                         (finished — not retrained)
##   A3  A2 + spatial gate
##   A4  A2, L12 link weights 1
##   A5  A2, L12 link weights 2
##   A6  A2, balanced sampler + focal
##   A7  A2 + balanced L12, no gate       (finished — not retrained)
##   A8  A3 + balanced L12                (finished — not retrained)
##
##   CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A0
## A finished id prints its weight and exits. It does not retrain.
set -euo pipefail

arch="${1:-}"
case "$arch" in
  A0|A1|A2|A3|A4|A5|A6|A7|A8) ;;
  A9|A10|A11)
    echo "${arch} was removed. The neighbor and tissue heads did not improve L2 macro-F1." >&2
    exit 1
    ;;
  *)
    echo "usage: CUDA_DEVICE=0 bash code/PanCancer/run_uni2_new1066.sh A0|A1|A2|A3|A4|A5|A6|A7|A8" >&2
    exit 1
    ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
out="${repo}/data/PanCancer"

if [[ "${SPLITTER:-cross_dataset_section}" == cross_dataset_section ]]; then
  done_ckpt=""
  case "$arch" in
    A2) done_ckpt="${out}/result_uni2_A2_adjacent_spatial_k8mean_new1066_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt" ;;
    A7) done_ckpt="${out}/result_uni2_A2_l12_balanced_new1066_paired_v2_seed42/cross_dataset_cv/A2_l12_balanced/best_mlp_gpu.pt" ;;
    A8) done_ckpt="${out}/result_uni2_A8_l12_balanced_paired_v2_seed42/cross_dataset_cv/A8_l12_balanced/best_mlp_gpu.pt" ;;
  esac
  if [[ -n "$done_ckpt" && -f "$done_ckpt" ]]; then
    echo "${arch} is already trained on the 1,066-section index."
    echo "  ${done_ckpt}"
    echo "not retraining"
    exit 0
  fi
fi

# Internal recipe token. Public ids above are the ablation names.
token="$arch"
case "$arch" in
  A4) token="A5" ;;
  A5) token="A6" ;;
  A6) token="A7" ;;
  A7) token="A2_l12_balanced" ;;
  A8) token="A8" ;;
esac

export RUN_SUFFIX="new1066_paired_v2_seed${TRAIN_SEED:-42}"
if [[ "$RUN_SUFFIX" != new1066_* ]]; then
  echo "refusing to write a published paired_v2_seed42 directory" >&2
  exit 1
fi

exec bash code/PanCancer/1_demo_train.sh uni2 "$token"
