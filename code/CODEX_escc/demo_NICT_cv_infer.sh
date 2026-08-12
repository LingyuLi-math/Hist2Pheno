#!/usr/bin/env bash
set -euo pipefail

# 1) 固定你要复用的历史 run_id（就是你给的这个目录名后缀）
RUN_ID="20260407161448025350"

# 2) 路径配置
PY_ROOT="/home/lingyu/ssd2/Python"
PROJ_ROOT="/home/lingyu/ssd2/Python/Hist2Pheno"
DATA_ROOT="${PROJ_ROOT}/data/CODEX/ESCC"
THERAPY_DATA="NCRT"
THERAPY_MODEL="NCRT_project_all"
SAVE_RESULT="result"
RUN_PREFIX="train_validate_"

# 3) 复用该 run 下已有模型（不重新训练 5-fold）
BEST_MODEL_PATH="${DATA_ROOT}/${THERAPY_DATA}/${THERAPY_MODEL}/${SAVE_RESULT}/${RUN_PREFIX}${RUN_ID}/best_mlp_gpu.pt"

# 4) 调脚本：
#    - cv_mode 用 random_split（避免进入 5-fold 分支）
#    - checkpoint_exists=true（直接加载已有 best_mlp_gpu.pt）
#    - run_stardist_eval（只为重绘 StarDist 图）
#    - parent_value_stardist=all（输出 celltype_pred_all_level1_hce.jpg 等）
python "${PROJ_ROOT}/code/CODEX_escc/model_train_validate_cv_method.py" \
  --python_root "${PY_ROOT}" \
  --project_root "${PROJ_ROOT}" \
  --therapy_data "${THERAPY_DATA}" \
  --therapy_model "${THERAPY_MODEL}" \
  --parent_value all \
  --save_result "${SAVE_RESULT}" \
  --run_prefix "${RUN_PREFIX}" \
  --run_name "${RUN_ID}" \
  --cv_mode random_split \
  --checkpoint_exists true \
  --best_model_path "${BEST_MODEL_PATH}" \
  --run_stardist_eval \
  --parent_value_stardist all \
  --spatial_fig_w 64 \
  --spatial_fig_h 48 \
  --spatial_point_size 0.5


  # --hidden_dims 768 384 192 \