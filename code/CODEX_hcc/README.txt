## 2026.07.06 
后台任务
已启动 GT → StarDist 顺序跑完 38 个样本：

日志：logs/codex_uni_batch_20260706.log
reg001 已跳过（33706 / 42877 个 .pth 已存在）
当前正在处理 reg002

说明
这是我在 Cursor 里通过 Shell 工具直接执行的，不是写进某个 .sh 里的固定步骤。
若你本地又手动跑了 demo_GT_feature_extraction_Single.sh stardist，会和后台任务共用 GPU，可能变慢或冲突。
若要自己重新启动同样任务，直接复制上面的 nohup 命令即可；若只想跑 GT 或 StarDist，删掉对应那一行。




mkdir -p /home/lingyu/ssd2/Python/Hist2Pheno/logs && nohup bash -c '
  set -euo pipefail
  cd /home/lingyu/ssd2/Python/Hist2Pheno
  echo "=== START $(date) ==="
  bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh gt
  bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh stardist
  echo "=== FINISH $(date) ==="
' > /home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/HCC/Michael_data_transfer/Results/Img_fea_ext_logs/codex_uni_batch_20260706.log 2>&1 &
echo "PID=$!"


nohup ... &
终端关闭后进程仍继续跑；& 放到后台
bash -c '...'
在一个子 shell 里顺序执行 GT → StarDist
> ...log 2>&1
stdout、stderr 都写入日志
echo "PID=$!"
打印后台进程 PID（当时是 1762315）


# 实时看日志
tail -f /home/lingyu/ssd2/Python/Hist2Pheno/logs/codex_uni_batch_20260706.log

# 确认是否还在跑
ps -p 1762315 -o pid,cmd




