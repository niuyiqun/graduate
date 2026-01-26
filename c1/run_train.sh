#!/bin/bash

# ================= 配置区 =================
# 显式指定使用前 6 张显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# 项目根目录
PROJECT_DIR="/root/.nyq/graduate"
# =========================================

# 关键：无论脚本在哪，先跳回项目根目录，这样后续的路径引用才不会错
cd "$PROJECT_DIR" || exit

# 激活环境
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mem

echo ">>> [Train] 启动 GRPO 6卡分布式训练..."
echo ">>> 脚本位置: c1/run_train.sh"
echo ">>> 日志输出: c1/grpo_train.log"

# 启动训练
torchrun --nproc_per_node=6 --master_port=29500 c1/grpo.py