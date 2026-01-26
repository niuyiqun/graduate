#!/bin/bash
cd /root/.nyq/graduate

# ========================================================
# 🛠️ 动态定位 Conda (无论安装在哪里都能找到)
# ========================================================
# 1. 询问 conda 可执行文件它的安装根目录在哪里
CONDA_BASE=$(conda info --base)

# 2. 拼接出 conda.sh 的绝对路径并加载
source "$CONDA_BASE/etc/profile.d/conda.sh"
# ========================================================

conda activate mem

echo ">>> 启动 Qwen Server..."
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
  --model /root/.nyq/graduate/model/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --trust-remote-code \
  --port 8001 \
  --gpu-memory-utilization 0.95