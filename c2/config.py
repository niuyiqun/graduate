# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：config.py.py
@Author  ：niu
@Date    ：2026/1/9 15:48 
@Desc    ：
"""

import os

# === 基础路径配置 ===
# 获取 graduate/ 根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 配置文件路径
LLM_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "llm_config.yaml")

# 模型路径
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "all-MiniLM-L6-v2")

# 存档路径
GRAPH_SAVE_PATH = os.path.join(CURRENT_DIR, "c2_graph_storage.pkl")

# === 算法参数配置 ===

# 1. 语义侧 (Semantic)
SEMANTIC_THRESHOLD = 0.5  # 暂时预留，虽然目前主要靠实体交集

# 2. 演化侧 (Evolution)
# 每次新原子进来，向前检索多少个旧原子进行冲突检测
CONFLICT_RETRIEVAL_WINDOW = 10
# 冲突降权系数 (Conflict Penalty)
DECAY_FACTOR = 0.5

# 3. 符号侧 (Structural / GNN)
GNN_IN_DIM = 384
GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 32
GNN_RELATIONS = 4

GNN_EPOCHS = 50           # 训练轮数
GNN_LR = 0.01             # 学习率
LINK_PREDICTION_THRESHOLD = 0.6  # 隐式连接的置信度阈值 (建议比Demo高一点，因为真实数据噪声大)

# === API 优化配置 ===
USE_BATCHING = False      # 未来开启批量处理
BATCH_SIZE = 5