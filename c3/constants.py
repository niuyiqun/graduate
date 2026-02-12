"""
本文件定义了 SDAA 算法中涉及的所有物理常数与超参数，
严格对应论文第四章实验设置。
"""


class SDAAConstants:
    # --- 势能建模参数 ---
    ALPHA = 1.2  # PageRank 权重系数 (公式 4-3)
    BETA = 0.5  # 概念势能基置信度 (公式 4-3)
    GAMMA = 2.0  # 惊奇度影响因子 (公式 4-4)
    LAMBDA = 0.05  # 时间衰减系数 (公式 4-4)

    # --- 扩散门控参数 ---
    DIFFUSION_STEPS_K = 3  # 扩散步数 K
    SEMANTIC_THRESHOLD_DELTA = 0.75  # 语义共振硬门控阈值 (公式 4-6)
    FIRING_THRESHOLD_EPSILON = 0.65  # 节点激活阈值 (公式 4-8)

    # --- 轨迹预测参数 ---
    TIME_STEP_DELTA_T = 0.1  # 推演尺度 (公式 4-10)
    TOP_K_IMP = 5  # 隐式节点召回数量 (公式 4-12)

    # --- 剪枝控制参数 ---
    ENTROPY_BASE_K = 10  # 基础 Top-K 窗口
    ENTROPY_SCALING_FACTOR = 2.5  # 熵值对窗口的缩放系数