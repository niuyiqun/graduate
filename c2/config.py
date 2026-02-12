"""
c2.config - 系统参数配置
该文件定义了神经符号系统的全局超参数，对应论文中的实验设置。
"""


class SystemConfig:
    # --- 语义阈值 (Semantic Thresholds) ---
    # 判定两个原子是否存在显式语义关联的 Cosine 相似度下限
    SEMANTIC_CONNECT_THRESHOLD = 0.78

    # --- 演化参数 (Evolution Parameters) ---
    # 判定潜在冲突的召回阈值
    CONFLICT_RECALL_THRESHOLD = 0.85
    # 艾宾浩斯记忆衰减系数 (alpha in decay formula)
    EBBINGHAUS_DECAY_RATE = 0.15

    # --- 拓扑推理参数 (Topological Inference) ---
    # Adamic-Adar 指标阈值，用于判定隐式连接 (Implicit Link)
    TOPOLOGY_INFERENCE_THRESHOLD = 2.5

    # --- 概念涌现参数 (Emergence Parameters) ---
    # 社区发现算法的分辨率 (Resolution for Leiden/Louvain)
    # 较高的分辨率会产生更小的簇
    COMMUNITY_RESOLUTION = 1.0
    # 触发概念归纳的最小社区规模
    MIN_COMMUNITY_SIZE = 3

    # --- 存储配置 ---
    LOG_PATH = "logs/neuro_symbolic_construction.log"
    GRAPH_EXPORT_PATH = "outputs/memory_graph.gml"