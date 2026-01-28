# c3/potential.py

import networkx as nx
import numpy as np


class PotentialField:
    """
    Step 1: 双原子差异化势能建模
    """

    def __init__(self, graph_storage):
        self.graph = graph_storage.graph  # 假设底层是 NetworkX 或类似图结构

    def calculate_initial_mass(self, node_id, atom_data):
        """
        根据原子类型计算初始质量 (Mass)
        """
        atom_type = atom_data.get("atom_type", "unknown")

        # A. 概念原子 (Concept Atom) -> 认知锚点
        # 质量来源: 拓扑中心性 (PageRank / Degree Centrality)
        if "semantic_" in atom_type:
            # 简化：使用度中心性作为基础质量，模拟"引力"
            degree = self.graph.degree(node_id)
            # 给予极高的基础质量权重，形成稳态盆地
            mass = 10.0 * (1 + np.log(1 + degree))
            decay_rate = 0.1  # 低衰减

        # B. 情景原子 (Event Atom) -> 活跃粒子
        # 质量来源: C1 信息权重 (Information Weight / Surprise)
        elif "episodic_" in atom_type:
            # 获取 C1 计算的惊奇度/权重，默认为 1.0
            info_weight = atom_data.get("info_weight", 1.0)
            # 只有高惊奇度事件才有显著质量
            mass = 5.0 * info_weight
            decay_rate = 0.8  # 高衰减，随时间迅速冷却

        else:
            mass = 1.0
            decay_rate = 0.5

        return mass, decay_rate