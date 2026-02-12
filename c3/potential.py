import numpy as np
import networkx as nx
from .constants import SDAAConstants
from ..c2.data_models import MemoryNode, NodeType


class PotentialModeler:
    """
    4.3.2(1) 双原子势能建模 (Dual-Atom Potential Modeling)
    负责初始化节点势能场，区分“稳态锚点”与“活跃刺激”。
    """

    def __init__(self, graph_kernel):
        self.kernel = graph_kernel

    def compute_initial_potential(self, node_id: str, t_now: float) -> float:
        """
        根据公式 (4-2) (4-3) (4-4) 计算初始激活势能 A0(v)
        """
        node: MemoryNode = self.kernel.graph.nodes[node_id]['data']

        # 判断原子流类型
        if node.node_type in [NodeType.CONCEPT_PROFILE, NodeType.CONCEPT_KNOWLEDGE]:
            return self._compute_concept_potential(node_id)
        else:
            return self._compute_episodic_potential(node, t_now)

    def _compute_concept_potential(self, node_id: str) -> float:
        """
        公式 (4-3): 概念原子势能 - 稳态锚点
        E0(v_con) = alpha * PR(v_con) + beta
        """
        # 利用 NetworkX 计算全局 PageRank 分数
        pr_scores = nx.pagerank(self.kernel.graph.to_undirected())
        pr_val = pr_scores.get(node_id, 0.0)

        potential = SDAAConstants.ALPHA * pr_val + SDAAConstants.BETA
        return self._softplus(potential)

    def _compute_episodic_potential(self, node: MemoryNode, t_now: float) -> float:
        """
        公式 (4-4): 情景原子势能 - 活跃刺激
        E0(v_epi) = gamma * (-log P(v_epi | M_sem)) * exp(-lambda * delta_t)
        """
        # 获取由 C1 阶段计算并存储在节点中的逻辑惊奇度 (Surprise Score)
        # 逻辑惊奇度 = -log P(v_epi | M_sem)
        surprise_score = node.base_importance

        # 计算时间冷却效应
        delta_t = t_now - node.timestamp
        time_decay = np.exp(-SDAAConstants.LAMBDA * delta_t)

        potential = SDAAConstants.GAMMA * surprise_score * time_decay
        return self._softplus(potential)

    def _softplus(self, x: float) -> float:
        """映射函数 phi (公式 4-2)，确保非负性"""
        return np.log(1 + np.exp(x))