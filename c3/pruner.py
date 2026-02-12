import numpy as np
from typing import List, Dict
from .constants import SDAAConstants
from ..c2.data_models import MemoryNode


class EntropyPruner:
    """
    4.3.2(4) 熵减自适应剪枝 (Entropy-based Adaptive Pruning)
    动态调整预取窗口大小，平衡召回率与 Token 开销。
    """

    def prune(self, candidate_nodes: List[MemoryNode], energy_map: Dict[str, float]):
        """
        基于信息熵评估聚焦程度 (公式 4-13, 4-14)
        """
        if not candidate_nodes: return []

        # 1. 计算归一化激活分布 p(v) (公式 4-13)
        node_ids = [n.node_id for n in candidate_nodes]
        raw_energies = np.array([energy_map.get(nid, 0.01) for nid in node_ids])
        p_v = raw_energies / (np.sum(raw_energies) + 1e-9)

        # 2. 计算候选集信息熵 H(C) (公式 4-14)
        entropy = -np.sum(p_v * np.log(p_v + 1e-12))

        # 3. 动态调整 Top-K 阈值
        # 熵低 (分布尖锐) -> 聚焦 -> K小
        # 熵高 (分布平坦) -> 迷茫 -> K大
        dynamic_k = int(SDAAConstants.ENTROPY_BASE_K * (1 + entropy * SDAAConstants.ENTROPY_SCALING_FACTOR))

        # 4. 执行剪枝
        sorted_indices = np.argsort(raw_energies)[::-1]
        selected_nodes = [candidate_nodes[i] for i in sorted_indices[:dynamic_k]]

        return selected_nodes, entropy