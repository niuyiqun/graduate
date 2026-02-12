from typing import Dict
import numpy as np
from .constants import SDAAConstants
from ..c2.data_models import EdgeType, NodeType


class GatedDiffuser:
    """
    4.3.2(2) 神经符号门控扩散 (Neuro-Symbolic Gated Diffusion)
    实现各向异性传播，模拟有方向、有证据的思维联想。
    """

    def __init__(self, graph):
        self.graph = graph

    def run_k_step_diffusion(self, initial_energies: Dict[str, float]) -> Dict[str, float]:
        """
        执行 K 步扩散循环 (公式 4-5)
        """
        current_A = initial_energies

        for k in range(SDAAConstants.DIFFUSION_STEPS_K):
            next_A = {}
            # 遍历所有可能被激活的邻域节点
            for v_j in self.graph.nodes:
                # 获取节点 v_j 的所有入边邻居 v_i
                incoming_neighbors = list(self.graph.predecessors(v_j))

                sum_energy = 0.0
                for v_i in incoming_neighbors:
                    energy_i = current_A.get(v_i, 0.0)
                    if energy_i <= 0: continue

                    # 获取边属性
                    edge_data = self.graph.get_edge_data(v_i, v_j)
                    # 处理 MultiDiGraph 结构
                    for _, attr in edge_data.items():
                        e_type = attr['type']
                        w_ij = attr.get('weight', 1.0)

                        # 计算门控函数 G(v_i, v_j)
                        g_val = self._compute_gate(v_i, v_j, e_type)
                        sum_energy += w_ij * g_val * energy_i

                if sum_energy > 0:
                    # 公式 4-5 中的非负激活函数 psi (ReLU)
                    next_A[v_j] = max(0.0, sum_energy)

            current_A = next_A

        return current_A

    def _compute_gate(self, v_i: str, v_j: str, edge_type: str) -> float:
        """
        实现公式 (4-6) 和 (4-7) 的门控逻辑
        """
        node_i = self.graph.nodes[v_i]['data']
        node_j = self.graph.nodes[v_j]['data']

        # 1. 语义共振门控 (公式 4-6)
        if edge_type == EdgeType.SEMANTIC_REL.value:
            # 硬门控机制：抑制语义无关的传播
            similarity = self._cosine_sim(node_i.embedding, node_j.embedding)
            return 1.0 if similarity > SDAAConstants.SEMANTIC_THRESHOLD_DELTA else 0.0

        # 2. 时序因果门控 (公式 4-7)
        elif edge_type == EdgeType.TEMPORAL_NEXT.value:
            t_i, t_j = node_i.timestamp, node_j.timestamp
            # Mask 用于约束正向传播 (过去 -> 未来)
            if t_i > t_j: return 0.0
            # Decay 刻画时间差衰减
            return np.exp(-(t_j - t_i) / 3600)  # 以小时为单位的衰减

        return 0.5  # 默认传导率

    def _cosine_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)