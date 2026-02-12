import numpy as np
from typing import List, Dict
from .constants import SDAAConstants
from ..c2.data_models import MemoryNode


class TrajectoryPredictor:
    """
    4.3.2(3) 流形轨迹预测 (Manifold Trajectory Prediction)
    跨越拓扑稀疏区域，推演思维重心的下一落点。
    """

    def __init__(self, all_nodes_pool: List[MemoryNode]):
        self.pool = all_nodes_pool
        # 简化版预测器 F_theta 的参数 (实际通过公式 4-11 离线训练得到)
        self.theta_velocity_vector = np.random.normal(0, 0.01, 128)

    def predict_implicit_nodes(self, act_nodes: List[MemoryNode], energies: Dict[str, float]):
        """
        执行轨迹外推与隐式召回 (公式 4-9 至 4-12)
        """
        if not act_nodes: return []

        # 1. 计算语义重心 h_ctx (公式 4-9)
        h_ctx = self._calculate_centroid(act_nodes, energies)

        # 2. 轨迹推演 h_next (公式 4-10)
        # 模拟预测器产生的位移向量
        h_next = h_ctx + self.theta_velocity_vector * SDAAConstants.TIME_STEP_DELTA_T

        # 3. 全局全息投影召回 (公式 4-12)
        return self._ann_search(h_next)

    def _calculate_centroid(self, nodes, energies):
        """加权聚合当前思维状态坐标"""
        vecs = []
        weights = []
        for n in nodes:
            vecs.append(n.embedding)
            weights.append(energies.get(n.node_id, 0.0))

        return np.average(vecs, axis=0, weights=weights)

    def _ann_search(self, query_vec):
        """近似最近邻搜索模拟"""
        scores = []
        for node in self.pool:
            sim = np.dot(query_vec, node.embedding) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(node.embedding) + 1e-9)
            scores.append((node, sim))

        # 召回 Top-K 隐式落点
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scores[:SDAAConstants.TOP_K_IMP]]