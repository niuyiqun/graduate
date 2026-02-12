# c3/prediction.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryPredictor(nn.Module):
    """
    轻量级轨迹预测网络
    Input: h_context (当前思维重心)
    Output: h_next (下一时刻预测重心)
    """

    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, h_context):
        # 简单的残差预测
        delta = F.relu(self.fc1(h_context))
        delta = self.fc2(delta)
        h_next = self.layer_norm(h_context + delta)
        return h_next


class VectorEvolution:
    """
    Step 3: 基于子图表征的流形轨迹预测
    """

    def __init__(self, predictor_model, embedding_index):
        self.model = predictor_model
        self.index = embedding_index  # 向量数据库索引 (e.g., FAISS)

    def predict_next_nodes(self, active_subgraph, energy_map, top_k=10):
        """
        1. 结构语义聚合 -> h_context
        2. 向量演化 -> h_next
        3. 全局投影 -> 候选节点
        """
        # 1. 加权聚合计算语义重心
        h_context = 0
        total_energy = 0

        for node_id in active_subgraph:
            energy = energy_map.get(node_id, 0)
            emb = self._get_node_embedding(node_id)  # 从存储获取
            h_context += emb * energy
            total_energy += energy

        if total_energy > 0:
            h_context /= total_energy

        # 2. 预测演化
        with torch.no_grad():
            h_tensor = torch.tensor(h_context).float()
            h_next = self.model(h_tensor).numpy()

        # 3. 全局全息投影 (近似最近邻搜索)
        # 这一步能命中"逻辑相关但未连接"的跳跃节点
        scores, indices = self.index.search(h_next.reshape(1, -1), top_k)

        return indices[0], scores[0]  # 返回候选节点ID及其相似度

    def _get_node_embedding(self, node_id):
        # Mock: 实际应从 GraphStorage 或 VectorDB 获取
        return np.random.randn(768)