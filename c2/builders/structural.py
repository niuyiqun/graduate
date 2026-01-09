# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：structural.py
@Author  ：niu
@Date    ：2026/1/8 13:26 
@Desc    ：
"""

import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

# === 路径与配置导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入配置
try:
    from ..config import (
        GNN_IN_DIM, GNN_HIDDEN_DIM, GNN_OUT_DIM, GNN_RELATIONS,
        GNN_EPOCHS, GNN_LR, LINK_PREDICTION_THRESHOLD, LLM_CONFIG_PATH
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        GNN_IN_DIM, GNN_HIDDEN_DIM, GNN_OUT_DIM, GNN_RELATIONS,
        GNN_EPOCHS, GNN_LR, LINK_PREDICTION_THRESHOLD, LLM_CONFIG_PATH
    )

# 导入 LLM
try:
    from general.model import ZhipuChat
except ImportError:
    try:
        from model import ZhipuChat
    except ImportError:
        pass

    # 导入 PyG
try:
    from torch_geometric.nn import RGCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("❌ [Error] torch_geometric import failed. GNN training will be skipped.")

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import LOGIC_VERIFICATION_PROMPT


# ==========================================
# 🧠 1. 定义真正的 GNN 模型 (Encoder)
# ==========================================
class NeuroSymbolicGNN(nn.Module):
    """
    [Real Model] 神经符号图神经网络
    架构: RGCN (Encoder) -> Dot Product (Decoder)
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        if HAS_PYG:
            # 第一层: 压缩语义，融合邻居信息
            self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
            # 第二层: 进一步抽象出结构化特征
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)

            # 激活与正则化
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

    def encode(self, x, edge_index, edge_type):
        """生成节点的结构化 Embedding (z)"""
        if not HAS_PYG: return x

        # Layer 1
        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z, edge_index):
        """
        链路预测解码器 (Link Prediction Decoder)
        计算边两端节点的相似度分数
        """
        # z[src] * z[dst]
        src, dst = edge_index
        score = (z[src] * z[dst]).sum(dim=-1)
        return score


# ==========================================
# 🏗️ 2. StructuralBuilder (带训练循环)
# ==========================================
class StructuralBuilder(BaseGraphBuilder):
    def __init__(self):
        # 使用 Config 中的配置
        self.in_dim = GNN_IN_DIM
        self.hidden_dim = GNN_HIDDEN_DIM
        self.out_dim = GNN_OUT_DIM
        self.num_rels = GNN_RELATIONS

        self.epochs = GNN_EPOCHS
        self.lr = GNN_LR
        self.threshold = LINK_PREDICTION_THRESHOLD

        # 初始化模型
        self.model = NeuroSymbolicGNN(self.in_dim, self.hidden_dim, self.out_dim, self.num_rels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化 LLM (验证用)
        self.llm = ZhipuChat(LLM_CONFIG_PATH)

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Structural] 准备启动 GNN 自监督训练...")

        nodes = graph.get_all_nodes()
        if not nodes: return
        node_map = {n.id: i for i, n in enumerate(nodes)}

        # === A. 数据准备 (Graph -> PyG Data) ===
        x_list = []
        for n in nodes:
            if n.embedding is not None and len(n.embedding) > 0:
                x_list.append(torch.tensor(n.embedding))
            else:
                x_list.append(torch.randn(self.in_dim))  # Fallback

        x = torch.stack(x_list)

        # 构建边 (Edge Index)
        edge_indices = []
        edge_types = []

        type_map = {EdgeType.SEMANTIC: 0, EdgeType.TEMPORAL: 1, EdgeType.VERSION: 2, EdgeType.IMPLICIT: 3}

        edge_count = 0
        for n in nodes:
            u = node_map[n.id]
            for e in n.edges:
                if e.target in node_map:
                    v = node_map[e.target]
                    edge_indices.append([u, v])
                    edge_types.append(type_map.get(e.type, 0))
                    edge_count += 1

        if edge_count == 0:
            print("    ⚠️ 图中暂无边，跳过 GNN 训练。")
            return

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # === B. 自监督训练 (Self-Supervised Training) ===
        z = x
        if HAS_PYG:
            self._train_gnn(x, edge_index, edge_type)

            # 生成最终的 embedding z
            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(x, edge_index, edge_type)

        # === C. 召回与验证 (Recall & Verify) ===
        self._predict_links(z, new_nodes, nodes, node_map, graph)

    def _train_gnn(self, x, edge_index, edge_type):
        """训练循环"""
        print(f"    🏋️ [GNN Training] Start ({self.epochs} epochs)...")
        self.model.train()

        final_loss = 0.0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            z = self.model.encode(x, edge_index, edge_type)
            pos_score = self.model.decode(z, edge_index)
            neg_edge_index = self._negative_sampling(edge_index, x.size(0))
            neg_score = self.model.decode(z, neg_edge_index)

            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            loss = pos_loss + neg_loss

            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()

        print(f"    ✅ [GNN Training] Done. Final Loss: {final_loss:.4f}")

    def _negative_sampling(self, edge_index, num_nodes):
        """简单随机负采样"""
        num_edges = edge_index.size(1)
        neg_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        return neg_edge_index

    def _predict_links(self, z, new_nodes, all_nodes, node_map, graph):
        """推理阶段：基于训练好的 z 找新连接"""
        new_idxs = [node_map[n.id] for n in new_nodes if n.id in node_map]

        # 动态计算 Top-K (避免节点过少报错)
        num_nodes = len(all_nodes)
        k = min(5, num_nodes)
        if k == 0: return

        for i in new_idxs:
            sims = torch.cosine_similarity(z[i].unsqueeze(0), z)
            top_vals, top_idxs = torch.topk(sims, k=k)

            for val, idx in zip(top_vals, top_idxs):
                j = idx.item()
                if i == j: continue

                source_node = all_nodes[i]
                target_node = all_nodes[j]

                if any(e.target == target_node.id for e in source_node.edges): continue

                # === LLM 验证 ===
                if val > self.threshold:
                    print(
                        f"    🔍 [GNN Proposal] '{source_node.content[:8]}...' <-> '{target_node.content[:8]}...' (Score: {val:.2f})")

                    if self._llm_verify(source_node, target_node):
                        print(f"      ✅ [LLM Verified] 建立隐式关联 (Implicit Link)")
                        graph.add_edge(source_node.id, target_node.id, EdgeType.IMPLICIT, weight=val.item())

    def _llm_verify(self, n1, n2) -> bool:
        prompt = LOGIC_VERIFICATION_PROMPT.format(text_a=n1.content, text_b=n2.content)
        try:
            result = self.llm.chat([{"role": "user", "content": prompt}])
            if isinstance(result, dict):
                return "PASS" in result.get("status", "REJECT").upper()
            return "PASS" in str(result).upper()
        except:
            return False