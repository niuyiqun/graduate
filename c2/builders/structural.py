# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：structural.py
@Author  ：niu
@Date    ：2026/1/8 13:26 
@Desc    ：
"""

# c2/builders/structural.py
import torch
import torch.nn as nn
from typing import List
from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph

# 尝试导入 PyG
try:
    from torch_geometric.nn import RGCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class MockGNN(nn.Module):
    """简单的 RGCN 模型"""

    def __init__(self, in_dim, out_dim, num_rels):
        super().__init__()
        if HAS_PYG:
            self.conv1 = RGCNConv(in_dim, out_dim, num_rels)

    def forward(self, x, edge_index, edge_type):
        if HAS_PYG:
            return self.conv1(x, edge_index, edge_type)
        return x  # 降级模式：直接返回 Embedding


class StructuralBuilder(BaseGraphBuilder):
    """
    [Phase 3] 符号侧 (Symbolic Side)
    GNN 结构推理(Recall) -> LLM 语义验证(Verify)
    """

    def __init__(self):
        self.gnn = MockGNN(768, 64, 4)  # 假设4种边类型
        self.threshold = 0.8

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Structural] GNN 正在进行隐式推理...")

        # 1. 构图 (Graph -> Tensor)
        nodes = graph.get_all_nodes()
        node_map = {n.id: i for i, n in enumerate(nodes)}

        # Mock Embedding
        x = torch.stack([torch.randn(768) for _ in nodes])

        # 简单构建 Edge Index
        edge_indices = []
        edge_types = []
        type_map = {EdgeType.SEMANTIC: 0, EdgeType.TEMPORAL: 1, EdgeType.VERSION: 2, EdgeType.IMPLICIT: 3}

        for n in nodes:
            u = node_map[n.id]
            for e in n.edges:
                if e.target in node_map:
                    edge_indices.append([u, node_map[e.target]])
                    edge_types.append(type_map[e.type])

        if not edge_indices: return

        # 2. GNN 推理
        with torch.no_grad():
            if HAS_PYG:
                edge_index = torch.tensor(edge_indices).t().contiguous()
                et = torch.tensor(edge_types)
                z = self.gnn(x, edge_index, et)
            else:
                z = x  # 无 GNN 环境下的 Fallback

        # 3. 链接预测 (Recall)
        # 简单计算新节点与所有节点的相似度
        new_idxs = [node_map[n.id] for n in new_nodes]
        for i in new_idxs:
            sims = torch.cosine_similarity(z[i].unsqueeze(0), z)
            top_vals, top_idxs = torch.topk(sims, k=3)

            for val, idx in zip(top_vals, top_idxs):
                j = idx.item()
                if i == j: continue

                # 4. LLM 验证 (Verify)
                if val > self.threshold and self._llm_verify(nodes[i], nodes[j]):
                    graph.add_edge(nodes[i].id, nodes[j].id, EdgeType.IMPLICIT, weight=val.item())

    def _llm_verify(self, n1, n2):
        # 模拟 LLM 验证通过
        return True