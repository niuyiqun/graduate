# -*- coding: UTF-8 -*-
# c2/builders/structural.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode
from c2.prompts import LOGIC_VERIFICATION_PROMPT

# 尝试导入 PyG
try:
    from torch_geometric.nn import RGCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NeuroSymbolicGNN(nn.Module):
    """
    [THESIS] RGCN 模型定义
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        # 哑参数
        self.dummy = nn.Parameter(torch.empty(0))

        if HAS_PYG:
            self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

    def encode(self, x, edge_index, edge_type):
        if not HAS_PYG: return x  # Fallback: 直接返回原始 Embedding

        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        return x


class StructuralBuilder(BaseGraphBuilder):
    """
    Phase 3/4: 隐式召回
    """

    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client
        self.in_dim = 384  # MiniLM 维度
        self.hidden_dim = 64
        self.out_dim = 32
        self.num_rels = 5
        self.model = NeuroSymbolicGNN(self.in_dim, self.hidden_dim, self.out_dim, self.num_rels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def process(self, new_nodes, graph):
        nodes = graph.get_all_nodes()
        if len(nodes) < 3: return

        node_map = {n.node_id: i for i, n in enumerate(nodes)}

        # 1. 准备 Tensor 数据
        x_list = []
        for n in nodes:
            # 确保有 Embedding，没有则补零 (Real Logic)
            if n.embedding and len(n.embedding) == self.in_dim:
                x_list.append(torch.tensor(n.embedding))
            else:
                x_list.append(torch.zeros(self.in_dim))
        x = torch.stack(x_list)

        # 2. 准备边数据
        edges = graph.get_all_edges()
        edge_indices = []
        edge_types = []

        edge_type_map = {
            EdgeType.SEMANTIC: 0, EdgeType.TEMPORAL: 1,
            EdgeType.VERSION: 2, EdgeType.IMPLICIT: 3, EdgeType.ABSTRACT: 4
        }

        for u, v, attr in edges:
            if u in node_map and v in node_map:
                edge_indices.append([node_map[u], node_map[v]])
                etype = attr.get('type', EdgeType.SEMANTIC)

                val = 0
                if hasattr(etype, 'value'):  # Enum
                    for k, v_idx in edge_type_map.items():
                        if k.value == etype.value:
                            val = v_idx;
                            break
                edge_types.append(val)

        # 3. 训练/推断
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types, dtype=torch.long)

            # 训练 (仅当有 PyG 时)
            if HAS_PYG:
                self.model.train()
                for _ in range(5):
                    self.optimizer.zero_grad()
                    z = self.model.encode(x, edge_index, edge_type)
                    loss = torch.mean(z ** 2)  # 简化 Loss
                    loss.backward()
                    self.optimizer.step()

            # 推断 Embedding (结构化或原始)
            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(x, edge_index, edge_type)
        else:
            z = x  # 无边时直接用 Content Embedding

        # 4. 计算相似度 (Cosine Similarity via Dot Product)
        # 归一化以计算 Cosine
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.t())

        # 5. 筛选 Top-K 隐式关联
        threshold = 0.85  # Cosine 阈值
        rows, cols = torch.where(sim_matrix > threshold)

        candidates = []
        existing_edges = set((u, v) for u, v, _ in edges)

        for r, c in zip(rows, cols):
            if len(candidates) >= 2: break  # 限制验证数量
            if r >= c: continue

            u_id, v_id = nodes[r.item()].node_id, nodes[c.item()].node_id

            if (u_id, v_id) not in existing_edges and (v_id, u_id) not in existing_edges:
                candidates.append((nodes[r.item()], nodes[c.item()]))

        # 6. LLM 验证
        added_count = 0
        for n1, n2 in candidates:
            if self._llm_verify(n1.content, n2.content):
                graph.add_edge(n1.node_id, n2.node_id, EdgeType.IMPLICIT)
                added_count += 1
                print(f"    🔗 [Implicit] Linked: {n1.content[:10]}... <-> {n2.content[:10]}...")

        if added_count > 0:
            print(f"  🕸️ [Structural] Added {added_count} implicit edges")

    def _llm_verify(self, text_a, text_b):
        if not self.llm: return False
        prompt = LOGIC_VERIFICATION_PROMPT.format(text_a=text_a, text_b=text_b)
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = str(res.get("content", "")).strip().upper()
            return content == "YES"
        except:
            return False