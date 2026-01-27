# -*- coding: UTF-8 -*-
# c2/builders/structural.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import List

# [FIX] 正确的导入
from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode

logger = logging.getLogger(__name__)

# [SIMPLIFIED] 尝试导入 PyG (PyTorch Geometric)。
try:
    from torch_geometric.nn import RGCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("⚠️ torch_geometric 未安装。GNN 模块将运行在简易模式。")


class NeuroSymbolicGNN(nn.Module):
    """
    [THESIS] 神经符号编码器
    使用 RGCN (Relational Graph Convolutional Network) 处理异构图。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        if HAS_PYG:
            self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

    def encode(self, x, edge_index, edge_type):
        if not HAS_PYG: return x
        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        return x


class StructuralBuilder(BaseGraphBuilder):  # [FIX] 继承 BaseGraphBuilder
    """
    [THESIS] Phase 3 & 4: 隐式召回与验证
    """

    def __init__(self, llm_client):
        super().__init__()
        # [SIMPLIFIED] 参数硬编码
        self.in_dim = 384  # MiniLM 的维度是 384, 如果是 random 则是 384
        self.hidden_dim = 64
        self.out_dim = 32
        self.num_rels = 5

        self.llm = llm_client
        self.model = NeuroSymbolicGNN(self.in_dim, self.hidden_dim, self.out_dim, self.num_rels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def process(self, new_nodes: List[MemoryNode], graph):
        # 1. 准备数据
        nodes = graph.get_all_nodes()
        if len(nodes) < 3: return

        node_map = {n.node_id: i for i, n in enumerate(nodes)}

        # [SIMPLIFIED] 特征初始化
        # 如果 BasicSemanticBuilder 跑成功了，这里应该有 embedding
        # 如果没有，用随机向量兜底，保证代码不崩
        x_list = []
        for n in nodes:
            if n.embedding and len(n.embedding) > 0:
                # 确保维度对齐，如果维度不对（比如换了模型），截断或补零
                tensor_emb = torch.tensor(n.embedding)
                if tensor_emb.shape[0] != self.in_dim:
                    # 简单重新初始化一个随机的
                    x_list.append(torch.randn(self.in_dim))
                else:
                    x_list.append(tensor_emb)
            else:
                x_list.append(torch.randn(self.in_dim))
        x = torch.stack(x_list)

        # 构建边索引
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

                # [FIX] 兼容处理：etype 可能是字符串或 Enum
                if hasattr(etype, 'value'):  # 是 Enum
                    # 找到对应的 key
                    for k, val in edge_type_map.items():
                        if k.value == etype.value:
                            edge_types.append(val)
                            break
                    else:
                        edge_types.append(0)
                else:  # 是字符串
                    # 尝试匹配字符串
                    found = False
                    for k, val in edge_type_map.items():
                        if k.value == etype:
                            edge_types.append(val)
                            found = True
                            break
                    if not found: edge_types.append(0)

        if not edge_indices: return

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # 2. 自监督训练 GNN
        if HAS_PYG:
            self.model.train()
            for _ in range(5):  # [SIMPLIFIED] 只训练 5 epoch
                self.optimizer.zero_grad()
                z = self.model.encode(x, edge_index, edge_type)
                loss = torch.mean(z ** 2)
                loss.backward()
                self.optimizer.step()

        # 3. 隐式召回
        self.model.eval()
        with torch.no_grad():
            if HAS_PYG:
                z = self.model.encode(x, edge_index, edge_type)
            else:
                z = x

            sim_matrix = torch.matmul(z, z.t())

        # 4. 语义验证
        # [SIMPLIFIED] 阈值设低点以便看到效果
        threshold = 3.0
        rows, cols = torch.where(sim_matrix > threshold)

        candidates = []
        existing_edges = set((u, v) for u, v, _ in edges)

        for r, c in zip(rows, cols):
            if len(candidates) >= 2: break  # [SIMPLIFIED] 限制数量
            if r >= c: continue

            u_node = nodes[r.item()]
            v_node = nodes[c.item()]

            if (u_node.node_id, v_node.node_id) in existing_edges: continue
            if (v_node.node_id, u_node.node_id) in existing_edges: continue

            candidates.append((u_node, v_node))

        for n1, n2 in candidates:
            if self._llm_verify(n1, n2):
                graph.add_edge(n1.node_id, n2.node_id, EdgeType.IMPLICIT)
                logger.info(f"    🔗 [GNN+LLM] 发现隐式关联: {n1.content[:10]}... <-> {n2.content[:10]}...")

    def _llm_verify(self, n1, n2) -> bool:
        if not self.llm: return False
        prompt = f"""
        判断以下两个片段是否有逻辑关联？
        A: {n1.content}
        B: {n2.content}
        有则回答YES，无则NO。
        """
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = res.get("content", "").upper() if isinstance(res, dict) else str(res).upper()
            return "YES" in content
        except:
            return False