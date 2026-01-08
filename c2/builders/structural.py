# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：structural.py
@Author  ：niu
@Date    ：2026/1/8 13:26 
@Desc    ：
"""

# c2/builders/structural.py
import sys
import os
import torch
import torch.nn as nn
from typing import List

# === 路径设置：为了导入 model.py ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入 ZhipuChat
try:
    from general.model import ZhipuChat
except ImportError:
    sys.path.append("..")
    from model import ZhipuChat

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import LOGIC_VERIFICATION_PROMPT  # 导入提示词

# 尝试导入 PyG (可选)
try:
    from torch_geometric.nn import RGCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class MockGNN(nn.Module):
    """
    简易 GNN 模型
    在没有训练数据的情况下，暂时作为一个特征变换层
    """

    def __init__(self, in_dim, out_dim, num_rels):
        super().__init__()
        # 假设 Embedding 维度是 384 (MiniLM)
        # 如果安装了 PyG，这里可以初始化真实的 RGCN
        if HAS_PYG:
            self.conv1 = RGCNConv(in_dim, out_dim, num_rels)

    def forward(self, x, edge_index, edge_type):
        if HAS_PYG and edge_index.numel() > 0:
            return self.conv1(x, edge_index, edge_type)
        # 如果没有 PyG 或者没有边，直接返回原始特征 (降级处理)
        return x


class StructuralBuilder(BaseGraphBuilder):
    """
    [Phase 3] 符号侧 (Symbolic Side)
    逻辑：
    1. 结构推理: 利用 GNN (或向量相似度) 挖掘潜在关系。
    2. 语义验证: 将高置信度候选交给 LLM 判别逻辑关联。
    """

    def __init__(self):
        # 1. 初始化 GNN
        # input_dim=384 (all-MiniLM-L6-v2 的维度), hidden=64, relations=4
        self.gnn = MockGNN(384, 64, 4)

        # 设定阈值：只有相似度高于此值的才交给 LLM 验证 (节省 Token)
        self.threshold = 0.4

        # 2. 初始化 LLM (用于验证)
        config_path = os.path.join(project_root, "config/llm_config.yaml")
        if not os.path.exists(config_path):
            config_path = "./config/llm_config.yaml"

        print(f"  [Structural] Loading LLM for Link Verification...")
        self.llm = ZhipuChat(config_path)

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Structural] GNN 正在进行隐式推理 + LLM 逻辑验证...")

        nodes = graph.get_all_nodes()
        if not nodes: return
        node_map = {n.id: i for i, n in enumerate(nodes)}

        # === 1. 准备节点特征 (Embeddings) ===
        x_list = []
        for n in nodes:
            if n.embedding is not None and len(n.embedding) > 0:
                x_list.append(torch.tensor(n.embedding))
            else:
                # 兜底：如果 Semantic 步没生成 Embedding，用零向量或随机向量
                x_list.append(torch.zeros(384))

        if not x_list: return
        x = torch.stack(x_list)

        # === 2. GNN 前向传播 (Feature Propagation) ===
        # 这里为了简化代码，暂时略过复杂的 EdgeIndex 构建
        # 直接使用原始 Embedding 计算相似度 (相当于 GNN 的第 0 层)
        # 随着系统演进，这里会将 edge_index 传入 self.gnn(x, ...)
        z = x

        # === 3. 链接预测 (Recall) ===
        # 计算新节点与现有节点的相似度矩阵
        new_idxs = [node_map[n.id] for n in new_nodes if n.id in node_map]

        for i in new_idxs:
            # 计算余弦相似度
            sims = torch.cosine_similarity(z[i].unsqueeze(0), z)

            # 取前 3 个最相似的候选
            top_vals, top_idxs = torch.topk(sims, k=3)

            for val, idx in zip(top_vals, top_idxs):
                j = idx.item()
                if i == j: continue  # 跳过自己

                # 检查是否已经连过线了 (避免重复)
                source_node = nodes[i]
                target_node = nodes[j]
                already_linked = any(e.target == target_node.id for e in source_node.edges)
                if already_linked: continue

                # === 4. LLM 语义验证 (Verify) ===
                # 只有当相似度达标时，才花钱调 LLM
                if val > self.threshold:
                    print(
                        f"    🔍 [GNN Proposal] Score={val:.2f}: {source_node.content[:10]}... <-> {target_node.content[:10]}...")

                    if self._llm_verify(source_node, target_node):
                        print(f"      ✅ [LLM Verified] 建立隐式关联 (Implicit Link)")
                        graph.add_edge(source_node.id, target_node.id, EdgeType.IMPLICIT, weight=val.item())

    def _llm_verify(self, n1: GraphNode, n2: GraphNode) -> bool:
        """
        调用 LLM 验证两个节点是否存在逻辑关联
        """
        # 1. 组装 Prompt
        prompt = LOGIC_VERIFICATION_PROMPT.format(text_a=n1.content, text_b=n2.content)
        messages = [{"role": "user", "content": prompt}]

        # 2. 调用 LLM
        try:
            result = self.llm.chat(messages)
            if isinstance(result, dict):
                status = result.get("status", "REJECT").upper()
                return "PASS" in status
        except Exception as e:
            print(f"  [Structural] LLM Verify Error: {e}")

        return False