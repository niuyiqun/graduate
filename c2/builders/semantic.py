# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：semantic.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""
# c2/builders/semantic.py
import sys
import os
import json
import re
from typing import List, Set

# === 路径与配置导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入配置
try:
    from ..config import EMBEDDING_MODEL_PATH, LLM_CONFIG_PATH
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import EMBEDDING_MODEL_PATH, LLM_CONFIG_PATH

# 导入 LLM
try:
    from general.model import ZhipuChat
except ImportError:
    from model import ZhipuChat

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import ENTITY_EXTRACTION_PROMPT

# 导入 SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer

    HAS_EMBEDDING_MODEL = True
except ImportError:
    HAS_EMBEDDING_MODEL = False
    print("⚠️ sentence-transformers not found. Embeddings will be random.")


class SemanticBuilder(BaseGraphBuilder):
    """
    [Phase 1] 语义侧 (Semantic Side)
    职责:
    1. 提取实体 (Entities) -> 构建显式语义关联
    2. 生成向量 (Embeddings) -> 为后续 GNN 准备特征
    """

    def __init__(self):
        # 1. 初始化 LLM (用于实体提取)
        print(f"  [Semantic] Loading LLM from config: {LLM_CONFIG_PATH}")
        self.llm = ZhipuChat(LLM_CONFIG_PATH)

        # 2. 初始化 Embedding 模型 (用于生成节点特征)
        self.encoder = None
        if HAS_EMBEDDING_MODEL:
            print(f"  [Semantic] Loading Local Embedding Model: {EMBEDDING_MODEL_PATH}")
            if os.path.exists(EMBEDDING_MODEL_PATH):
                self.encoder = SentenceTransformer(EMBEDDING_MODEL_PATH)
            else:
                print(f"    ⚠️ Model path not found, downloading 'all-MiniLM-L6-v2'...")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Semantic] 正在利用 LLM 提取实体并构建骨架...")

        # 1. 批量/逐个生成 Embedding
        texts = [n.content for n in new_nodes]
        if self.encoder:
            embeddings = self.encoder.encode(texts)
            for node, emb in zip(new_nodes, embeddings):
                node.embedding = emb.tolist()  # 转存为 list 方便序列化

        # 2. 实体提取与连边
        for node in new_nodes:
            # A. 提取实体
            entities = self._extract_entities(node.content)
            node.entities = list(entities)

            # B. 寻找语义关联 (显式)
            # 遍历图中已有节点，如果实体有交集，则连边
            all_nodes = graph.get_all_nodes()
            for existing_node in all_nodes:
                if existing_node.id == node.id: continue

                # 计算实体交集
                intersection = set(node.entities) & set(existing_node.entities)
                if intersection:
                    # 建立双向语义边
                    graph.add_edge(node.id, existing_node.id, EdgeType.SEMANTIC, weight=len(intersection))
                    graph.add_edge(existing_node.id, node.id, EdgeType.SEMANTIC, weight=len(intersection))

    def _extract_entities(self, text: str) -> Set[str]:
        """调用 LLM 提取实体"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages)
            # 解析 JSON
            if isinstance(response, dict):
                content = response.get("content", "[]")
            else:
                content = str(response)

            # 清理 Markdown 代码块
            content = re.sub(r'```json\s*|\s*```', '', content).strip()

            entities = json.loads(content)
            return set(entities)
        except Exception as e:
            print(f"    ⚠️ Entity extraction failed: {e}")
            return set()