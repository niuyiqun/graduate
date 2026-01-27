# -*- coding: UTF-8 -*-
# c2/builders/semantic.py
import sys
import os
import json
import re
from typing import List, Set

# === 路径配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode  # [FIX] 引用 MemoryNode
from c2.graph_storage import MemoryGraph  # [FIX] 引用 MemoryGraph
from c2.prompts import ENTITY_EXTRACTION_PROMPT

# 导入配置
try:
    from c2.config import EMBEDDING_MODEL_PATH, LLM_CONFIG_PATH
except ImportError:
    # [SIMPLIFIED] 如果没有 config 文件，使用默认路径或占位符
    EMBEDDING_MODEL_PATH = "./model/all-MiniLM-L6-v2"
    LLM_CONFIG_PATH = "./config/llm_config.yaml"

# 导入 LLM
try:
    from general.model import QwenChat
except ImportError:
    pass  # 这里 pipeline 会传入 llm，所以 import 失败也没关系

# 导入 SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer

    HAS_EMBEDDING_MODEL = True
except ImportError:
    HAS_EMBEDDING_MODEL = False
    print("⚠️ sentence-transformers not found. Embeddings will be random.")


class BasicSemanticBuilder(BaseGraphBuilder):
    """
    [Phase 1] 语义侧 (Semantic Side) - 基础版
    职责:
    1. 提取实体 (Entities) -> 构建显式语义关联 (SEMANTIC Edge)
    2. 生成向量 (Embeddings) -> 为后续 GNN 准备特征
    """

    def __init__(self, llm_client=None):
        super().__init__()
        # [SIMPLIFIED] 生产环境可能需要独立的 LLM 实例，这里复用传入的 client
        # 如果没传，就尝试从 config 初始化 (但不建议，因为会多占显存)
        self.llm = llm_client

        # 2. 初始化 Embedding 模型 (用于生成节点特征)
        self.encoder = None
        if HAS_EMBEDDING_MODEL:
            print(f"  [Semantic] Loading Local Embedding Model: {EMBEDDING_MODEL_PATH}")
            if os.path.exists(EMBEDDING_MODEL_PATH):
                self.encoder = SentenceTransformer(EMBEDDING_MODEL_PATH)
            else:
                print(f"    ⚠️ Model path not found, downloading 'all-MiniLM-L6-v2'...")
                # [SIMPLIFIED] 自动下载小模型，方便跑通代码
                try:
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                except:
                    print("    ❌ Download failed. Embedding will be skipped.")

    def process(self, new_nodes: List[MemoryNode], graph: MemoryGraph):
        if not new_nodes: return
        print("  [Semantic] 正在处理 Embedding 和实体提取...")

        # 1. 批量/逐个生成 Embedding
        # [THESIS] 节点的向量化是 GNN (Phase 3) 的基础
        texts = [n.content for n in new_nodes]
        if self.encoder:
            try:
                embeddings = self.encoder.encode(texts)
                for node, emb in zip(new_nodes, embeddings):
                    node.embedding = emb.tolist()  # 转存为 list 方便序列化
            except Exception as e:
                print(f"    ⚠️ Embedding generation failed: {e}")

        # 2. 实体提取与连边 (如果是新节点)
        # 这里为了演示，我们只处理前 5 个节点，避免 API 调用过多
        for node in new_nodes[:5]:
            # A. 提取实体
            entities = self._extract_entities(node.content)
            # 将提取到的实体存入 meta 数据
            if not node.meta: node.meta = {}
            node.meta['entities'] = list(entities)

            # B. 寻找语义关联 (显式)
            # [SIMPLIFIED] 暴力遍历 O(N)，生产环境应使用倒排索引 (Inverted Index)
            all_nodes = graph.get_all_nodes()
            for existing_node in all_nodes:
                if existing_node.node_id == node.node_id: continue

                existing_entities = existing_node.meta.get('entities', [])
                if not existing_entities: continue

                # 计算实体交集
                intersection = set(entities) & set(existing_entities)
                if intersection:
                    # 建立双向语义边
                    # [THESIS] 这种基于实体的硬连接构成了图谱的“骨架”
                    graph.add_edge(node.node_id, existing_node.node_id, EdgeType.SEMANTIC, weight=len(intersection))
                    graph.add_edge(existing_node.node_id, node.node_id, EdgeType.SEMANTIC, weight=len(intersection))

    def _extract_entities(self, text: str) -> Set[str]:
        """调用 LLM 提取实体"""
        if not self.llm: return set()

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        # 适配 QwenChat 的接口
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages)
            # 解析 JSON
            if isinstance(response, dict):
                content = response.get("content", "[]")
            else:
                content = str(response)

            # [SIMPLIFIED] 简单的正则清洗，防止模型输出 Markdown
            content = re.sub(r'```json\s*|\s*```', '', content).strip()

            # 尝试找到列表的起止
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                content = content[start:end + 1]

            entities = json.loads(content)
            if isinstance(entities, dict) and "entities" in entities:
                return set(entities["entities"])
            return set(entities) if isinstance(entities, list) else set()
        except Exception as e:
            # print(f"    ⚠️ Entity extraction failed: {e}")
            return set()