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
from typing import List, Set

# === 路径设置 ===
# 获取当前文件的上上上级目录 (即 graduate 根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入 ZhipuChat
try:
    from general.model import ZhipuChat
except ImportError:
    sys.path.append("..")
    from model import ZhipuChat

# 导入 SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import ENTITY_EXTRACTION_PROMPT


class SemanticBuilder(BaseGraphBuilder):
    def __init__(self):
        # 1. 初始化 LLM (ZhipuChat)
        config_path = os.path.join(project_root, "config/llm_config.yaml")
        if not os.path.exists(config_path):
            config_path = "./config/llm_config.yaml"

        print(f"  [Semantic] Loading LLM from config: {config_path}")
        self.llm = ZhipuChat(config_path)

        # 2. 初始化本地 Embedding 模型
        # 目标路径: graduate/model/all-MiniLM-L6-v2
        local_model_path = os.path.join(project_root, "model", "all-MiniLM-L6-v2")

        self.embed_model = None
        if SentenceTransformer:
            if os.path.exists(local_model_path):
                print(f"  [Semantic] Loading Local Embedding Model: {local_model_path}")
                self.embed_model = SentenceTransformer(local_model_path)
            else:
                print(f"⚠️ [Warning] Local model not found at {local_model_path}")
                print("  Trying to download from Hugging Face instead...")
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("❌ [Error] sentence_transformers library not installed.")

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Semantic] 正在利用 LLM 提取实体并构建骨架...")
        existing_nodes = graph.get_all_nodes()

        for node in new_nodes:
            # === A. 生成 Embedding (使用本地模型) ===
            if node.embedding is None and self.embed_model:
                try:
                    # encode 返回 numpy array，需转 list
                    node.embedding = self.embed_model.encode(node.content).tolist()
                except Exception as e:
                    print(f"  Embedding Error: {e}")

            # === B. 真实 LLM 提取实体 ===
            if not node.entities:
                node.entities = self._llm_extract(node.content)

            if not node.entities: continue

            # === C. 连线逻辑 ===
            for other in existing_nodes:
                if node.id == other.id: continue
                shared = node.entities.intersection(other.entities)
                if shared:
                    w = len(shared) * 1.0
                    graph.add_edge(node.id, other.id, EdgeType.SEMANTIC, weight=w)
                    graph.add_edge(other.id, node.id, EdgeType.SEMANTIC, weight=w)

    def _llm_extract(self, text: str) -> Set[str]:
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat(messages)
            if isinstance(result, dict) and "entities" in result:
                return set(result["entities"])
        except Exception as e:
            print(f"LLM Extract Error: {e}")

        return set()