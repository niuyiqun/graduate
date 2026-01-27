# -*- coding: UTF-8 -*-
# c2/builders/semantic.py

import sys
import os
import json
import re
from typing import List, Set
from sentence_transformers import SentenceTransformer

# 导入基础组件
from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode
from c2.graph_storage import MemoryGraph
from c2.prompts import ENTITY_EXTRACTION_PROMPT


class BasicSemanticBuilder(BaseGraphBuilder):
    """
    [Phase 1] 语义侧 (Semantic Side)
    1. 确保节点拥有 Embedding (优先复用，缺失补算)
    2. 提取实体并建立显式语义连接
    """

    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client
        self.encoder = None
        self.model_path = "model/all-MiniLM-L6-v2"

        # 预加载模型，以防万一需要补算
        if os.path.exists(self.model_path):
            self.encoder = SentenceTransformer(self.model_path)
        else:
            # 如果本地没有，尝试下载
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass

    def process(self, new_nodes: List[MemoryNode], graph: MemoryGraph):
        if not new_nodes: return

        # === 1. Embedding 检查与补算 ===
        nodes_to_compute = []
        reused_count = 0

        for node in new_nodes:
            # 🔥 优先复用 C1 传入的向量
            if node.embedding and len(node.embedding) > 0:
                reused_count += 1
            else:
                nodes_to_compute.append(node)

        if nodes_to_compute and self.encoder:
            try:
                texts = [n.content for n in nodes_to_compute]
                embeddings = self.encoder.encode(texts)
                for node, emb in zip(nodes_to_compute, embeddings):
                    node.embedding = emb.tolist()
            except Exception as e:
                print(f"    ❌ [Semantic] 向量补算失败: {e}")

        print(f"  🧠 [Semantic] Embedding Ready: {reused_count} Reused, {len(nodes_to_compute)} Computed")

        # === 2. 实体提取与连边 ===
        entity_hit = 0
        for node in new_nodes:
            # 如果 meta 里还没实体，就提取
            if not node.meta.get('entities'):
                entities = self._extract_entities(node.content)
                node.meta['entities'] = list(entities)

            if node.meta['entities']:
                entity_hit += 1

            # 寻找共现实体并连边
            current_entities = set(node.meta['entities'])
            if not current_entities: continue

            all_nodes = graph.get_all_nodes()
            for existing_node in all_nodes:
                if existing_node.node_id == node.node_id: continue

                existing_entities = set(existing_node.meta.get('entities', []))
                intersection = current_entities & existing_entities

                if intersection:
                    graph.add_edge(node.node_id, existing_node.node_id, EdgeType.SEMANTIC, weight=len(intersection))
                    graph.add_edge(existing_node.node_id, node.node_id, EdgeType.SEMANTIC, weight=len(intersection))

        print(f"  🏷️ [Semantic] Entities extracted for {entity_hit} nodes")

    def _extract_entities(self, text: str) -> Set[str]:
        if not self.llm: return set()

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = str(res.get("content", "")) if isinstance(res, dict) else str(res)

            # 清理 Markdown
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                content = content[start:end + 1]

            entities = json.loads(content)
            if isinstance(entities, dict) and "entities" in entities:
                return set(entities["entities"])
            elif isinstance(entities, list):
                return set(entities)
            return set()
        except:
            return set()