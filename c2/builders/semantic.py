# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：semantic.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""
# c2/builders/semantic.py
from typing import List, Set
from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph


class SemanticBuilder(BaseGraphBuilder):
    """
    [Phase 1] 神经侧 (Neural Side)
    利用 LLM 语义理解提取实体，构建显式语义关联。
    """

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Semantic] 正在提取实体并构建骨架...")
        existing_nodes = graph.get_all_nodes()

        for node in new_nodes:
            # 1. 实体提取 (模拟 LLM)
            if not node.entities:
                node.entities = self._llm_extract(node.content)

            if not node.entities: continue

            # 2. 实体共现连线
            for other in existing_nodes:
                if node.id == other.id: continue

                shared = node.entities.intersection(other.entities)
                if shared:
                    w = len(shared) * 1.0
                    # 语义通常双向
                    graph.add_edge(node.id, other.id, EdgeType.SEMANTIC, weight=w)
                    graph.add_edge(other.id, node.id, EdgeType.SEMANTIC, weight=w)

    def _llm_extract(self, text: str) -> Set[str]:
        # TODO: 接入真正的大模型
        # 这里用简单规则模拟：提取首字母大写的词
        entities = set()
        for word in text.replace(".", "").split():
            if word and word[0].isupper() and len(word) > 1:
                entities.add(word)
        return entities
