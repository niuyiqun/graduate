# -*- coding: UTF-8 -*-
# c2/builders/emergence.py

import logging
import uuid
import networkx as nx
from c2.builders.base import BaseGraphBuilder
from c2.definitions import NodeType, AtomCategory, EdgeType, MemoryNode
from c2.prompts import CONCEPT_ABSTRACTION_PROMPT


class EmergenceBuilder(BaseGraphBuilder):
    """
    Phase 5: 结构诱导的概念涌现
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def process(self, new_nodes, graph):
        self.build_emergence(graph)

    def build_emergence(self, graph):
        nx_graph = graph.get_nx_graph()

        # 只在 Activity 层面上寻找模式
        activity_nodes = [n for n, attr in nx_graph.nodes(data=True)
                          if attr.get('type') == NodeType.EPISODIC.value]

        if len(activity_nodes) < 4: return

        # 构建无向子图
        subgraph = nx_graph.subgraph(activity_nodes).to_undirected()

        # 社区发现 (优先使用 Louvain, 降级使用 Label Prop)
        try:
            communities = nx.community.louvain_communities(subgraph)
        except:
            try:
                communities = nx.community.label_propagation_communities(subgraph)
            except:
                return

        emerged_count = 0
        for comm in communities:
            if len(comm) < 3: continue  # 忽略小簇

            # 收集内容
            contents = []
            ids = []
            for node_id in comm:
                node = graph.get_node(node_id)
                if node:
                    contents.append(f"- {node.content}")
                    ids.append(node_id)

            context_str = "\n".join(contents[:10])  # 限制长度

            # LLM 抽象
            abstract_content = self._abstract(context_str)
            if not abstract_content: continue

            # 回写图谱
            new_id = f"concept_emerged_{uuid.uuid4().hex[:6]}"
            new_node = MemoryNode(
                node_id=new_id,
                content=abstract_content,
                category=AtomCategory.PROFILE,
                node_type=NodeType.CONCEPTUAL,
                meta={"source": "emergence"}
            )
            graph.add_node(new_node)
            emerged_count += 1
            print(f"    ✨ [Emerged] New Concept: {abstract_content[:30]}...")

            # 连线
            for eid in ids:
                graph.add_edge(new_id, eid, EdgeType.ABSTRACT)

        if emerged_count > 0:
            print(f"  🆙 [Emergence] Created {emerged_count} high-level concepts")

    def _abstract(self, context_str):
        if not self.llm: return None
        prompt = CONCEPT_ABSTRACTION_PROMPT.format(context_str=context_str)
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = str(res.get("content", "")).strip()
            if len(content) > 5 and "无法" not in content:
                return content
            return None
        except:
            return None