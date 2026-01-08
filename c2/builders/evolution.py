# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：evolution.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""

# c2/builders/evolution.py
from typing import List
from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph


class EvolutionBuilder(BaseGraphBuilder):
    """
    [Phase 2] 演化侧 (Evolution Side)
    检测冲突，建立版本演进边。
    """

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Evolution] 正在检测冲突...")
        all_nodes = graph.get_all_nodes()

        for new_node in new_nodes:
            # 1. 模拟向量检索找相似节点
            candidates = all_nodes[:5]

            for old_node in candidates:
                if new_node.id == old_node.id: continue

                # 2. 模拟 NLI 冲突检测
                if self._check_conflict(new_node, old_node):
                    print(f"    -> 冲突检测: {old_node.content} vs {new_node.content}")
                    # 3. 建立版本边 (Old -> New)
                    graph.add_edge(old_node.id, new_node.id, EdgeType.VERSION)
                    # 4. 旧节点降权
                    old_node.activation *= 0.5

    def _check_conflict(self, n1, n2):
        # 简单模拟：如果有共享实体但长度差异巨大，视为冲突
        shared = n1.entities.intersection(n2.entities)
        if shared and abs(len(n1.content) - len(n2.content)) > 20:
            return True
        return False