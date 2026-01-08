# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：temporal.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""

# c2/builders/temporal.py
from typing import List
from datetime import datetime
from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph


class TemporalBuilder(BaseGraphBuilder):
    """
    [Phase 1] 规则侧 (Rule Side)
    构建时间流顺序。
    """

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        if not new_nodes: return

        # 解析时间字符串
        def parse(n):
            try:
                return datetime.strptime(n.timestamp, "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.min

        sorted_nodes = sorted(new_nodes, key=parse)

        # 建立线性连接
        for i in range(len(sorted_nodes) - 1):
            src, tgt = sorted_nodes[i], sorted_nodes[i + 1]
            graph.add_edge(src.id, tgt.id, EdgeType.TEMPORAL)