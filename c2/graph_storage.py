# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：graph_storage.py.py
@Author  ：niu
@Date    ：2026/1/8 13:24 
@Desc    ：
"""

# c2/graph_storage.py
from typing import Dict, List
from .definitions import GraphNode, MemoryEdge, EdgeType, DecoupledMemoryAtom

class AtomGraph:
    """
    内存记忆图谱容器
    """
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        # 邻接表索引: SourceID -> List[Edge]
        self.adjacency: Dict[str, List[MemoryEdge]] = {}

    def add_atom(self, atom: DecoupledMemoryAtom) -> GraphNode:
        """标准化接入"""
        if atom.id not in self.nodes:
            node = GraphNode(atom)
            self.nodes[atom.id] = node
            self.adjacency[atom.id] = []
        return self.nodes[atom.id]

    def add_edge(self, source_id: str, target_id: str, type: EdgeType, weight: float = 1.0, **kwargs):
        """建立连接"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return

        # 简单查重
        for edge in self.adjacency[source_id]:
            if edge.target == target_id and edge.type == type:
                return

        new_edge = MemoryEdge(source_id, target_id, type, weight, metadata=kwargs)
        self.nodes[source_id].edges.append(new_edge)
        self.adjacency[source_id].append(new_edge)

    def get_all_nodes(self) -> List[GraphNode]:
        return list(self.nodes.values())