import networkx as nx
from typing import List, Tuple
from .definitions import GraphNode, EdgeType


class MemoryGraph:
    def __init__(self):
        # 使用 NetworkX 作为内存图数据库
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: GraphNode):
        self.graph.add_node(node.id, data=node)

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0):
        self.graph.add_edge(source_id, target_id, type=edge_type, weight=weight)

    def get_neighbors(self, node_id: str, edge_types: List[EdgeType] = None):
        # 根据边类型过滤邻居
        neighbors = []
        if node_id not in self.graph:
            return []

        for neighbor, edge_data in self.graph[node_id].items():
            # edge_data 是一个字典，包含多条边 (0, 1, ...)
            for _, attr in edge_data.items():
                if edge_types is None or attr['type'] in edge_types:
                    neighbors.append((neighbor, attr['type'], attr['weight']))
        return neighbors

    def get_all_nodes(self) -> List[GraphNode]:
        return [data['data'] for _, data in self.graph.nodes(data=True)]