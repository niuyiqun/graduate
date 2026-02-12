"""
c2.graph_kernel - 图数据库接口层
"""
import networkx as nx
import pickle
from .data_models import MemoryNode, EdgeType


class GraphKernel:
    def __init__(self):
        # 使用 MultiDiGraph 以支持两个节点间存在多种类型的边
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: MemoryNode):
        """添加节点，如果存在则更新"""
        if not self.graph.has_node(node.node_id):
            self.graph.add_node(node.node_id, data=node)
        else:
            # 更新属性
            self.graph.nodes[node.node_id]['data'] = node

    def add_edge(self, u_id: str, v_id: str, edge_type: EdgeType, weight: float = 1.0):
        """添加有向边"""
        self.graph.add_edge(u_id, v_id, type=edge_type, weight=weight)

    def get_nodes_by_type(self, node_type) -> list:
        return [
            data['data']
            for _, data in self.graph.nodes(data=True)
            if data['data'].node_type == node_type
        ]

    def get_stats(self):
        return f"Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}"

    def save_to_disk(self, path):
        # 序列化保存
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)