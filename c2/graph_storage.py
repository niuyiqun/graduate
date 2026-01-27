import networkx as nx
import json
import logging
import os
from typing import List, Dict, Any, Tuple
from c2.definitions import MemoryNode, EdgeType, NodeType, AtomCategory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self):
        # 使用 MultiDiGraph 因为两个节点间可能存在多种类型的边
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: MemoryNode):
        """添加节点到图中"""
        self.graph.add_node(
            node.node_id,
            content=node.content,
            category=node.category.value,
            type=node.node_type.value,
            timestamp=node.timestamp,
            obj=node  # 存储完整对象引用，方便内存中访问
        )

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0, meta: Dict = None):
        """添加边"""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(f"Cannot add edge {edge_type}: Node {source_id} or {target_id} not found.")
            return

        attr = {
            "type": edge_type.value,
            "weight": weight,
            "created_at": str(meta.get('created_at', '')) if meta else ''
        }
        if meta:
            attr.update(meta)

        self.graph.add_edge(source_id, target_id, **attr)
        # logger.debug(f"Added Edge: {source_id} --[{edge_type.value}]--> {target_id}")

    def get_node(self, node_id: str) -> MemoryNode:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]['obj']
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> List[MemoryNode]:
        """获取特定类型的节点"""
        nodes = []
        for n_id, data in self.graph.nodes(data=True):
            if data.get("type") == node_type.value:
                nodes.append(data["obj"])
        return nodes

    def get_nodes_sorted_by_time(self, node_type: NodeType = None) -> List[MemoryNode]:
        """按时间戳获取排序后的节点（用于构建时序链）"""
        nodes = []
        for n_id, data in self.graph.nodes(data=True):
            if node_type is None or data.get("type") == node_type.value:
                nodes.append(data["obj"])

        # 按时间戳排序
        return sorted(nodes, key=lambda x: x.timestamp)

    def save_graph(self, filepath: str):
        """保存图结构为 JSON (用于可视化)"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = nx.node_link_data(self.graph)
        # 清理无法序列化的对象 (obj)
        # 注意：这里我们做深拷贝或者在序列化前移除，序列化后再恢复（如果还需要用）
        # 简单起见，我们生成一个用于保存的副本数据

        save_data = {
            "directed": data["directed"],
            "multigraph": data["multigraph"],
            "graph": data["graph"],
            "nodes": [],
            "links": data["links"]
        }

        for node in data['nodes']:
            node_copy = node.copy()
            if 'obj' in node_copy:
                del node_copy['obj']  # 移除对象引用
            save_data['nodes'].append(node_copy)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Graph saved to {filepath} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def load_graph(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        # 注意：从 JSON 加载回来时，'obj' 字段丢失了。
        # 如果需要恢复 obj 对象，需要遍历 nodes 并重新实例化 MemoryNode。
        # 暂时用于可视化目的，这步先略过。