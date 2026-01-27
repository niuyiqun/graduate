# -*- coding: UTF-8 -*-
# c2/graph_storage.py

import networkx as nx
import json
import logging
import os
from typing import List, Dict, Any, Tuple
from c2.definitions import MemoryNode, EdgeType, NodeType, AtomCategory

logger = logging.getLogger(__name__)


class MemoryGraph:
    """
    基于 NetworkX 的内存图存储封装
    维护了 MemoryNode 对象与图结构的映射
    """

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
            # 关键：存储完整对象引用，方便后续 Builder 直接操作对象的 embedding 等属性
            obj=node
        )

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0, meta: Dict = None):
        """添加边"""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            # logger.warning(f"Cannot add edge {edge_type}: Node {source_id} or {target_id} not found.")
            return

        attr = {
            "type": edge_type.value,
            "weight": weight,
            "created_at": str(meta.get('created_at', '')) if meta else ''
        }
        if meta:
            attr.update(meta)

        self.graph.add_edge(source_id, target_id, **attr)

    def get_node(self, node_id: str) -> MemoryNode:
        """根据 ID 获取节点对象"""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id].get('obj')
        return None

    # === 🔥 [新增] 缺失的方法 🔥 ===

    def get_all_nodes(self) -> List[MemoryNode]:
        """获取图中所有节点对象 (修复 AttributeError)"""
        nodes = []
        for _, data in self.graph.nodes(data=True):
            if 'obj' in data:
                nodes.append(data['obj'])
        return nodes

    def get_all_edges(self) -> List[Tuple[str, str, Dict]]:
        """获取所有边 (u, v, attr)，用于 StructuralBuilder"""
        return list(self.graph.edges(data=True))

    def get_nx_graph(self) -> nx.MultiDiGraph:
        """获取底层的 NetworkX 图对象，用于 EmergenceBuilder 的聚类算法"""
        return self.graph

    # ================================

    def get_nodes_by_type(self, node_type: NodeType) -> List[MemoryNode]:
        """获取特定类型的节点"""
        nodes = []
        for _, data in self.graph.nodes(data=True):
            if data.get("type") == node_type.value:
                nodes.append(data["obj"])
        return nodes

    def get_nodes_sorted_by_time(self, node_type: NodeType = None) -> List[MemoryNode]:
        """按时间戳获取排序后的节点（用于构建时序链）"""
        nodes = []
        for _, data in self.graph.nodes(data=True):
            if node_type is None or data.get("type") == node_type.value:
                if 'obj' in data:
                    nodes.append(data["obj"])

        # 按时间戳排序 (处理 None timestamp 的情况)
        return sorted(nodes, key=lambda x: x.timestamp if x.timestamp else 0)

    def save_graph(self, filepath: str):
        """保存图结构为 JSON (用于可视化)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 使用 networkx 自带的序列化
        data = nx.node_link_data(self.graph)

        # 清理无法被 JSON 序列化的 Python 对象 ('obj')
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
                del node_copy['obj']  # 移除对象引用，只保留基本数据
            save_data['nodes'].append(node_copy)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Graph saved to {filepath} (Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()})")

    def load_graph(self, filepath: str):
        """加载图"""
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        # 注意：从 JSON 加载回来时 'obj' 对象会丢失，仅用于可视化或图结构分析