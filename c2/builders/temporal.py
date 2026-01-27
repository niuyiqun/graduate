# -*- coding: UTF-8 -*-
# c2/builders/temporal.py
from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, NodeType, AtomCategory, MemoryNode
import logging

logger = logging.getLogger(__name__)


class TemporalBuilder(BaseGraphBuilder):
    """
    Phase 1.2: 构建时间主干
    """

    def process(self, new_nodes, graph):
        # 1. 获取所有的 Episodic Activity 节点
        # 注意：这里我们重新获取整个图的按时间排序的节点，确保顺序正确
        all_episodic = graph.get_nodes_sorted_by_time(NodeType.EPISODIC)
        activities = [n for n in all_episodic if n.category == AtomCategory.ACTIVITY]

        if len(activities) < 2:
            return

        # 2. 按顺序连接 T -> T+1
        # [SIMPLIFIED] 简单的线性连接
        # 实际逻辑应检查是否已有边，避免重复添加
        for i in range(len(activities) - 1):
            current_node = activities[i]
            next_node = activities[i + 1]

            # 简单的判重逻辑：检查是否已有 TEMPORAL 边
            has_edge = False
            # 这里 graph.graph 是 NetworkX 对象
            if graph.graph.has_edge(current_node.node_id, next_node.node_id):
                # 检查边类型
                edge_data = graph.graph.get_edge_data(current_node.node_id, next_node.node_id)
                # NetworkX 的 MultiGraph 返回的是 dict of dicts {key: {attr}}
                for key, attr in edge_data.items():
                    if attr.get('type') == EdgeType.TEMPORAL.value:
                        has_edge = True
                        break

            if not has_edge:
                graph.add_edge(
                    current_node.node_id,
                    next_node.node_id,
                    EdgeType.TEMPORAL,
                    weight=1.0
                )