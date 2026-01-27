# -*- coding: UTF-8 -*-
# c2/builders/temporal.py

from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, NodeType, AtomCategory, MemoryNode


class TemporalBuilder(BaseGraphBuilder):
    """
    Phase 1.2: 构建时间主干
    逻辑: 将 Activity 节点按时间戳串联。
    """

    def process(self, new_nodes, graph):
        # 获取全图所有的 Episodic Activity 节点，并按时间排序
        all_episodic = graph.get_nodes_sorted_by_time(NodeType.EPISODIC)
        activities = [n for n in all_episodic if n.category == AtomCategory.ACTIVITY]

        if len(activities) < 2: return

        count = 0
        for i in range(len(activities) - 1):
            current_node = activities[i]
            next_node = activities[i + 1]

            # 简单的判重：检查是否已有边
            if not graph.graph.has_edge(current_node.node_id, next_node.node_id):
                graph.add_edge(
                    current_node.node_id,
                    next_node.node_id,
                    EdgeType.TEMPORAL,
                    weight=1.0
                )
                count += 1

        if count > 0:
            print(f"  ⏳ [Temporal] Linked {count} time-steps")