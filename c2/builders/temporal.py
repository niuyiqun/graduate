from c2.builders.base import GraphBuilder
from c2.definitions import EdgeType, NodeType, AtomCategory
import logging

logger = logging.getLogger(__name__)


class TemporalBuilder(GraphBuilder):
    """
    Phase 1.2: 构建时间主干
    逻辑: 仅将 Activity 节点按时间戳串联，形成纯净的时间流。
    Edge: TEMPORAL
    """

    def build(self):
        # 1. 获取所有的 Episodic Activity 节点
        all_episodic = self.graph.get_nodes_sorted_by_time(NodeType.EPISODIC)
        activities = [n for n in all_episodic if n.category == AtomCategory.ACTIVITY]

        if not activities:
            logger.warning("No activities found to build temporal chain.")
            return

        logger.info(f"Building Temporal Chain for {len(activities)} activities...")

        count = 0
        # 2. 按顺序连接 T -> T+1
        for i in range(len(activities) - 1):
            current_node = activities[i]
            next_node = activities[i + 1]

            # 计算时间差 (此处为简化版，未来可加入时间衰减计算)
            time_diff = next_node.timestamp - current_node.timestamp

            self.graph.add_edge(
                current_node.node_id,
                next_node.node_id,
                EdgeType.TEMPORAL,
                weight=1.0,
                meta={"time_diff": time_diff}
            )
            count += 1

        logger.info(f"Constructed {count} TEMPORAL edges.")


class BasicSemanticBuilder(GraphBuilder):
    """
    Phase 1.1: 基础语义连接
    逻辑: 将 episodic_thought（意图）与其对应的 episodic_activity（行为）通过 SEMANTIC 边强关联。
    """

    def build(self):
        thoughts = [n for n in self.graph.get_nodes_by_type(NodeType.EPISODIC)
                    if n.category == AtomCategory.THOUGHT]
        activities = [n for n in self.graph.get_nodes_by_type(NodeType.EPISODIC)
                      if n.category == AtomCategory.ACTIVITY]

        # 建立简单的查找表 (Activity by turn_id)
        # 假设 C1 输出的 metadata 里包含 'turn_id' 或类似标识
        # 如果没有，则退化为时间戳临近匹配

        act_by_turn = {}
        for act in activities:
            # 优先使用 metadata 中的关联 ID，其次是 turn_id
            key = act.meta.get('turn_id')
            if key:
                if key not in act_by_turn: act_by_turn[key] = []
                act_by_turn[key].append(act)

        count = 0
        for thought in thoughts:
            t_turn = thought.meta.get('turn_id')
            if not t_turn or t_turn not in act_by_turn:
                continue

            # 连接同一个 Turn 下的 Thought 和 Activity
            related_acts = act_by_turn[t_turn]
            for act in related_acts:
                self.graph.add_edge(
                    thought.node_id,
                    act.node_id,
                    EdgeType.SEMANTIC,
                    weight=2.0,  # 强关联：意图驱动行为
                    meta={"relation": "intention_of"}
                )
                count += 1

        logger.info(f"Constructed {count} SEMANTIC edges (Thought <-> Activity).")