# -*- coding: UTF-8 -*-
# c2/builders/evolution.py
import logging
from typing import List

from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode

logger = logging.getLogger(__name__)

class EvolutionBuilder(BaseGraphBuilder):
    """
    [THESIS] Phase 2: 冲突检测与记忆演化
    """
    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client
        self.decay_factor = 0.5

    def process(self, new_nodes: List[MemoryNode], graph):
        all_nodes = graph.get_all_nodes()
        # [SIMPLIFIED] 只看最近的 10 个节点
        recent_nodes = all_nodes[-10:]

        for new_node in new_nodes:
            for old_node in recent_nodes:
                if new_node.node_id == old_node.node_id: continue
                # 只比较同类型
                if new_node.node_type != old_node.node_type: continue
                # 只比较 Profile (画像) 类，这类最容易冲突
                if "profile" not in new_node.category.value: continue

                if self._detect_conflict(new_node.content, old_node.content):
                    logger.info(f"    ⚔️ [Conflict] 冲突: '{new_node.content[:10]}' vs '{old_node.content[:10]}'")
                    graph.add_edge(new_node.node_id, old_node.node_id, EdgeType.VERSION)
                    old_node.energy_level *= self.decay_factor

    def _detect_conflict(self, text_a: str, text_b: str) -> bool:
        if not self.llm: return False
        prompt = f"""
        判断以下两句话是否存在【事实冲突】？
        1: {text_a}
        2: {text_b}
        冲突回答YES，否则NO。
        """
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = res.get("content", "").upper() if isinstance(res, dict) else str(res).upper()
            return "YES" in content
        except:
            return False