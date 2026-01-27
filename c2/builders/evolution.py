# -*- coding: UTF-8 -*-
# c2/builders/evolution.py

from c2.builders.base import BaseGraphBuilder
from c2.definitions import EdgeType, MemoryNode
from c2.prompts import CONFLICT_DETECTION_PROMPT


class EvolutionBuilder(BaseGraphBuilder):
    """
    Phase 2: 冲突检测与版本更替
    """

    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client
        self.decay_factor = 0.5

    def process(self, new_nodes, graph):
        all_nodes = graph.get_all_nodes()
        if len(all_nodes) < 2: return

        # 只比较 Profile 类型的节点
        profile_nodes = [n for n in all_nodes if "profile" in n.category.value]
        if len(profile_nodes) < 2: return

        conflict_count = 0

        for i in range(len(profile_nodes)):
            for j in range(i + 1, len(profile_nodes)):
                n1 = profile_nodes[i]
                n2 = profile_nodes[j]

                # NLI 检测
                is_conflict, debug_msg = self._detect_conflict(n1.content, n2.content)

                if is_conflict:
                    print(f"    ⚔️ [Conflict] '{n1.content}' vs '{n2.content}'")

                    # 判断谁新谁旧
                    ts1 = float(n1.timestamp) if n1.timestamp else 0
                    ts2 = float(n2.timestamp) if n2.timestamp else 0

                    newer = n2 if ts2 >= ts1 else n1
                    older = n1 if newer == n2 else n2

                    graph.add_edge(newer.node_id, older.node_id, EdgeType.VERSION)
                    older.energy_level *= self.decay_factor
                    conflict_count += 1

        if conflict_count > 0:
            print(f"  🧬 [Evolution] Resolved {conflict_count} conflicts")

    def _detect_conflict(self, text_a: str, text_b: str):
        if not self.llm: return False, "No LLM"

        prompt = CONFLICT_DETECTION_PROMPT.format(text_a=text_a, text_b=text_b)

        try:
            res = self.llm.chat([{"role": "user", "content": prompt}])
            content = ""
            if isinstance(res, dict):
                content = res.get("content", "")
            else:
                content = str(res)

            content = content.strip().upper()

            # 严格匹配 YES
            if content == "YES":
                return True, content
            else:
                return False, content

        except Exception as e:
            return False, str(e)