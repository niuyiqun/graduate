from ..definitions import EdgeType, AtomType


class EvolutionBuilder:
    """Phase 2: 动态演化 - 模拟 NLI 冲突检测"""

    def build(self, graph, new_nodes):
        # 仅处理 Profile (用户画像) 类型的演化，如喜好变更
        profile_nodes = [n for n in new_nodes if n.atom.atom_type == AtomType.PROFILE]

        all_nodes = graph.get_all_nodes()

        for new_node in profile_nodes:
            # 模拟：简单查找内容包含相同关键词的旧节点 (真实环境用 Vector Search + LLM NLI)
            # 假设 new_node: "我不吃辣了", old_node: "我喜欢吃辣"
            for old_node in all_nodes:
                if old_node == new_node: continue
                if old_node.atom.atom_type != AtomType.PROFILE: continue

                # --- Mock NLI Logic ---
                # 如果讨论的是同一个话题但内容冲突
                if self._mock_nli_conflict(new_node.atom.content, old_node.atom.content):
                    # 建立演化边：旧 -> 新
                    graph.add_edge(old_node.id, new_node.id, EdgeType.VERSION, weight=1.0)
                    # 艾宾浩斯衰减：降低旧节点的基础质量
                    old_node.base_mass *= 0.5
                    print(f"[Evolution] Conflict Detected! Evolving {old_node.id} -> {new_node.id}")

    def _mock_nli_conflict(self, new_text, old_text):
        # 简单的规则模拟 LLM 判断
        keywords = ["辣", "甜", "咖啡"]
        for k in keywords:
            if k in new_text and k in old_text:
                # 简单假设：只要提到同类关键词，就算潜在更新
                return True
        return False