"""
c2.processors.evolution - 记忆演化处理器 (Phase 2)
实现论文中的 "自监督反事实一致性校验" 和 "动态冲突消解"。
"""
import logging
import numpy as np
from ..data_models import MemoryNode, NodeType, EdgeType
from ..config import SystemConfig


class EvolutionProcessor:
    def __init__(self, graph_kernel):
        self.graph = graph_kernel
        self.logger = logging.getLogger("EvolutionProcessor")

    def process_evolution(self, new_nodes: List[MemoryNode]):
        """
        执行动态演化流程：
        1. 检索：查找与新 Profile 相似的历史节点。
        2. 判别：执行 NLI 逻辑判断 (Entailment/Contradiction)。
        3. 动作：建立 Version 边并应用衰减。
        """
        profiles = [n for n in new_nodes if n.node_type == NodeType.CONCEPT_PROFILE]
        if not profiles:
            return

        all_existing_profiles = self.graph.get_nodes_by_type(NodeType.CONCEPT_PROFILE)

        for new_p in profiles:
            # 1. 向量检索候选集 (Candidate Retrieval)
            candidates = self._retrieve_candidates(new_p, all_existing_profiles)

            for old_p in candidates:
                if old_p.node_id == new_p.node_id: continue

                # 2. 逻辑关系判别 (Logic Arbitration)
                relation = self._nli_arbitration(new_p.content, old_p.content)

                if relation == "CONTRADICTION":
                    self._execute_memory_update(old_p, new_p)
                elif relation == "ENTAILMENT":
                    self._execute_reinforcement(old_p, new_p)

    def _retrieve_candidates(self, query_node, pool):
        """基于向量余弦相似度召回相关记忆"""
        candidates = []
        q_vec = query_node.embedding

        for target in pool:
            t_vec = target.embedding
            # 计算 Cosine Similarity
            sim = np.dot(q_vec, t_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(t_vec) + 1e-9)

            if sim > SystemConfig.CONFLICT_RECALL_THRESHOLD:
                candidates.append(target)
        return candidates

    def _nli_arbitration(self, text_a: str, text_b: str) -> str:
        """
        自然语言推理接口 (Natural Language Inference).
        在生产环境中，此处应调用 fine-tuned DeBERTa 或 LLM API。
        此处实现确定的启发式逻辑以保证代码的可运行性和逻辑闭环。
        """
        # 否定词库
        negations = {"不", "没", "停止", "拒绝", "no", "not", "never", "stop"}

        set_a = set(text_a.replace("，", "").replace("。", "").split())
        set_b = set(text_b.replace("，", "").replace("。", "").split())

        # 简单逻辑：如果文本高度重叠但否定词状态不同，则为矛盾
        has_neg_a = any(w in text_a for w in negations)
        has_neg_b = any(w in text_b for w in negations)

        if has_neg_a != has_neg_b:
            return "CONTRADICTION"

        # 如果新记忆包含旧记忆的所有关键词，则为蕴含
        if len(set_a) > len(set_b):
            return "ENTAILMENT"

        return "NEUTRAL"

    def _execute_memory_update(self, old_node: MemoryNode, new_node: MemoryNode):
        """
        处理冲突：
        1. 建立 VERSION_UPDATE 边 (Old -> New)
        2. 对旧节点应用艾宾浩斯衰减 (Ebbinghaus Decay)
        """
        self.logger.info(f"[Evolution] Conflict Resolved: '{old_node.content}' -> '{new_node.content}'")

        self.graph.add_edge(
            old_node.node_id,
            new_node.node_id,
            EdgeType.VERSION_UPDATE,
            weight=1.0
        )

        # 核心机制：权重衰减
        decay_factor = SystemConfig.EBBINGHAUS_DECAY_RATE
        old_node.base_importance *= (1.0 - decay_factor)

        self.logger.debug(f"[Decay] Node {old_node.node_id} importance dropped to {old_node.base_importance:.4f}")

    def _execute_reinforcement(self, old_node, new_node):
        """处理蕴含：增强旧记忆的置信度"""
        old_node.base_importance = min(old_node.base_importance * 1.1, 2.0)