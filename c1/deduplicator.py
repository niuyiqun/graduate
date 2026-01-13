"""
@Project ：graduate
@File    ：deduplicator.py
@Author  ：niu
@Date    ：2025/12/3 10:15
@Desc
"""

# c1/deduplicator.py
import json
from typing import List
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# 导入通用结构
from general.decoupled_memory import DecoupledMemoryAtom
from general.model import BaseModel
from general.base_memory import AgenticMemorySystem
# 导入 Prompt 库
from c1.prompts import DeduplicatorPrompt


@dataclass
class ResolutionAction:
    """LLM 返回的消解动作"""
    action_type: str  # 'add' (新增/高权重) 或 'drop' (丢弃/低权重)
    reasoning: str = ""


class SemanticRedundancyFilter:
    """
    【研究内容一(3)：基于逻辑博弈与预测偏差的双层压缩器】
    (Dual-Layer Compression via Logic Game & Prediction Deviation)

    核心科研逻辑：
    该模块负责将高维的 Atom 流压缩进有限的存储空间，依托双流架构的分离特性，
    实施“批次内-全局”双层过滤架构。

    架构设计 (Dual-Layer Architecture):
    -------------------------------------------------------
    Layer 1: 批次内跨视图消解 (Intra-Batch Cross-View Resolution)
    - 目标：解决单次提取中，四个正交槽位之间的视图冗余。
    - 机制：构建“四视图博弈矩阵 (4-View Game Matrix)”。
    - 判据：信息增益 (Information Gain)。
      不粗暴地让 Rule 覆盖 Event，而是判断 Event 是否提供了 Rule 未涵盖的特例细节。

    Layer 2: 全局差异化增量去重 (Global Differentiated Deduplication)
    - 目标：解决新原子与存量记忆库之间的冗余。
    - 机制：差异化治理。
      A. 针对情景流 (Episodic) -> 基于“逻辑惊奇度”的动态过滤 (Prediction Deviation)。
         利用存量的 Semantic 规则对新 Event 进行零样本预测。
         - 符合预测 (如每天喝咖啡) -> 低权重 -> Drop/Merge。
         - 违背预测 (如突然倒掉咖啡) -> 高权重 (Surprise) -> Add。
      B. 针对语义流 (Semantic) -> 基于“逻辑蕴含”的结构化去重 (Logic Entailment)。
         识别新旧规则的包含关系。若新规则被旧规则蕴含 (上位概念)，则剔除。
    -------------------------------------------------------
    """

    def __init__(self, memory_system: AgenticMemorySystem, llm_model: BaseModel):
        self.memory_sys = memory_system
        self.llm = llm_model

        # --- 自适应向量门控阈值 ---
        # 仅在 Semantic 流去重或 Episodic 初筛时使用
        self.SIM_THRESHOLD_LOW = 0.6  # 差异过大，直接新增
        self.SIM_THRESHOLD_HIGH = 0.92  # 极度相似，直接丢弃

    def filter_and_add_batch(self, new_atoms: List[DecoupledMemoryAtom]):
        """
        [主入口] 执行双层过滤流程。
        """
        if not new_atoms:
            return

        print(f"--- [Filter] 收到 {len(new_atoms)} 条原子，启动双层自适应压缩 ---")

        # === Layer 1: 批次内跨视图消解 (Game Matrix) ===
        if len(new_atoms) > 1:
            clean_atoms = self._intra_batch_cross_view_compression(new_atoms)
        else:
            clean_atoms = new_atoms

        print(f"--- [Layer 1] 跨视图博弈完成，剩余 {len(clean_atoms)} 条，进入全局处理 ---")

        # === Layer 2: 全局差异化处理 (Global Differentiated) ===
        for atom in clean_atoms:
            # 根据原子类型分流
            if "episodic" in atom.atom_type:
                self._process_episodic_global(atom)
            else:
                self._process_semantic_global(atom)

    # =========================================================================
    # Layer 1: 批次内跨视图消解 (Intra-Batch Cross-View)
    # =========================================================================
    def _intra_batch_cross_view_compression(self, atoms: List[DecoupledMemoryAtom]) -> List[DecoupledMemoryAtom]:
        """
        Layer 1: 基于四视图博弈矩阵与信息增益的消解
        """
        # 如果类型单一，不存在跨视图冲突
        if len(set(a.atom_type for a in atoms)) == 1:
            return atoms

        # 构造 Prompt
        atoms_text = "\n".join([f"ID[{i}] Type={atom.atom_type}: {atom.content}" for i, atom in enumerate(atoms)])
        user_content = DeduplicatorPrompt.build_layer1_input(atoms_text)

        messages = [{"role": "user", "content": user_content}]

        try:
            res_data = self.llm.chat(messages)

            # 解析 JSON
            if isinstance(res_data, str):
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res_data, re.DOTALL)
                res_data = json.loads(match.group(1)) if match else {}

            keep_ids = res_data.get("keep_ids", [])

            # 容错：默认全保留
            if not keep_ids and "keep_ids" not in res_data:
                return atoms

            kept_atoms = []
            for i, atom in enumerate(atoms):
                if i in keep_ids:
                    kept_atoms.append(atom)
                else:
                    print(f"  [Layer 1 Drop] 视图博弈淘汰: [{atom.atom_type}] {atom.content[:20]}... (低信息增益)")
            return kept_atoms

        except Exception as e:
            print(f"  [Layer 1 Error] {e}，跳过压缩")
            return atoms

    # =========================================================================
    # Layer 2A: 情景流 - 预测偏差过滤 (Episodic Prediction Deviation)
    # =========================================================================
    def _process_episodic_global(self, new_atom: DecoupledMemoryAtom):
        """
        针对 Event/Thought：计算“逻辑惊奇度”。
        检索 Semantic Rules，看是否能预测此 Event。
        """
        # 1. 检索上下文 (寻找可能解释该事件的 Profile/Knowledge)
        # 注意：这里我们检索的是混合记忆，希望检索到相关的 Semantic 规则
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        # 如果没有相关记忆，说明是全新领域的事件 -> 高惊奇度 -> Add
        if not related_memories:
            self._execute_action(ResolutionAction('add', "全新领域事件"), new_atom)
            return

        # 2. 调用 LLM 进行预测偏差判定
        old_mems_text = "\n".join([f"- [{m.atom_type}] {m.content}" for m in related_memories])
        user_content = DeduplicatorPrompt.build_episodic_predict_input(old_mems_text, new_atom.content)

        messages = [
            {"role": "system", "content": DeduplicatorPrompt.LAYER2_EPISODIC_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            res = self.llm.chat(messages)
            # 解析
            if isinstance(res, str):
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res, re.DOTALL)
                res = json.loads(match.group(1)) if match else {"surprise_level": "high"}

            # 3. 根据惊奇度决策
            # "low" -> 符合预测 (冗余) -> Drop (或仅更新计数器，此处简化为Drop)
            # "high" -> 违背预测/新特例 -> Add
            if res.get("surprise_level") == "low":
                print(f"  [Dev. Drop] 符合预期 (Low Surprise): {new_atom.content[:20]}... -> {res.get('reasoning')}")
            else:
                self._execute_action(ResolutionAction('add', f"Surprise: {res.get('reasoning')}"), new_atom)

        except Exception as e:
            # 容错：默认作为新事件存入
            self._execute_action(ResolutionAction('add', "Error fallback"), new_atom)

    # =========================================================================
    # Layer 2B: 语义流 - 逻辑蕴含去重 (Semantic Logic Entailment)
    # =========================================================================
    def _process_semantic_global(self, new_atom: DecoupledMemoryAtom):
        """
        针对 Profile/Knowledge：基于逻辑蕴含判断上位关系。
        """
        # 1. 向量门控初筛 (Semantic Stream 比较适合用向量相似度先过滤一遍)
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        if not related_memories:
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # 向量相似度检查 (针对纯文本重复)
        emb_model = self.memory_sys.retriever.model
        new_emb = emb_model.encode([new_atom.content])
        old_emb = emb_model.encode([related_memories[0].content])
        similarity = cosine_similarity(new_emb, old_emb)[0][0]

        if similarity > self.SIM_THRESHOLD_HIGH:
            print(f"  [Gate Fast-Drop] 语义高度重复 (Sim={similarity:.2f}) -> Drop")
            return

        if similarity < self.SIM_THRESHOLD_LOW:
            # 差异很大，直接存，省 LLM
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # 2. LLM 逻辑蕴含判定 (Entailment Check)
        old_mems_text = "\n".join([f"- {m.content}" for m in related_memories])
        user_content = DeduplicatorPrompt.build_semantic_entailment_input(old_mems_text, new_atom.content)

        messages = [
            {"role": "system", "content": DeduplicatorPrompt.LAYER2_SEMANTIC_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            res = self.llm.chat(messages)
            if isinstance(res, str):
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res, re.DOTALL)
                res = json.loads(match.group(1)) if match else {"action": "add"}

            action_type = res.get('action', 'add')
            reasoning = res.get('reasoning', '')

            if action_type == 'drop':
                print(f"  [Logic Drop] 逻辑被蕴含: {new_atom.content[:20]}... -> {reasoning}")
            else:
                self._execute_action(ResolutionAction('add', reasoning), new_atom)

        except Exception as e:
            self._execute_action(ResolutionAction('add'), new_atom)

    def _execute_action(self, action: ResolutionAction, new_atom: DecoupledMemoryAtom):
        """执行入库"""
        self.memory_sys.memory_manager.add_memory(new_atom)
        self.memory_sys.retriever.add_documents([new_atom.content])
        print(f"  [+ Save] 入库成功: {new_atom.content[:20]}... ({action.reasoning})")