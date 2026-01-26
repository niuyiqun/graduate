# c1/deduplicator.py
import json
import re
from typing import List
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from general.decoupled_memory import DecoupledMemoryAtom
from general.model import BaseModel
from general.base_memory import AgenticMemorySystem
from c1.prompts import DeduplicatorPrompt


@dataclass
class ResolutionAction:
    """LLM 返回的消解动作"""
    action_type: str  # 'add' 或 'drop'
    reasoning: str = ""


class SemanticRedundancyFilter:
    """
    【研究内容一(3)：基于逻辑博弈与预测偏差的双层压缩器】

    架构设计:
    Layer 1: 批次内跨视图消解 (Intra-Batch Cross-View Resolution)
             - 机制：四视图博弈矩阵
             - 判据：信息增益 (Information Gain)

    Layer 2: 全局差异化增量去重 (Global Differentiated Deduplication)
             - Episodic: 逻辑惊奇度 (Surprise Level)
             - Semantic: 逻辑蕴含 (Entailment)
    """

    def __init__(self, memory_system: AgenticMemorySystem, llm_model: BaseModel):
        self.memory_sys = memory_system
        self.llm = llm_model
        # 向量门控阈值
        self.SIM_THRESHOLD_LOW = 0.6
        self.SIM_THRESHOLD_HIGH = 0.92
        self.json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def filter_and_add_batch(self, new_atoms: List[DecoupledMemoryAtom]):
        """[主入口] 执行双层过滤流程"""
        if not new_atoms: return

        # === Layer 1: 批次内跨视图消解 ===
        if len(new_atoms) > 1:
            clean_atoms = self._intra_batch_cross_view_compression(new_atoms)
        else:
            clean_atoms = new_atoms

        # === Layer 2: 全局差异化处理 ===
        for atom in clean_atoms:
            if "episodic" in atom.atom_type:
                self._process_episodic_global(atom)
            else:
                self._process_semantic_global(atom)

    def _intra_batch_cross_view_compression(self, atoms: List[DecoupledMemoryAtom]) -> List[DecoupledMemoryAtom]:
        """Layer 1: 基于信息增益的消解"""
        if len(set(a.atom_type for a in atoms)) == 1: return atoms

        atoms_text = "\n".join([f"ID[{i}] Type={atom.atom_type}: {atom.content}" for i, atom in enumerate(atoms)])
        user_content = DeduplicatorPrompt.build_layer1_input(atoms_text)
        messages = [{"role": "user", "content": user_content}]

        try:
            res_data = self.llm.chat(messages)
            if isinstance(res_data, str):
                match = self.json_pattern.search(res_data)
                res_data = json.loads(match.group(1)) if match else {}

            keep_ids = res_data.get("keep_ids", [])
            if not keep_ids and "keep_ids" not in res_data: return atoms

            kept_atoms = [atom for i, atom in enumerate(atoms) if i in keep_ids]
            return kept_atoms
        except:
            return atoms

    def _process_episodic_global(self, new_atom: DecoupledMemoryAtom):
        """Layer 2A: Episodic - 预测偏差过滤"""
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        # 无相关记忆 -> 高惊奇度 -> Add
        if not related_memories:
            self._execute_action(ResolutionAction('add', "全新领域事件"), new_atom)
            return

        old_mems_text = "\n".join([f"- [{m.atom_type}] {m.content}" for m in related_memories])
        user_content = DeduplicatorPrompt.build_episodic_predict_input(old_mems_text, new_atom.content)
        messages = [
            {"role": "system", "content": DeduplicatorPrompt.LAYER2_EPISODIC_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            res = self.llm.chat(messages)
            if isinstance(res, str):
                match = self.json_pattern.search(res)
                res = json.loads(match.group(1)) if match else {"surprise_level": "high"}

            # Low Surprise -> Drop (冗余); High -> Add
            if res.get("surprise_level") != "low":
                self._execute_action(ResolutionAction('add', str(res.get('reasoning'))), new_atom)
            # else:
                # print(f"  [Drop] Low Surprise: {new_atom.content}")
        except:
            self._execute_action(ResolutionAction('add', "Error fallback"), new_atom)

    def _process_semantic_global(self, new_atom: DecoupledMemoryAtom):
        """Layer 2B: Semantic - 逻辑蕴含去重"""
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        if not related_memories:
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # 向量相似度快速门控
        emb_model = self.memory_sys.retriever.model
        new_emb = emb_model.encode([new_atom.content])
        old_emb = emb_model.encode([related_memories[0].content])
        similarity = cosine_similarity(new_emb, old_emb)[0][0]

        if similarity > self.SIM_THRESHOLD_HIGH: return # Drop
        if similarity < self.SIM_THRESHOLD_LOW:
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # LLM 逻辑判定
        old_mems_text = "\n".join([f"- {m.content}" for m in related_memories])
        user_content = DeduplicatorPrompt.build_semantic_entailment_input(old_mems_text, new_atom.content)
        messages = [
            {"role": "system", "content": DeduplicatorPrompt.LAYER2_SEMANTIC_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            res = self.llm.chat(messages)
            if isinstance(res, str):
                match = self.json_pattern.search(res)
                res = json.loads(match.group(1)) if match else {"action": "add"}

            if res.get('action', 'add') != 'drop':
                self._execute_action(ResolutionAction('add', str(res.get('reasoning'))), new_atom)
        except:
            self._execute_action(ResolutionAction('add'), new_atom)

    def _execute_action(self, action: ResolutionAction, new_atom: DecoupledMemoryAtom):
        """执行入库"""
        self.memory_sys.memory_manager.add_memory(new_atom)
        self.memory_sys.retriever.add_documents([new_atom.content])
        # print(f"  [+ Save] {new_atom.content[:30]}...")