# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：deduplicator.py
@Desc    ：双层压缩器 - 完整版
          ✅ 修正：_execute_action 全属性存储 (Type, Timestamp, etc.)
          ✅ 修正：atom_type 安全获取
"""

# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：deduplicator.py
@Desc    ：【Debug 显影版】打印所有 Drop 原因，彻底查清为什么记忆被杀
"""

import json
import re
from typing import List
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

try:
    from general.decoupled_memory import DecoupledMemoryAtom
except ImportError:
    pass

from general.model import BaseModel
from general.base_memory import AgenticMemorySystem
from c1.prompts import DeduplicatorPrompt


@dataclass
class ResolutionAction:
    action_type: str
    reasoning: str = ""


class SemanticRedundancyFilter:
    def __init__(self, memory_system: AgenticMemorySystem, llm_model: BaseModel):
        self.memory_sys = memory_system
        self.llm = llm_model
        # 调整阈值：稍微放宽一点，避免误杀
        self.SIM_THRESHOLD_LOW = 0.6
        self.SIM_THRESHOLD_HIGH = 0.95
        self.json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def filter_and_add_batch(self, new_atoms: List):
        if not new_atoms: return

        # print(f"  🔍 [Deduplicator] Input Batch: {len(new_atoms)}")

        # Layer 1
        if len(new_atoms) > 1:
            clean_atoms = self._intra_batch_cross_view_compression(new_atoms)
        else:
            clean_atoms = new_atoms

        # Layer 2
        for atom in clean_atoms:
            atype = getattr(atom, 'atom_type', 'episodic')
            if "episodic" in atype:
                self._process_episodic_global(atom)
            else:
                self._process_semantic_global(atom)

    def _intra_batch_cross_view_compression(self, atoms: List) -> List:
        if len(set(getattr(a, 'atom_type', 'u') for a in atoms)) == 1: return atoms

        atoms_text = "\n".join(
            [f"ID[{i}] Type={getattr(atom, 'atom_type', 'mem')}: {atom.content}" for i, atom in enumerate(atoms)])
        user_content = DeduplicatorPrompt.build_layer1_input(atoms_text)
        messages = [{"role": "user", "content": user_content}]

        try:
            res_data = self.llm.chat(messages)
            if isinstance(res_data, str):
                match = self.json_pattern.search(res_data)
                res_data = json.loads(match.group(1)) if match else {}

            keep_ids = res_data.get("keep_ids", [])
            if not keep_ids and "keep_ids" not in res_data: return atoms

            # Debug 打印
            if len(keep_ids) < len(atoms):
                print(f"    ✂️ [Layer 1] Batch Compression: {len(atoms)} -> {len(keep_ids)}")

            return [atom for i, atom in enumerate(atoms) if i in keep_ids]
        except:
            return atoms

    def _process_episodic_global(self, new_atom):
        """Layer 2A: Episodic"""
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        if not related_memories:
            # print("    ✅ [Logic] No related memory -> Add")
            self._execute_action(ResolutionAction('add', "全新事件"), new_atom)
            return

        old_mems_text = "\n".join([f"- [{getattr(m, 'atom_type', 'stored')}] {m.content}" for m in related_memories])

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

            level = res.get("surprise_level", "high")
            reason = res.get("reasoning", "No reasoning")

            if level != "low":
                # print(f"    ✅ [Logic] Surprise={level} -> Add")
                self._execute_action(ResolutionAction('add', str(reason)), new_atom)
            else:
                # 🔥 重点：打印为什么被 Drop
                print(f"    🗑️ [DROP] Surprise=LOW | Content: {new_atom.content[:30]}... | Reason: {reason}")

        except Exception as e:
            print(f"    ⚠️ [Error] LLM Check Failed: {e}")
            self._execute_action(ResolutionAction('add', "Error fallback"), new_atom)

    def _process_semantic_global(self, new_atom):
        """Layer 2B: Semantic"""
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)
        related_memories = [m for m in related_memories if m is not None]

        if not related_memories:
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # 1. 向量门控
        try:
            emb_model = self.memory_sys.retriever.model
            new_emb = emb_model.encode([new_atom.content])
            old_emb = emb_model.encode([related_memories[0].content])
            similarity = cosine_similarity(new_emb, old_emb)[0][0]

            if similarity > self.SIM_THRESHOLD_HIGH:
                print(f"    🗑️ [DROP] Vector Sim Too High ({similarity:.4f}) | Content: {new_atom.content[:30]}...")
                return

            if similarity < self.SIM_THRESHOLD_LOW:
                # print(f"    ✅ [Logic] Vector Sim Low ({similarity:.4f}) -> Add")
                self._execute_action(ResolutionAction('add'), new_atom)
                return
        except:
            pass

        # 2. LLM 逻辑判定
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

            action = res.get('action', 'add')
            reason = res.get('reasoning', "")

            if action != 'drop':
                self._execute_action(ResolutionAction('add', str(reason)), new_atom)
            else:
                print(f"    🗑️ [DROP] Entailment=True | Content: {new_atom.content[:30]}... | Reason: {reason}")

        except:
            self._execute_action(ResolutionAction('add'), new_atom)

    def _execute_action(self, action: ResolutionAction, new_atom):
        try:
            note_id = self.memory_sys.add_note(
                content=new_atom.content,
                atom_type=getattr(new_atom, 'atom_type', 'general'),
                timestamp=getattr(new_atom, 'timestamp', None),
                retrieval_count=0,
                importance_score=1.0
            )
            # print(f"      💾 Stored: {new_atom.content[:20]}...")
        except Exception as e:
            print(f"❌ [Save Error] {e}")