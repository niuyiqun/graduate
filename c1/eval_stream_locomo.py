# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：eval_stream_locomo.py
@Desc    ：【第一章：流式评测标准脚本 - 最终完整版】
          包含：
          1. Locomo 时间戳解析 (Inject Real Timestamp)
          2. 完整的提取-校验-去重流程
          3. 完整的 QA 评测与 F1 计算逻辑
          4. 完整的 JSONL 结果保存逻辑
"""

# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：eval_stream_locomo.py
@Desc    ：【Debug 最终版】流式评测 + 全链路显影日志
"""

import os
import sys
import json
import re
import collections
import string
from datetime import datetime
from tqdm import tqdm

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入组件
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter
from general.base_memory import AgenticMemorySystem
from general.model import QwenGRPOChat

# ================= 配置区 =================
CONFIG_PATH = os.path.join(root_dir, "config", "llm_config.yaml")
TEST_DATA_PATH = os.path.join(root_dir, "dataset", "locomo10.json")
OUTPUT_MEM_PATH = os.path.join(root_dir, "c1", "output", "locomo_extracted_atoms_no_embedding.jsonl")
WINDOW_SIZE = 6


# =========================================

class LocomoStreamEvaluator:
    def __init__(self):
        print(">>> [Eval] 初始化流式评测引擎 (Debug Mode)...")
        self.llm = QwenGRPOChat(CONFIG_PATH)
        self.memory_sys = AgenticMemorySystem()

        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

    def reset_system(self):
        """完全重置记忆系统"""
        print("🔄 [System] 正在重置记忆系统...")
        self.memory_sys.clear()

    def get_all_current_memories(self):
        """
        [最终修正版] 获取当前所有记忆，包含 Embedding
        """
        try:
            if hasattr(self.memory_sys, 'memory_manager'):
                all_notes = self.memory_sys.memory_manager.get_all_memories()

                serialized_memories = []
                for note in all_notes:
                    serialized_memories.append({
                        "id": note.id,
                        "content": note.content,
                        "atom_type": getattr(note, 'atom_type', 'general'),
                        "timestamp": getattr(note, 'timestamp', 'unknown'),
                        "source_text": getattr(note, 'source_text', ''),
                        "created_at": note.timestamp,
                        # 🔥 新增：导出 Embedding (这会是一个 float 列表)
                        "embedding": note.embedding
                    })
                return serialized_memories
            else:
                return []
        except Exception as e:
            print(f"❌ [Read Error] 读取记忆失败: {e}")
            return []

    def _calculate_f1(self, prediction, ground_truth):
        def normalize_answer(s):
            s = str(s).lower()
            s = "".join(ch for ch in s if ch not in set(string.punctuation))
            return s.split()

        pred_tokens = normalize_answer(prediction)
        gold_tokens = normalize_answer(ground_truth)

        if not pred_tokens or not gold_tokens:
            return 1.0 if pred_tokens == gold_tokens else 0.0

        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)

    def _parse_locomo_timestamp(self, time_str: str) -> str:
        if not time_str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return str(time_str)

    def parse_locomo_sample(self, sample):
        all_turns = []
        turn_mapping = {}
        conv_data = sample.get('conversation', {})
        session_keys = [k for k in conv_data.keys() if 'session' in k and 'date' not in k]
        try:
            session_keys.sort(key=lambda x: int(x.split('_')[1]))
        except:
            pass

        global_idx = 0
        for s_key in session_keys:
            try:
                s_num = s_key.split('_')[1]
            except:
                s_num = "1"

            # 解析时间
            date_key = f"{s_key}_date_time"
            raw_time_str = conv_data.get(date_key, "")
            formatted_time = self._parse_locomo_timestamp(raw_time_str)

            turns = conv_data[s_key]
            for t_idx, turn in enumerate(turns):
                turn_id_constructed = f"D{s_num}:{t_idx + 1}"
                turn_mapping[turn_id_constructed] = global_idx
                if 'dia_id' in turn: turn_mapping[turn['dia_id']] = global_idx

                # 注入时间戳
                turn['timestamp'] = formatted_time
                all_turns.append(turn)
                global_idx += 1

        questions = sample.get('qa', [])
        q_map = {}
        for q in questions:
            evidence_raw_list = q.get('evidence', [])
            trigger_idx = -1
            if evidence_raw_list:
                max_idx = -1
                for ev_str in evidence_raw_list:
                    sub_ids = re.split(r'[;,\s]+', ev_str)
                    for sub_id in sub_ids:
                        sub_id = sub_id.strip()
                        if not sub_id: continue
                        idx = turn_mapping.get(sub_id)
                        if idx is not None and idx > max_idx: max_idx = idx
                if max_idx != -1: trigger_idx = max_idx

            if trigger_idx == -1: trigger_idx = len(all_turns) - 1
            if trigger_idx not in q_map: q_map[trigger_idx] = []
            q_map[trigger_idx].append(q)

        return all_turns, q_map

    def run(self, limit=1):
        if not os.path.exists(TEST_DATA_PATH): return

        os.makedirs(os.path.dirname(OUTPUT_MEM_PATH), exist_ok=True)
        with open(OUTPUT_MEM_PATH, 'w', encoding='utf-8') as f:
            pass

        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_samples = data[:limit] if limit else data
        print(f"\n🚀 开始评测 (Debug Mode, Samples={len(test_samples)})...")

        with open(OUTPUT_MEM_PATH, 'a', encoding='utf-8') as f_out:

            for idx, sample in enumerate(test_samples):
                # 1. 重置 (注意：只在每个 Sample 开始时重置)
                self.reset_system()

                source_id = sample.get('source_id') or sample.get('id') or f"sample_{idx}"
                all_turns, q_map = self.parse_locomo_sample(sample)
                history_buffer = []

                print(f"\n🔶 处理样本: {source_id} (共 {len(all_turns)} 轮对话)")

                # === 遍历对话流 ===
                for i, turn in enumerate(all_turns):
                    # 打印当前库大小，检查是否被异常清空
                    current_mem_count = len(self.get_all_current_memories())
                    print(f"\n=== Turn {i + 1} Start | Current Memory Size: {current_mem_count} ===")

                    current_text = f"[{turn['speaker']}]: {turn['text']}"
                    context_text = "\n".join(history_buffer[-WINDOW_SIZE:]) if history_buffer else ""
                    turn_timestamp = turn.get('timestamp')

                    print(f"Time: {turn_timestamp}")
                    print(f"Target: {current_text}")

                    # [Step 1] 提取
                    raw_obj = RawInputObj(text=current_text, context=context_text, timestamp=turn_timestamp)
                    dirty_atoms = self.decoupler.decouple(raw_obj)

                    if not dirty_atoms:
                        print("⚠️ [Decoupler] 提取为空")
                    else:
                        print(f"✅ [Decoupler] 提取: {[a.content for a in dirty_atoms]}")

                    # [Step 2] 校验
                    if dirty_atoms:
                        full_evidence = f"{context_text}\n{current_text}"
                        clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)

                        # Debug: 打印校验结果
                        if len(clean_atoms) < len(dirty_atoms):
                            print(f"✂️ [Verifier] 删除了 {len(dirty_atoms) - len(clean_atoms)} 条")
                        print(f"🛡️ [Verifier] 校验后: {[a.content for a in clean_atoms]}")

                        if clean_atoms:
                            # [Step 3] 存储
                            print(f"⚙️ [Deduplicator] 准备入库 {len(clean_atoms)} 条...")
                            self.deduplicator.filter_and_add_batch(clean_atoms)

                            # 立即检查
                            new_count = len(self.get_all_current_memories())
                            print(f"📥 [Storage] 操作后库大小: {new_count} (增量: {new_count - current_mem_count})")
                        else:
                            print("🚫 [Verifier] 全部拦截，跳过存储")

                    history_buffer.append(current_text)

                    # [QA]
                    if i in q_map:
                        for q_item in q_map[i]:
                            print("❓ 触发 QA 测试...")
                            question_text = q_item.get('question', '')
                            gold_answer = (q_item.get('answer') or q_item.get('answer_text') or "")

                            if not gold_answer: continue

                            relevant_mems = self.memory_sys.find_related_memories(question_text, k=3)
                            mem_context = "\n".join([f"- {m.content}" for m in
                                                     relevant_mems]) if relevant_mems else "No relevant memory found."

                            # 打印 QA 看到的记忆
                            # print(f"   [QA Context]: {mem_context}")

                            qa_system = "You are a helpful assistant. Answer the question based strictly on the provided memories."
                            prompt_content = f"Memories:\n{mem_context}\n\nQuestion: {question_text}\nAnswer (briefly):"
                            messages = [{"role": "system", "content": qa_system},
                                        {"role": "user", "content": prompt_content}]

                            response_dict = self.llm.chat(messages, parse_json=False)
                            prediction = str(
                                response_dict.get("content", "") if isinstance(response_dict, dict) else response_dict)

                            f1 = self._calculate_f1(prediction, gold_answer)
                            print(f"   [QA Result] F1: {f1:.2f} | Pred: {prediction} | Gold: {gold_answer}")

                # === End Turn Loop ===
                final_memories = self.get_all_current_memories()
                print(f"\n🏁 本样本最终记忆数: {len(final_memories)}")
                record = {"source_id": source_id, "extracted_atom_count": len(final_memories),
                          "memory_atoms": final_memories}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()


if __name__ == "__main__":
    evaluator = LocomoStreamEvaluator()
    evaluator.run(limit=5)