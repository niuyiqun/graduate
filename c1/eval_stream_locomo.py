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
OUTPUT_MEM_PATH = os.path.join(root_dir, "c1", "output", "locomo_extracted_atoms.jsonl")
WINDOW_SIZE = 6


# =========================================

class LocomoStreamEvaluator:
    def __init__(self):
        print(">>> [Eval] 初始化流式评测引擎 (GRPO LoRA Version)...")
        self.llm = QwenGRPOChat(CONFIG_PATH)
        self.memory_sys = AgenticMemorySystem()

        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

    def reset_system(self):
        """完全重置记忆系统"""
        self.memory_sys.clear()

    def get_all_current_memories(self):
        """
        [修正版] 获取当前所有记忆内容
        """
        try:
            if hasattr(self.memory_sys, 'memory_manager'):
                # 获取所有 MemoryNote 对象
                all_notes = self.memory_sys.memory_manager.get_all_memories()
                # 提取 content 字段返回
                return [note.content for note in all_notes]
            else:
                return []
        except Exception as e:
            print(f"❌ [Read Error] 读取记忆失败: {e}")
            return []

    def _calculate_f1(self, prediction, ground_truth):
        """计算单词级 F1 Score"""

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
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _parse_locomo_timestamp(self, time_str: str) -> str:
        """
        解析 Locomo 时间格式: "6:29 pm on 7 July, 2023" -> "2023-07-07 18:29:00"
        """
        if not time_str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(time_str)

    def parse_locomo_sample(self, sample):
        """
        解析样本，并将 session 时间注入到每一个 turn 中
        """
        all_turns = []
        turn_mapping = {}

        conv_data = sample.get('conversation', {})
        # 获取所有 session key 并排序
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

            # 🔥 核心：获取该 Session 的时间
            date_key = f"{s_key}_date_time"
            raw_time_str = conv_data.get(date_key, "")
            formatted_time = self._parse_locomo_timestamp(raw_time_str)

            turns = conv_data[s_key]
            for t_idx, turn in enumerate(turns):
                turn_id_constructed = f"D{s_num}:{t_idx + 1}"
                turn_mapping[turn_id_constructed] = global_idx
                if 'dia_id' in turn:
                    turn_mapping[turn['dia_id']] = global_idx

                # 🔥 注入时间戳
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
                        if idx is not None and idx > max_idx:
                            max_idx = idx
                if max_idx != -1:
                    trigger_idx = max_idx

            # 如果没找到 evidence，默认挂载到最后一句
            if trigger_idx == -1:
                trigger_idx = len(all_turns) - 1

            if trigger_idx not in q_map:
                q_map[trigger_idx] = []
            q_map[trigger_idx].append(q)

        return all_turns, q_map

    def run(self, limit=1):
        """执行完整评测流程"""
        if not os.path.exists(TEST_DATA_PATH):
            print(f"❌ 错误: 找不到文件 {TEST_DATA_PATH}")
            return

        os.makedirs(os.path.dirname(OUTPUT_MEM_PATH), exist_ok=True)
        with open(OUTPUT_MEM_PATH, 'w', encoding='utf-8') as f:
            pass

        print(f"📂 [DEBUG] 结果保存至: {OUTPUT_MEM_PATH}")

        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_samples = data[:limit] if limit else data
        print(f"\n🚀 开始评测 (Limit={len(test_samples)})...")
        print("-" * 50)

        all_f1_scores = []

        with open(OUTPUT_MEM_PATH, 'a', encoding='utf-8') as f_out:

            for idx, sample in enumerate(test_samples):
                self.reset_system()
                source_id = sample.get('source_id') or sample.get('id') or f"sample_{idx}"
                all_turns, q_map = self.parse_locomo_sample(sample)
                history_buffer = []

                print(f"\n🔶 处理样本: {source_id} (共 {len(all_turns)} 轮对话)")

                # === 1. 遍历对话流 ===
                for i, turn in enumerate(all_turns):
                    current_text = f"[{turn['speaker']}]: {turn['text']}"
                    context_text = "\n".join(history_buffer[-WINDOW_SIZE:]) if history_buffer else ""
                    turn_timestamp = turn.get('timestamp')

                    print(f"\n--- Turn {i + 1} ---")
                    print(f"Time: {turn_timestamp}")
                    print(f"Target: {current_text}")

                    # [Step 1] 提取 (传入时间戳)
                    raw_obj = RawInputObj(
                        text=current_text,
                        context=context_text,
                        timestamp=turn_timestamp
                    )
                    dirty_atoms = self.decoupler.decouple(raw_obj)

                    # [Step 2] 校验
                    if dirty_atoms:
                        print(f"✅ [Decoupler] 提取: {[a.content for a in dirty_atoms]}")
                        full_evidence = f"{context_text}\n{current_text}"
                        clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)

                        if clean_atoms:
                            # [Step 3] 存储
                            self.deduplicator.filter_and_add_batch(clean_atoms)
                            print(f"📥 [Memory] 入库成功 (当前库大小: {len(self.get_all_current_memories())})")
                        else:
                            print("✂️ [Verifier] 全部拦截")
                    else:
                        print("⚠️ [Decoupler] 提取为空")

                    history_buffer.append(current_text)

                    # === 2. 触发 QA (完整逻辑) ===
                    if i in q_map:
                        for q_item in q_map[i]:
                            print("❓ 触发 QA 测试...")
                            question_text = q_item.get('question', '')
                            gold_answer = (q_item.get('answer') or q_item.get('answer_text') or q_item.get(
                                'adversarial_answer') or "")

                            if not gold_answer: continue

                            # 检索相关记忆
                            relevant_mems = self.memory_sys.find_related_memories(question_text, k=3)
                            # 这里 relevant_mems 是 MemoryNote 对象列表
                            mem_context = "\n".join([f"- {m.content}" for m in
                                                     relevant_mems]) if relevant_mems else "No relevant memory found."

                            # 构造 Prompt
                            qa_system = "You are a helpful assistant. Answer the question based strictly on the provided memories."
                            prompt_content = f"Memories:\n{mem_context}\n\nQuestion: {question_text}\nAnswer (briefly):"
                            messages = [{"role": "system", "content": qa_system},
                                        {"role": "user", "content": prompt_content}]

                            # 调用 LLM (QA 不需要 JSON)
                            response_dict = self.llm.chat(messages, parse_json=False)

                            if isinstance(response_dict, dict):
                                prediction = response_dict.get("answer") or response_dict.get("content") or str(
                                    response_dict)
                            else:
                                prediction = str(response_dict)

                            # 计算 F1
                            f1 = self._calculate_f1(str(prediction), str(gold_answer))
                            all_f1_scores.append(f1)
                            print(f"   [QA Result] F1: {f1:.2f} | Pred: {prediction} | Gold: {gold_answer}")

                # === 3. 保存结果 (完整保存) ===
                final_memories = self.get_all_current_memories()
                print(f"🏁 本样本最终记忆数: {len(final_memories)}")

                record = {
                    "source_id": source_id,
                    "extracted_atom_count": len(final_memories),
                    "memory_atoms": final_memories
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

        # 打印最终平均分
        if all_f1_scores:
            final_score = sum(all_f1_scores) / len(all_f1_scores)
            print(f"\n{'=' * 40}")
            print(f"✅ 评测完成 | Final Avg F1: {final_score:.4f}")
            print(f"{'=' * 40}")
        else:
            print("\n⚠️ 跑完了，但没有触发 QA 或没有有效分数。")


if __name__ == "__main__":
    evaluator = LocomoStreamEvaluator()
    # 跑前 5 个样本
    evaluator.run(limit=5)