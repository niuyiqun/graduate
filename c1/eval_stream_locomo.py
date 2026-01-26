# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：eval_stream_locomo.py
@Desc    ：【第一章：流式评测标准脚本 - 最终版】
          功能升级：
          1. 支持 GRPO 微调模型 (QwenGRPOChat)
          2. ✅ 新增：将生成的记忆原子持久化存储到 JSONL，供 C2/C3 使用
"""

import os
import sys
import json
import re
import collections
import string
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
from general.model import QwenGRPOChat  # <--- 使用 GRPO 模型

# ================= 配置区 =================
CONFIG_PATH = os.path.join(root_dir, "config", "llm_config.yaml")
TEST_DATA_PATH = os.path.join(root_dir, "dataset", "locomo10.json")
# 结果保存路径 (供 C2/C3 使用)
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
        获取当前记忆库中的所有原子内容
        假设 memory_sys 内部有一个 memories 列表，且每个 memory 对象有 content 属性
        """
        # 兼容不同的 memory_sys 实现，尝试获取原子列表
        if hasattr(self.memory_sys, 'memories'):
            return [m.content for m in self.memory_sys.memories]
        elif hasattr(self.memory_sys, 'get_all'):
            return self.memory_sys.get_all()
        else:
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

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def parse_locomo_sample(self, sample):
        """解析 LoCoMo 格式"""
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

            turns = conv_data[s_key]
            for t_idx, turn in enumerate(turns):
                turn_id_constructed = f"D{s_num}:{t_idx + 1}"
                turn_mapping[turn_id_constructed] = global_idx
                if 'dia_id' in turn:
                    turn_mapping[turn['dia_id']] = global_idx
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

            if trigger_idx == -1:
                trigger_idx = len(all_turns) - 1

            if trigger_idx not in q_map:
                q_map[trigger_idx] = []
            q_map[trigger_idx].append(q)

        return all_turns, q_map

    def run(self, limit=5):
        """执行评测并保存记忆"""
        if not os.path.exists(TEST_DATA_PATH):
            print(f"❌ 错误: 找不到文件 {TEST_DATA_PATH}")
            return

        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_MEM_PATH), exist_ok=True)
        # 清空旧文件，重新写入
        with open(OUTPUT_MEM_PATH, 'w', encoding='utf-8') as f:
            pass

        print(f"📂 记忆提取结果将保存至: {OUTPUT_MEM_PATH}")

        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_samples = data[:limit] if limit else data
        all_f1_scores = []

        print(f"\n🚀 开始评测 (GRPO Model) | 样本数: {len(test_samples)}")
        print("-" * 50)

        # 打开文件准备追加写入 (Append Mode)
        with open(OUTPUT_MEM_PATH, 'a', encoding='utf-8') as f_out:

            for idx, sample in enumerate(tqdm(test_samples, desc="Processing Samples")):
                self.reset_system()  # 每个样本开始前清空记忆库

                # 获取样本 ID，方便后续 C2/C3 对应
                source_id = sample.get('source_id') or sample.get('id') or f"sample_{idx}"

                all_turns, q_map = self.parse_locomo_sample(sample)
                history_buffer = []

                # === 1. 遍历对话流，提取记忆 ===
                for i, turn in enumerate(all_turns):
                    current_text = f"[{turn['speaker']}]: {turn['text']}"
                    context_text = "\n".join(history_buffer[-WINDOW_SIZE:]) if history_buffer else ""

                    # DEBUG: 打印
                    # print(f"Processing Turn {i}: {current_text[:50]}...")

                    try:
                        raw_obj = RawInputObj(text=current_text, context=context_text)
                        dirty_atoms = self.decoupler.decouple(raw_obj)

                        if dirty_atoms:
                            # 验证 + 存入记忆库
                            full_evidence = f"{context_text}\n{current_text}"
                            clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)
                            if clean_atoms:
                                self.deduplicator.filter_and_add_batch(clean_atoms)
                    except Exception as e:
                        # print(f"❌ Error: {e}")
                        pass

                    history_buffer.append(current_text)

                    # === 2. 触发 QA (计算 F1) ===
                    if i in q_map:
                        for q_item in q_map[i]:
                            question_text = q_item.get('question', '')
                            gold_answer = (q_item.get('answer') or q_item.get('answer_text') or q_item.get(
                                'adversarial_answer') or "")
                            if not gold_answer: continue

                            relevant_mems = self.memory_sys.find_related_memories(question_text, k=3)
                            mem_context = "\n".join([f"- {m.content}" for m in
                                                     relevant_mems]) if relevant_mems else "No relevant memory found."

                            qa_system = "You are a helpful assistant. Answer the question based strictly on the provided memories."
                            prompt_content = f"Memories:\n{mem_context}\n\nQuestion: {question_text}\nAnswer (briefly):"
                            messages = [{"role": "system", "content": qa_system},
                                        {"role": "user", "content": prompt_content}]

                            response_dict = self.llm.chat(messages)

                            if isinstance(response_dict, dict):
                                prediction = response_dict.get("answer") or response_dict.get("content") or str(
                                    response_dict)
                            else:
                                prediction = str(response_dict)

                            f1 = self._calculate_f1(str(prediction), str(gold_answer))
                            all_f1_scores.append(f1)

                # === 3. 【核心新增】保存本轮提取的所有记忆 ===
                # 在 reset 之前，把记忆库里的东西捞出来存盘
                final_memories = self.get_all_current_memories()

                record = {
                    "source_id": source_id,
                    "extracted_atom_count": len(final_memories),
                    "memory_atoms": final_memories
                    # 如果需要，也可以把 QA 的 F1 存下来分析
                }

                # 写入 JSONL
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()  # 强制刷新缓冲区，防止程序中断丢失数据

        if all_f1_scores:
            final_score = sum(all_f1_scores) / len(all_f1_scores)
            print(f"\n{'=' * 40}")
            print(f"✅ [GRPO] 评测完成 | Final F1: {final_score:.4f}")
            print(f"📂 记忆原子已保存至: {OUTPUT_MEM_PATH}")
            print(f"{'=' * 40}")
        else:
            print("\n⚠️ 跑完了，但没有有效分数。")


if __name__ == "__main__":
    evaluator = LocomoStreamEvaluator()
    # 建议先跑 5 个验证 output 文件内容是否正确
    evaluator.run(limit=5)