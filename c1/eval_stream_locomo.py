# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：eval_stream_locomo.py
@Desc    ：【第一章：流式评测标准脚本】
          功能：模拟 Agent 实时对话，按轮次提取记忆并存入系统，在指定触发点进行 QA 测试。
          指标：计算 F1 Score (单词级重合度)。
"""

import os
import sys
import json
import collections
import string
from tqdm import tqdm
from typing import List, Dict

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入第一章核心组件
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter
from general.base_memory import AgenticMemorySystem
from general.model import QwenChat

# ================= 配置区 =================
CONFIG_PATH = os.path.join(root_dir, "config", "llm_config.yaml")
TEST_DATA_PATH = os.path.join(root_dir, "dataset", "locomo10.json")
WINDOW_SIZE = 6  # 解耦时参考的历史窗口大小


# =========================================

class LocomoStreamEvaluator:
    def __init__(self):
        print(">>> [Eval] 初始化流式评测引擎 (Baseline)...")
        # 1. 初始化大模型后端
        self.llm = QwenChat(CONFIG_PATH)

        # 2. 初始化记忆流水线组件
        self.memory_sys = AgenticMemorySystem()
        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

    def reset_system(self):
        """重置记忆库，确保样本之间不干扰"""
        self.memory_sys.memory_manager.clear()

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

    def run(self, limit=10):
        """执行评测，limit 参数用于限制测试样本数以节省时间"""
        if not os.path.exists(TEST_DATA_PATH):
            print(f"❌ 错误: 找不到数据集文件 {TEST_DATA_PATH}")
            return

        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_samples = data[:limit]
        all_f1_scores = []

        print(f"\n🚀 开始评测 | 模式: Baseline | 样本数: {len(test_samples)}")
        print("-" * 50)

        for sample in tqdm(test_samples, desc="Processing Samples"):
            self.reset_system()

            # 建立问题索引：哪些轮次需要触发提问
            q_map = {}
            for q in sample.get('questions', []):
                idx = q.get('trigger_turn', -1)
                q_map.setdefault(idx, []).append(q)

            # 整理对话流 (将所有 Session 展平为 Turn 序列)
            history_buffer = []
            all_turns = []
            sessions = sample['conversation']['sessions']
            for s_id in sorted(sessions.keys()):
                all_turns.extend(sessions[s_id]['turns'])

            # --- 流式循环 ---
            for i, turn in enumerate(all_turns):
                current_text = f"[{turn['speaker']}]: {turn['text']}"
                context_text = "\n".join(history_buffer[-WINDOW_SIZE:]) if history_buffer else ""

                # 1. 记忆处理：解耦 -> 校验 -> 去重入库
                try:
                    raw_obj = RawInputObj(text=current_text, context=context_text)
                    dirty_atoms = self.decoupler.decouple(raw_obj)
                    if dirty_atoms:
                        full_evidence = f"{context_text}\n{current_text}"
                        clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)
                        if clean_atoms:
                            self.deduplicator.filter_and_add_batch(clean_atoms)
                except Exception as e:
                    pass  # 评测时忽略单轮次异常

                # 更新历史窗口
                history_buffer.append(current_text)

                # 2. 检查是否有问题触发
                if i in q_map:
                    for q_item in q_map[i]:
                        # A. 检索记忆
                        relevant_mems = self.memory_sys.retrieve(q_item['question'], k=5)
                        mem_context = "\n".join([f"- {m.content}" for m in relevant_mems])

                        # B. 生成答案
                        answer_prompt = f"""Based on the memories below, answer the question briefly.
Memories:
{mem_context}

Question: {q_item['question']}
Answer:"""
                        prediction = self.llm.chat(user_input=answer_prompt)

                        # C. 评分
                        f1 = self._calculate_f1(prediction, q_item['answer'])
                        all_f1_scores.append(f1)

        # 最终汇总
        if all_f1_scores:
            final_score = sum(all_f1_scores) / len(all_f1_scores)
            print(f"\n{'=' * 40}")
            print(f"✅ 评测完成 | Final Result")
            print(f"   平均 F1 分数: {final_score:.4f}")
            print(f"{'=' * 40}")
        else:
            print("\n⚠️ 评测未产出有效分数，请检查数据集 trigger_turn 是否正确。")


if __name__ == "__main__":
    evaluator = LocomoStreamEvaluator()
    # 第一次建议先跑 5-10 个样本看效果
    evaluator.run(limit=10)