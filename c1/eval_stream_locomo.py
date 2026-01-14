# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：eval_stream_locomo.py
@Date    ：2026/1/13 20:38
@Desc    ：【流式评测引擎】模拟真实 Agent 的 "观察-提取-记忆-回答" 闭环。
          包含 Mock 数据模式，可直接运行验证代码逻辑。
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from c1.prompts import DecouplerPrompt

# ================= 配置区 =================
# 您的底座模型
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
# 您的 GRPO 训练结果 (训练完后这里会有文件)
LORA_PATH = os.path.join(current_dir, "output", "grpo_v1")

# LoCoMo 测试数据路径 (如果没有文件，会自动使用 Mock 数据)
LOCOMO_FILE = os.path.join(root_dir, "data", "locomo_test.json")


# =========================================

class GrpoMemoryAgent:
    def __init__(self, use_lora=True):
        print(f">>> 正在加载模型... (LoRA启用: {use_lora})")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

        # 加载底座
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if use_lora and os.path.exists(LORA_PATH):
            print(f">>> 加载 GRPO 适配器: {LORA_PATH}")
            self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        else:
            print("⚠️ 未找到 LoRA 权重或被禁用，将使用纯底座模型进行测试。")
            self.model = base_model

        self.model.eval()

        # 运行时状态
        self.memory_stream = []  # 存 JSON 字符串
        self.history_buffer = []  # 滑动窗口缓存

    def observe_turn(self, user_text, agent_text):
        """
        【第一章核心】流式提取：看一轮，记一轮。
        """
        current_turn = f"[User]: {user_text}\n[Agent]: {agent_text}"

        # 1. 构造 Input (利用滑动窗口)
        context_str = "\n".join(self.history_buffer[-4:])
        prompt = DecouplerPrompt.build_user_input(context_str, current_turn)

        messages = [
            {"role": "system", "content": DecouplerPrompt.SYSTEM},
            {"role": "user", "content": prompt}
        ]

        # 2. 推理 (提取 JSON)
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # 原子提取通常很短，200 token 足够
            outputs = self.model.generate(**inputs, max_new_tokens=200, temperature=0.01)

        extracted_content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 3. 存储 (简单清洗)
        # 如果模型输出了有效内容 (不只是空的 [])，就存入记忆流
        if "semantic_" in extracted_content or "episodic_" in extracted_content:
            # 打印出来让您看看效果 (定性分析)
            print(f"  🔍 [提取记忆]: {extracted_content[:100]}...")
            self.memory_stream.append(extracted_content)

        # 4. 更新滑动窗口
        self.history_buffer.append(current_turn)

    def answer_question(self, question):
        """
        【验证环节】基于提取出的记忆回答问题
        """
        # 简单策略：拼接所有记忆
        memory_context = "\n".join(self.memory_stream)

        solve_prompt = f"""
Based on the following extracted memories, answer the question briefly.

### Memories:
{memory_context}

### Question:
{question}

### Answer:
"""
        inputs = self.tokenizer(solve_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.1)

        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def get_test_data():
    """获取测试数据 (优先读取文件，否则使用 Mock)"""
    if os.path.exists(LOCOMO_FILE):
        print(f">>> 读取 LoCoMo 文件: {LOCOMO_FILE}")
        with open(LOCOMO_FILE, 'r') as f:
            return json.load(f)
    else:
        print(">>> ⚠️ 使用内置 Mock 数据进行测试 (仅验证代码逻辑)")
        return [
            {
                "session_id": "mock_001",
                "history": [
                    {"user": "Hi, I am Alex. I love spicy food.", "agent": "Nice to meet you Alex!"},
                    {"user": "I recently bought a Tesla Model 3.", "agent": "Wow, nice car!"},
                    {"user": "But I hate the touch screen controls.", "agent": "Yeah, that's a common complaint."}
                ],
                "questions": [
                    {"trigger_turn": 0, "text": "What kind of food does the user like?", "answer": "Spicy food"},
                    {"trigger_turn": 2, "text": "Why does the user dislike their car?",
                     "answer": "Touch screen controls"}
                ]
            }
        ]


def run_streaming_eval():
    # 1. 准备环境
    data = get_test_data()
    # 如果训练还没完，这里设为 False 可以先跑通底座逻辑
    use_lora = os.path.exists(LORA_PATH)
    agent = GrpoMemoryAgent(use_lora=use_lora)

    total_correct = 0
    total_questions = 0

    # 2. 遍历 Session
    for session in data:
        print(f"\n{'=' * 40}")
        print(f"🎬 Session Start: {session.get('session_id')}")

        # 重置 Agent 状态
        agent.memory_stream = []
        agent.history_buffer = []

        turns = session['history']
        # 建立索引：第几轮触发什么问题
        q_map = {}
        for q in session['questions']:
            idx = q['trigger_turn']
            if idx not in q_map: q_map[idx] = []
            q_map[idx].append(q)

        # 3. 流式循环 (Streaming Loop)
        for i, turn in enumerate(turns):
            u_text = turn.get('user', '')
            a_text = turn.get('agent', '')

            print(f"Turn {i}: User said '{u_text[:30]}...'")

            # --- 动作: 观察并提取 ---
            agent.observe_turn(u_text, a_text)

            # --- 动作: 触发测试 ---
            if i in q_map:
                for q in q_map[i]:
                    print(f"\n  ❓ [Question]: {q['text']}")
                    pred = agent.answer_question(q['text'])
                    print(f"  🤖 [Answer]: {pred.strip()}")
                    print(f"  ✅ [Gold]: {q['answer']}")

                    # 简单打分
                    if q['answer'].lower() in pred.lower():
                        print("  Result: Correct 🎉")
                        total_correct += 1
                    else:
                        print("  Result: Wrong ❌")
                    total_questions += 1
                    print("-" * 20)

    print(f"\n Final Score: {total_correct}/{total_questions} (Accuracy: {total_correct / total_questions:.2%})")


if __name__ == "__main__":
    run_streaming_eval()