# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：synthetic_gen.py
@Author  ：niu
@Date    ：2026/01/12
@Desc    ：【数据增强引擎】滑动窗口切片生成器 (Sliding Window Slicer)
          修复版：强制对话生成阶段输出 JSON，避免底层接口报错。
"""

import os
import sys
import json
import time
import random
import re
from typing import List, Dict

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from general.model import ZhipuChat
from c1.prompts import DecouplerPrompt

# === 1. 人设库 (保持英文) ===
PERSONAS = [
    {
        "name": "Alex",
        "profile": "A tech blogger. Critical and fast-talking.",
        "style": "Tech jargon, concise."
    },
    {
        "name": "Bella",
        "profile": "Outdoor enthusiast. Loves hiking.",
        "style": "Cheerful, energetic."
    },
    {
        "name": "Charlie",
        "profile": "Backend engineer. Stressed and cynical.",
        "style": "Sarcastic, uses coding terms."
    },
    {
        "name": "Diana",
        "profile": "Foodie and artist. Sensitive.",
        "style": "Descriptive, gentle."
    },
    {
        "name": "Ethan",
        "profile": "Fitness coach. Disciplined.",
        "style": "Motivational, health-focused."
    }
]

# === 2. 话题库 (保持英文) ===
TOPICS = [
    "Comparing a new gadget vs an old one.",
    "Planning a detailed travel itinerary.",
    "Complaining about a bad day at work.",
    "Debating a controversial news topic.",
    "Sharing a childhood memory."
]


class DialogueSimulator:
    def __init__(self):
        config_path = os.path.join(root_dir, "config", "llm_config.yaml")
        try:
            self.llm = ZhipuChat(config_path)
            print(">>> [模拟器] LLM 模型加载成功。")
        except Exception as e:
            print(f"❌ [模拟器] LLM 加载失败: {e}")
            sys.exit(1)

    def generate_turn(self, speaker: Dict, receiver: Dict, history: List[str], topic: str) -> str:
        """让 LLM 自然地生成下一轮对话内容 (强制 JSON 输出以适配接口)"""

        # 上下文窗口：最近 4 轮 (您刚才修改的)
        context = "\n".join(history[-4:])

        # [修改点] Prompt 显式要求 JSON 格式
        system_prompt = f"""
Role: {speaker['name']} ({speaker['profile']})
Style: {speaker['style']}
Topic: {topic}
Task: Reply to {receiver['name']}.
Constraint: Keep it natural. Length: 20-50 words.

**IMPORTANT OUTPUT FORMAT**: 
You must return a valid JSON object with a single key "content".
Example: {{"content": "Hey there! How are you?"}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Dialogue History:\n{context}\n\nPlease respond (JSON ONLY):"}
        ]

        try:
            response = self.llm.chat(messages)

            # [修改点] 健壮的解析逻辑
            content_text = ""

            # 情况 1: 接口已经自动转成了 dict
            if isinstance(response, dict):
                content_text = response.get('content', '')

            # 情况 2: 接口返回了字符串 (即使它报错了，通常也会返回 raw string)
            elif isinstance(response, str):
                # 尝试清洗 Markdown 标记
                clean_str = response.replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean_str)
                    content_text = data.get('content', str(response))
                except json.JSONDecodeError:
                    # 如果还是解析不了，说明模型没听话，尝试正则提取
                    match = re.search(r'"content":\s*"(.*?)"', clean_str, re.DOTALL)
                    if match:
                        content_text = match.group(1)
                    else:
                        # 实在不行，直接用原始文本 (去掉可能存在的花括号)
                        content_text = clean_str.replace("{", "").replace("}", "").strip()

            return content_text.strip()

        except Exception as e:
            print(f"  [生成错误] {e}")
            return "..."

    def run_simulation(self, num_sessions=50):
        """主流程：滑动窗口切片生成"""
        output_file = os.path.join(current_dir, "dataset", "grpo_turn_level_data.jsonl")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f">>> 模式: 滑动窗口切片生成 (Sliding Window Slicing)")
        print(f">>> 目标: 生成 {num_sessions} 个 Session")
        print(f">>> 输出路径: {output_file}")

        total_samples = 0

        # 使用 'w' 模式覆盖旧文件
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(num_sessions):
                p1, p2 = random.sample(PERSONAS, 2)
                topic = random.choice(TOPICS)

                history_log = []

                print(f"[{i + 1}/{num_sessions}] {p1['name']} vs {p2['name']} | 话题: {topic[:15]}...")

                # 随机决定本次对话长度 (15-25 轮)
                num_turns = random.randint(15, 25)

                curr, other = p1, p2

                # --- 轮次循环 ---
                for turn_idx in range(num_turns):
                    # 1. 生成文本 (这里现在会返回纯文本，且不会报错了)
                    text = self.generate_turn(curr, other, history_log, topic)

                    # 只有生成了有效文本才保存
                    if not text or text == "...":
                        continue

                    current_turn_formatted = f"[{curr['name']}]: {text}"

                    # 2. 创建训练样本 (滑动窗口: 4)
                    context_window_text = "\n".join(history_log[-4:]) if history_log else "(Conversation Start)"

                    user_input_str = DecouplerPrompt.build_user_input(
                        history_text=context_window_text,
                        current_turn_text=current_turn_formatted
                    )

                    chat_messages = [
                        {"role": "system", "content": DecouplerPrompt.SYSTEM},
                        {"role": "user", "content": user_input_str}
                    ]

                    data_item = {
                        "prompt": chat_messages,
                        "source": "synthetic_turn_level_v1",
                        "metadata": {
                            "session_id": i,
                            "turn_index": turn_idx,
                            "speaker": curr['name']
                        }
                    }

                    f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
                    total_samples += 1

                    # 3. 更新历史
                    history_log.append(current_turn_formatted)
                    curr, other = other, curr

                    # time.sleep(0.1)

                f.flush()

        print(f"\n>>> 生成完成！共生成 {total_samples} 条原子级训练样本。")
        print(f">>> 准备就绪，可以开始 GRPO 训练了。")


if __name__ == "__main__":
    sim = DialogueSimulator()
    sim.run_simulation(num_sessions=50)