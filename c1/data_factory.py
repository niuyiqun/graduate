# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：data_factory.py
@Author  ：niu
@Date    ：2026/01/12
@Desc    ：GRPO 专用数据工厂
          功能：读取 LoCoMo 原始数据 -> 转换为 GRPO 训练所需的 Prompt 格式
          注意：GRPO 不需要 Ground Truth Output，只需要 Input Prompt。
"""

# c1/data_factory.py
import sys
import os
import json
from typing import List

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入 Prompt 模板
from c1.prompts import DecouplerPrompt
from env.load_locomo import load_locomo_dataset, LoCoMoSample


def format_turns(turns) -> str:
    """将对话 turns 转化为文本流"""
    lines = []
    for turn in turns:
        lines.append(f"[{turn.speaker}]: {turn.text}")
    return "\n".join(lines)


def build_grpo_dataset():
    # 1. 输入输出路径
    input_file = os.path.join(root_dir, "dataset", "locomo10.json")
    output_file = os.path.join(current_dir, "dataset", "grpo_train_data.jsonl")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f">>> [DataFactory] Loading raw data from: {input_file}")

    try:
        samples = load_locomo_dataset(input_file)
    except FileNotFoundError:
        print(f"❌ Error: 找不到数据集文件 {input_file}")
        return

    print(f"    Loaded {len(samples)} samples. Converting to GRPO prompts...")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            # LoCoMo 是按 Session 组织的，我们把每个 Session 作为一个训练样本
            for session_id, session in sample.conversation.sessions.items():
                # A. 提取原始对话文本
                dialogue_text = format_turns(session.turns)

                # B. 包装成 Prompt (System + User)
                # 注意：GRPO Trainer 会自动处理 chat template，
                # 但我们需要把 System Prompt 和 User Input 拼好，或者按照 huggingface dataset 格式存

                # 这里我们构建一个标准的 Chat 结构列表
                # {"prompt": [{"role": "system", ...}, {"role": "user", ...}]}

                user_content = DecouplerPrompt.build_user_input(dialogue_text)

                chat_messages = [
                    {"role": "system", "content": DecouplerPrompt.SYSTEM},
                    {"role": "user", "content": user_content}
                ]

                # 有些 TRL 版本需要纯文本 prompt，有些支持 list。
                # 为了通用性，我们这里使用 apply_chat_template 后的纯文本，
                # 或者让 Trainer 处理。最稳妥的方式是存 list，让 tokenizer apply_chat_template。

                data_item = {
                    "prompt": chat_messages,  # 存为列表，在 grpo.py 里处理
                    "source_id": f"{sample.sample_id}_{session_id}"  # 方便追踪
                }

                f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
                count += 1

    print(f">>> [DataFactory] Done! Saved {count} training samples to:")
    print(f"    {output_file}")
    print(">>> Next Step: Update 'DATASET_PATH' in c1/grpo.py to point to this file.")


if __name__ == "__main__":
    build_grpo_dataset()