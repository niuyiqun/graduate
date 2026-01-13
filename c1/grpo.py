# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：grpo.py
@Author  ：niu
@Date    ：2026/01/12
@Desc    ：基于 GRPO 的正交分解微调脚本 (训练主入口)
          功能：
          1. 加载 grpo_train_data.jsonl 数据
          2. 使用 apply_chat_template 格式化 Prompt
          3. 加载 Qwen/Llama 模型与 LoRA 配置
          4. 注入多维正交奖励函数 (Format, Orthogonality, Completeness)
          5. 执行强化学习微调并保存权重
"""

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 导入奖励函数 (确保 c1/reward.py 已更新为最新版) ---
try:
    from c1.reward import (
        format_reward_func,
        orthogonality_reward_func,
        completeness_reward_func
    )
except ImportError:
    print("Error: 无法导入 c1.reward，请检查文件是否存在。")
    sys.exit(1)

# --- 全局配置 ---
# 建议使用指令遵循能力强的模型作为基座
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# 数据集路径 (由 c1/data_factory.py 生成)
DATASET_PATH = os.path.join(current_dir, "dataset", "grpo_turn_level_data.jsonl")
# 输出路径
OUTPUT_DIR = os.path.join(root_dir, "checkpoints", "grpo_decoupler_v1")


def train_grpo():
    print(f"\n{'=' * 20} GRPO Training Pipeline {'=' * 20}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Dataset   : {DATASET_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}\n")

    # 1. 加载 Tokenizer (用于处理 Chat Template)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Qwen/Llama 常见的 padding 处理
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"[Error] Tokenizer 加载失败: {e}")
        return

    # 2. 加载并处理数据集
    try:
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    except Exception as e:
        print(f"[Error] 数据集加载失败，请先运行 c1/data_factory.py 生成数据: {e}")
        return

    # [关键步骤] 将 List[Dict] 格式的 Prompt 转换为纯文本 String
    # GRPO 需要纯文本 Prompt 来进行后续生成的 concat
    def format_chat_template(example):
        # apply_chat_template 会自动拼接 System Prompt 和 User Input
        # tokenize=False: 返回字符串
        # add_generation_prompt=True: 添加 <|im_start|>assistant\n 引导模型开始生成
        text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        return {"prompt": text}

    print(">>> Processing chat templates...")
    dataset = dataset.map(format_chat_template)
    # 打印一条样本检查格式
    print(f"Sample Prompt:\n{dataset[0]['prompt'][:200]}...\n")

    # 3. 配置 LoRA (参数高效微调)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 4. 配置 GRPO 参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,  # RL 学习率通常极小 (1e-6 ~ 5e-6)
        per_device_train_batch_size=2,  # 根据显存调整 (24G显存建议 2-4)
        gradient_accumulation_steps=4,  # 累计梯度模拟大 Batch
        max_prompt_length=1024,
        max_completion_length=512,  # 限制输出长度
        num_generations=4,  # 核心参数: 每次采样 4 个结果进行对比 (Group Size)
        beta=0.04,  # KL 散度惩罚系数 (防止偏离基座太远)
        logging_steps=10,
        save_steps=100,
        max_steps=500,  # 训练步数 (演示用，实际建议 1000+)
        fp16=True,  # 开启混合精度加速
        report_to="none",  # 设为 "wandb" 可看曲线
        logging_first_step=True,
    )

    # 5. 加载模型
    # device_map="auto" 会自动分配显卡
    print(">>> Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        # load_in_4bit=True  # 如果显存 < 24G，建议取消注释开启 4-bit 量化
    )

    # 6. 初始化 GRPOTrainer
    # 将我们定义好的 3 个奖励函数传入
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,  # 权重 1.0 (格式)
            orthogonality_reward_func,  # 权重 1.0 (正交 - 核心)
            completeness_reward_func  # 权重 1.0 (完备)
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL 新版参数名 (旧版可能叫 tokenizer)
        peft_config=peft_config,
    )

    # 7. 开始训练
    print("\n>>> [GRPO] Start Training...")
    trainer.train()

    # 8. 保存最终模型
    print(f"\n>>> [GRPO] Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    # 同时保存 tokenizer 方便推理
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train_grpo()


