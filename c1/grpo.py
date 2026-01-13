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

# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：grpo.py
@Author  ：niu
@Desc    ：GRPO Training Script (Optimized for Atomic Extraction)
          集成 4 种 Reward 函数，针对 380 条 Turn-level 数据进行微调。
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 引入 trl 的 GRPO
from trl import GRPOTrainer, GRPOConfig

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入所有 4 个 Reward 函数
from c1.reward import (
    format_reward_func,
    orthogonality_reward_func,
    brevity_reward_func,  # [新增] 简洁性
    deduplication_reward_func  # [新增] 去重
)

# --- 1. 配置参数 ---
# 确保这里指向刚刚生成的 380 条数据文件
DATASET_PATH = os.path.join(current_dir, "dataset", "grpo_turn_level_data.jsonl")
OUTPUT_DIR = os.path.join(current_dir, "output", "grpo_v1")

# 模型路径 (请根据您的实际路径修改)
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-7B-Instruct"  # 或您的本地路径


def train_grpo():
    print(f">>> Loading Dataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ 数据文件未找到: {DATASET_PATH}")

    # --- 2. 加载模型与 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载底座模型 (建议使用 4bit/8bit 量化以节省显存，根据您的硬件决定)
    # 如果显存够大 (24G+)，可以去掉 load_in_4bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # --- 3. 配置 LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 4. 配置 GRPO ---
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,  # GRPO 通常需要较小的 LR
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,  # 每一两步就打印日志，方便观察
        bf16=True,  # 开启 bf16 加速
        per_device_train_batch_size=4,  # 显存小的话改成 1 或 2
        gradient_accumulation_steps=4,
        num_generations=4,  # GRPO 核心参数：每条数据生成 4 个样本进行对比
        max_prompt_length=1024,  # 滑动窗口数据很短，1024 足够了
        max_completion_length=512,  # 输出的 JSON 也不会很长
        num_train_epochs=3,  # 380条数据，跑 3 个 epoch 差不多
        save_steps=50,
        max_grad_norm=0.1,
        use_vllm=False,  # 如果您没装 vLLM，设为 False
    )

    # --- 5. 初始化 Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,  # 格式必须对
            orthogonality_reward_func,  # 语义/情景必须分开
            brevity_reward_func,  # 必须简短原子
            deduplication_reward_func  # 内部不能重复
        ],
        args=training_args,
        train_dataset=DATASET_PATH,  # TRT 会自动处理 jsonl 加载
    )

    print(">>> Starting GRPO Training...")
    trainer.train()

    # 保存最终模型
    trainer.save_model(OUTPUT_DIR)
    print(f">>> Training Finished! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train_grpo()


