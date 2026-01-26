# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：grpo.py
@Desc    ：GRPO Training (Ultimate Performance Version)
          配置：QLoRA 4-bit (省显存) + Flash Attention 2 (极速)
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入 Reward 函数
from c1.reward import (
    format_reward_func,
    orthogonality_reward_func,
    brevity_reward_func,
    deduplication_reward_func
)

# === 配置参数 ===
DATASET_PATH = os.path.join(current_dir, "dataset", "grpo_turn_level_data.jsonl")
OUTPUT_DIR = os.path.join(current_dir, "output", "grpo_v1")
MODEL_NAME_OR_PATH = "/root/.nyq/graduate/model/Qwen2.5-7B-Instruct"


def train_grpo():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print(f">>> [Rank {local_rank}] Loading Dataset & Model (Flash Attention 2 Mode)...")

    # 1. 加载数据
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ 数据文件未找到: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 2. 配置 4-bit 量化 (显存从 28G -> 10G 的关键)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 3. 加载模型 (开启 flash_attention_2)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        quantization_config=bnb_config,
        device_map={"": local_rank},  # 强制 DDP 绑定
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # <--- 核心加速配置
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 5. 训练参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,  # 单卡 Batch 1
        gradient_accumulation_steps=8,  # Global Batch 48
        bf16=True,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        use_vllm=False,
        ddp_find_unused_parameters=False,
    )

    # 6. 启动 Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            orthogonality_reward_func,
            brevity_reward_func,
            deduplication_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if local_rank == 0:
        print(f">>> Training Finished! Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    train_grpo()