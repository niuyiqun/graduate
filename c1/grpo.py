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
@Desc    ：GRPO Training Script (Optimized for 6x RTX 4090 DDP)
          配置：BF16 Full Precision + Flash Attention 2 + LoRA
          数据量：~380 turn-level samples
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 引入 TRL 的 GRPO
from trl import GRPOTrainer, GRPOConfig

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入 4 个 Reward 函数 (确保 c1/reward.py 存在)
from c1.reward import (
    format_reward_func,  # 格式约束
    orthogonality_reward_func,  # 语义/情景解耦
    brevity_reward_func,  # 简洁性
    deduplication_reward_func  # 去重
)

# === 1. 配置参数 ===
# 数据路径
DATASET_PATH = os.path.join(current_dir, "dataset", "grpo_turn_level_data.jsonl")
# 输出路径
OUTPUT_DIR = os.path.join(current_dir, "output", "grpo_v1")

# 模型路径 (Qwen2.5-7B)
MODEL_NAME_OR_PATH = "/root/.nyq/graduate/model/Qwen2.5-7B-Instruct"


def train_grpo():
    # DDP 模式下，只会由 Rank 0 打印日志，避免 6 张卡重复刷屏
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        print(f"[{local_rank}] >>> Loading Dataset: {DATASET_PATH}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ 数据文件未找到: {DATASET_PATH}")

    # --- 2. 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 加载模型 (针对 4090 DDP 优化) ---
    if local_rank <= 0:
        print(f"[{local_rank}] >>> Loading Model with BF16 & Flash Attention 2...")

    # 【关键配置】
    # 1. torch_dtype=torch.bfloat16: 4090 的原生强项，比 fp16 更稳。
    # 2. attn_implementation="flash_attention_2": 极大提升训练速度。
    # 3. device_map="auto": 【已删除】 DDP 模式下严禁使用，否则会报错。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # --- 4. 配置 LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    # 虽然 GRPOTrainer 内部会自动处理 PEFT，但显式 wrap 一下更保险
    model = get_peft_model(model, peft_config)

    if local_rank <= 0:
        model.print_trainable_parameters()

    # --- 5. 配置 GRPO 训练参数 ---
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,  # GRPO 敏感，LR 保持较低
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # === 6卡 并行核心配置 ===
        # 单卡 Batch=4 -> 6卡并行 = 全局 Batch Size 24
        per_device_train_batch_size=4,
        # 梯度累积=2 -> 最终等效 Batch Size = 24 * 2 = 48 (非常适合 380 条数据)
        gradient_accumulation_steps=2,

        # 混合精度
        bf16=True,  # 开启 BF16

        # GRPO 采样参数
        num_generations=8,  # 必须能被 device_batch_size 整除或作为倍数
        # 这里每条数据生成 8 个样本用于对比 (Reward 计算)

        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=5,  # 数据少，跑 5 轮保证收敛

        # DDP 优化
        ddp_find_unused_parameters=False,

        # 日志与保存
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",  # 不上传 wandb，本地跑

        # 既然显存够，我们可以不用 vLLM (配置复杂)，直接用 PyTorch 生成，稳一点
        use_vllm=False,
    )

    # --- 6. 初始化 Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,  # JSON 格式
            orthogonality_reward_func,  # 双流解耦
            brevity_reward_func,  # 简洁原子
            deduplication_reward_func  # 内部去重
        ],
        args=training_args,
        train_dataset=DATASET_PATH,
    )

    if local_rank <= 0:
        print(">>> Starting Distributed GRPO Training (6 GPUs)...")

    trainer.train()

    # --- 7. 保存模型 ---
    # DDP 模式下，只让主进程保存，防止文件冲突
    if local_rank <= 0:
        print(f">>> Training Finished! Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    train_grpo()


