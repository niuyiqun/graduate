# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：merge_model.py.py
@Author  ：niu
@Date    ：2026/1/25 18:54 
@Desc    ：
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ================= 配置区 =================
# 1. 基座模型
BASE_MODEL_PATH = "/root/.nyq/graduate/model/Qwen2.5-7B-Instruct"
# 2. 您的 LoRA 训练产物
LORA_PATH = "/root/.nyq/graduate/c1/output/grpo_v1"
# 3. 合并后的模型保存路径 (建议单独起个名字)
OUTPUT_PATH = "/root/.nyq/graduate/model/Qwen2.5-GRPO-Merged"
# =========================================

print(f">>> [1/4] Loading Base Model...")
# 必须用 bfloat16 加载，防止精度溢出
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f">>> [2/4] Loading LoRA Adapter...")
# 挂载 LoRA
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print(f">>> [3/4] Merging Weights (This may take a minute)...")
# 核心步骤：合并并卸载 LoRA，变成一个普通模型
model = model.merge_and_unload()

print(f">>> [4/4] Saving to {OUTPUT_PATH}...")
# 保存模型权重
model.save_pretrained(OUTPUT_PATH)

# 重要：别忘了保存 Tokenizer！否则 vLLM 跑不起来
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f">>> ✅ Merge Completed! Your new model is ready at: {OUTPUT_PATH}")
