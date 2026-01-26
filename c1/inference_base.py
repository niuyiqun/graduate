# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：inference_base.py
@Author  ：niu
@Date    ：2026/1/25 18:46 
@Desc    ：
"""

import torch
import sys
import os
import json

# 添加路径以便能导入 c1 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
# 注意：这里不需要导入 PeftModel，因为我们只跑基座
from c1.prompts import DecouplerPrompt

# ================= 配置区 =================
# 只定义基座路径
BASE_MODEL_PATH = "/root/.nyq/graduate/model/Qwen2.5-7B-Instruct"
# =========================================

print(f">>> [Baseline] Loading Base Model ONLY: {BASE_MODEL_PATH}")
# 使用 bfloat16 加载基座
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model.eval()

# ================= 构造测试数据 (完全一致) =================
# 使用和刚才 inference_strict.py 完全一样的 Alex & Ben 对话
history_text = """
[Turn 1]
Alex: Hey Ben, did you manage to catch that new sci-fi movie "Starlight" yesterday?
Ben: No, I got stuck at work. My boss kept me late to finish the quarterly report.
"""

current_turn_text = """
[Turn 2]
Alex: That sucks. I heard the visual effects were amazing, though the plot was a bit weak.
Ben: Typical. Anyway, I'm planning to go hiking at Blue Ridge this Saturday to blow off some steam. Want to come?
"""

# ================= 使用严格的 Prompt =================
system_content = DecouplerPrompt.SYSTEM
user_content = DecouplerPrompt.build_user_input(
    history_text=history_text,
    current_turn_text=current_turn_text
)

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f">>> [Baseline] Running Inference (Base Model)...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "=" * 20 + " [基座模型] 原始输出 " + "=" * 20)
print(response)
print("=" * 54)

# 尝试解析
try:
    clean_json = response.strip()
    if clean_json.startswith("```json"):
        clean_json = clean_json[7:]
    if clean_json.endswith("```"):
        clean_json = clean_json[:-3]

    data = json.loads(clean_json)
    print("\n>>> 解析结果检查：")
    for key, val in data.items():
        print(f"[{key}]: {val}")
except Exception as e:
    print(f"\n>>> 基座模型输出可能不是合法 JSON，解析失败: {e}")