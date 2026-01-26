import torch
import sys
import os
import json

# 添加路径以便能导入 c1 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from c1.prompts import DecouplerPrompt  # <--- 关键：导入您定义的 Prompt

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/.nyq/graduate/model/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/.nyq/graduate/c1/output/grpo_v1"
# =========================================

print(f">>> [1/3] Loading Base Model: {BASE_MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print(f">>> [2/3] Mounting LoRA Adapter: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ================= 构造测试数据 (模拟 Locomo10) =================
# 这是一个典型的 Locomo 风格的 Multi-turn 数据片段
# 我们模拟正在处理第 2 轮对话
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

# ================= 核心：使用 DecouplerPrompt 组装 =================
# 1. 获取 System Prompt (严格英文)
system_content = DecouplerPrompt.SYSTEM

# 2. 获取 User Input (History + Current Turn)
user_content = DecouplerPrompt.build_user_input(
    history_text=history_text,
    current_turn_text=current_turn_text
)

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]

# 3. 格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f">>> [3/3] Running Strict Inference (DecouplerPrompt)...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,  # 保持低温，要求精确
        do_sample=True
    )

# 4. 解码并尝试解析 JSON
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "=" * 20 + " 模型原始输出 " + "=" * 20)
print(response)
print("=" * 54)

# 尝试解析一下看是不是扁平的
try:
    # 去掉可能的 markdown 包裹
    clean_json = response.strip()
    if clean_json.startswith("```json"):
        clean_json = clean_json[7:]
    if clean_json.endswith("```"):
        clean_json = clean_json[:-3]

    data = json.loads(clean_json)
    print("\n>>> JSON 解析成功！检查结构：")
    for key, val in data.items():
        print(f"[{key}]: {val}")
        # 简单检查是不是 List[str]
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], str):
                print(f"   -> ✅ 格式正确: List[String]")
            else:
                print(f"   -> ❌ 格式警告: 依然是嵌套结构 (List[{type(val[0]).__name__}])")
except Exception as e:
    print(f"\n>>> JSON 解析失败: {e}")