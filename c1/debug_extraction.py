# -*- coding: UTF-8 -*-
import sys
import os
import json

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from general.model import QwenGRPOChat
from c1.prompts import DecouplerPrompt


def verify_fix():
    print(">>> 1. 初始化模型 (Testing Model Fix)...")
    config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    # 初始化时会打印 "Fix Single Quote" 字样，说明代码是最新的
    llm = QwenGRPOChat(config_path)

    print(">>> 2. 构造 Prompt...")
    history = "[Turn 1] Alex: I am going to buy a coffee."
    current = "[Turn 2] Bob: Bring me a latte, please. I love milk."

    messages = [
        {"role": "system", "content": DecouplerPrompt.SYSTEM},
        {"role": "user", "content": DecouplerPrompt.build_user_input(history, current)}
    ]

    print(">>> 3. 调用模型 (期望底层自动处理单引号)...")

    # 这里的 parse_json=True (默认) 会触发 model.py 里的 ast.literal_eval
    response = llm.chat(messages)

    print("\n" + "=" * 20 + " 验证结果 " + "=" * 20)

    # 检查返回类型
    if isinstance(response, dict):
        # 检查是否包含核心字段（说明解析成功了）
        if "semantic_profile" in response or "episodic_activity" in response:
            print("✅ [Success] 验证通过！")
            print("底层 model.py 成功把单引号文本转成了 Python 字典。")
            print("-" * 30)
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("-" * 30)
            print("💡 现在您可以直接去跑 c1/eval_stream_locomo.py 了！")
        elif "content" in response:
            print("❌ [Fail] 依然返回了原始字符串 wrapper，解析失败。")
            print("Raw Content:", response["content"])
        else:
            print("⚠️ [Warn] 返回了字典，但格式奇怪：", response.keys())
    else:
        print(f"❌ [Fail] 返回类型错误: {type(response)}")
        print("Raw:", response)


if __name__ == "__main__":
    verify_fix()