# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：construct_dataset.py
@Author  ：niu
@Date    ：2025/12/24 13:08 
@Desc    ：
"""

# c1/construct_dataset.py
import sys
import os
import json
import time
from tqdm import tqdm

# 路径适配
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general.model import ZhipuChat
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.reward import RewardScorer


def construct_rft_data():
    """
    构造 RFT (Rejection Sampling Fine-Tuning) 数据集
    使用 API 生成多个样本，用 RewardScorer 筛选最好的，存为 JSONL
    """

    # 1. 初始化
    config_path = "../config/llm_config.yaml"
    try:
        # 【未来切换点】：以后这里换成 LocalLlamaChat 即可，下面逻辑不用变
        llm = ZhipuChat(config_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    decoupler = SemanticDecoupler(llm)
    scorer = RewardScorer()

    # 2. 准备原始数据 (这里模拟一些长程交互数据)
    # 实际毕设中，你可以用公开的对话数据集 (如 ShareGPT, DailyDialog)
    raw_data_list = [
        "哎，那个，我之前不是说喜欢吃苹果吗？我现在改主意了，因为苹果太酸了。你知道吗，苹果其实属于蔷薇科。以后每天早上给我准备一根香蕉吧。",
        "帮我把这个会议记录整理一下。张经理说下周一要交财报，李工说服务器得扩容。记住，以后这种财务相关的由于涉及机密，不要发到群里，私发给我。",
        "Python的列表推导式挺好用的，就是读起来有点费劲。我习惯用for循环。你以后给我写代码的时候，尽量别用太复杂的单行代码，我看不懂。",
        # ... 更多数据 ...
    ]

    output_file = "dataset/sft_train_data.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"开始构造数据，共 {len(raw_data_list)} 条，每条采样 3 次...")

    # 3. 循环采样与筛选
    with open(output_file, "w", encoding="utf-8") as f:
        for raw_text in tqdm(raw_data_list):
            best_score = -999
            best_output = None

            # 对每条数据采样 N 次 (Best-of-N 策略)
            # 在 API 模式下，主要靠 temperature=0.7 带来的随机性
            for i in range(3):
                raw_obj = RawInputObj(text=raw_text)

                # 调用 decoupler (它内部会调 API)
                # 注意：为了拿原始 JSON，我们可能需要微调 decoupler 或直接在此处解析
                # 这里我们直接利用 decoupler.decouple 返回的 Atoms 反推结构，或者直接存 Prompt/Completion

                # 为了微调，我们需要的是 {Input, Output_JSON_String}
                # 简单起见，我们直接构造 Prompt 调 LLM，复用 Decoupler 的 Prompt
                messages = [
                    {"role": "system", "content": decoupler.system_prompt},
                    {"role": "user", "content": f"Time: {raw_obj.timestamp}\nText: {raw_text}"}
                ]

                # 调用 API
                try:
                    response_dict = llm.chat(messages)  # ZhipuChat 返回的是 dict

                    # 计算分数
                    score = scorer.compute_score(raw_text, response_dict)

                    # 记录最佳
                    if score > best_score:
                        best_score = score
                        best_output = response_dict

                except Exception as e:
                    print(f"采样失败: {e}")
                    continue

            # 4. 保存最佳样本 (Golden Sample)
            if best_output:
                # 构造符合微调格式的数据 (通常是 Alpaca 或 ShareGPT 格式)
                training_sample = {
                    "instruction": decoupler.system_prompt,
                    "input": f"Time: {time.time()}\nText: {raw_text}",
                    "output": json.dumps(best_output, ensure_ascii=False)  # 存为 JSON 字符串
                }

                f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                # print(f"  -> 已保存最佳样本 (Score: {best_score:.1f})")

    print(f"\n数据构造完成！已保存至 {output_file}")
    print("后续可使用此文件对本地模型进行 LoRA 微调。")


if __name__ == "__main__":
    construct_rft_data()