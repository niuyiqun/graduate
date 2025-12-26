# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：test_verifier.py
@Author  ：niu
@Date    ：2025/12/25 11:32 
@Desc    ：
"""

# c1/test_verifier.py
import json
import random
import time

# 定义 Instruction 常量（与你提供的一致）
INSTRUCTION_TEMPLATE = """
        你是一个认知记忆系统的预处理模块。请将用户的输入文本进行“多粒度正交分解”，拆解为以下四类记忆原子：

        1. [Event] 情景/事件：包含时间、动作的动态过程（如"用户去了..."）。
        2. [Entity] 实体：关键名词、人名、地名或物体（仅提取核心实体）。
        3. [Knowledge] 知识：客观事实、定义或常识（如"地球是圆的"）。
        4. [Rule] 规则：用户的显式指令、偏好或约束条件（如"用户喜欢..."）。

        约束：
        - 去除口语冗余（如“那个”、“嗯”）。
        - 保持原子的独立语义完整性。
        - **必须**使用 Markdown 代码块输出 JSON，格式如下：
        ```json
        {
            "event": ["...", "..."],
            "entity": ["...", "..."],
            "knowledge": ["...", "..."],
            "rule": ["...", "..."]
        }
        ```
        """

# === 数据模板库 ===

# 1. 场景：饮食偏好 (Food)
food_templates = [
    {
        "text": "帮我定个外卖，要{food}。记住，{reason}，所以我{rule}。",
        "slots": {
            "food": ["川菜", "火锅", "日料", "汉堡"],
            "reason": ["最近上火", "我在减肥", "我不吃香菜", "我对海鲜过敏"],
            "rule": ["不要放辣", "只要沙拉", "千万别放香菜", "不要点任何海鲜"]
        },
        "output_logic": lambda s: {
            "event": [f"定外卖({s['food']})"],
            "entity": ["外卖", s['food']],
            "knowledge": [s['reason']],  # 这里的reason既可以是fact也可以是rule，视情况，这里简化为Know/Rule
            "rule": [s['rule']]
        }
    },
    {
        "text": "{city}的菜太{taste}了。但我{rule}，以后给我推荐{city}的菜时，只选{taste_rule}的。",
        "slots": {
            "city": ["杭州", "成都", "苏州", "无锡"],
            "taste": ["甜", "辣", "清淡"],
            "rule": ["不爱吃甜", "特别能吃辣", "口味重"],
            "taste_rule": ["咸口", "变态辣", "重口味"]
        },
        "output_logic": lambda s: {
            "event": [],
            "entity": [s['city'], "菜"],
            "knowledge": [f"{s['city']}的菜太{s['taste']}"],
            "rule": [s['rule'], f"推荐{s['city']}菜时只选{s['taste_rule']}"]
        }
    }
]

# 2. 场景：工作/代码 (Work)
work_templates = [
    {
        "text": "我在调试{lang}代码。{concept}是指{def}。以后写代码时，{rule}。",
        "slots": {
            "lang": ["Python", "Java", "C++", "Go"],
            "concept": ["GIL", "NullPointer", "Goroutine"],
            "def": ["全局解释器锁", "空指针异常", "轻量级线程"],
            "rule": ["尽量用多进程", "必须判空", "注意并发安全"]
        },
        "output_logic": lambda s: {
            "event": [f"调试{s['lang']}代码"],
            "entity": [s['lang'], s['concept']],
            "knowledge": [f"{s['concept']}是指{s['def']}"],
            "rule": [f"写代码时{s['rule']}"]
        }
    }
]

# 3. 场景：日程/出行 (Daily)
daily_templates = [
    {
        "text": "帮我订一张去{loc}的票。{loc}是{desc}。我是{member}会员，必须订{airline}。",
        "slots": {
            "loc": ["北京", "上海", "纽约", "巴黎"],
            "desc": ["首都", "魔都", "大苹果", "浪漫之都"],
            "member": ["金卡", "白金", "星空联盟"],
            "airline": ["国航", "东航", "外航"]
        },
        "output_logic": lambda s: {
            "event": [f"订去{s['loc']}的票"],
            "entity": [s['loc'], s['airline'], "会员"],
            "knowledge": [f"{s['loc']}是{s['desc']}"],
            "rule": [f"必须订{s['airline']}"]
        }
    }
]

# 合并所有模板
ALL_TEMPLATES = food_templates + work_templates + daily_templates


def generate_sample(base_time=1766553910.0):
    """生成单条样本"""
    tmpl = random.choice(ALL_TEMPLATES)

    # 填充槽位
    filled_slots = {k: random.choice(v) for k, v in tmpl['slots'].items()}
    text = tmpl['text'].format(**filled_slots)

    # 生成输出 JSON
    logic_out = tmpl['output_logic'](filled_slots)

    # 构造最终 JSON 结构
    return {
        "instruction": INSTRUCTION_TEMPLATE.strip(),
        "input": f"Time: {base_time + random.uniform(10, 10000):.6f}\nText: {text}",
        "output": json.dumps(logic_out, ensure_ascii=False)
    }


def main():
    print(">>> 正在生成 Mock 数据...")
    with open("sft_golden_data_1k.jsonl", "w", encoding="utf-8") as f:
        start_time = time.time()
        for i in range(1000):
            sample = generate_sample(start_time + i * 60)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(">>> 生成完成！已保存至 sft_golden_data_1k.jsonl")


if __name__ == "__main__":
    main()
