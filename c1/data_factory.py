# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：data_factory.py.py
@Author  ：niu
@Date    ：2025/12/23 19:15
@Desc    ：
"""

# -*- coding: UTF-8 -*-
"""
@Project ：graduate_project
@File    ：data_factory.py
@Author  ：Niu Yiqun
@Date    ：2025/12/26
@Desc    ：【研究内容一】SFT Golden Data 自动化合成工厂

           科研逻辑：
           为了解决通用大模型在“语义冲突”与“互补保留”上的策略缺陷，
           本模块采用“基于规则的模板生成 (Rule-based Template Generation)”技术，
           构建高信噪比、逻辑完备的 Golden Data，用于 Teacher-Student 蒸馏微调。
"""

import json
import random
import time
import uuid
from typing import List, Dict, Any, Callable

# === 1. 基础配置 ===

OUTPUT_FILE = "sft_golden_data_1k.jsonl"
TOTAL_SAMPLES = 1000

# 系统指令 (必须与 Decoupler 模块保持完全一致，保证训练目标对齐)
SYSTEM_INSTRUCTION = """
你是一个认知记忆系统的预处理模块。请将用户的输入文本进行“多粒度正交分解”，拆解为以下四类记忆原子：

1. [Event] 情景/事件：包含时间、动作的动态过程（如"用户去了..."）。
2. [Entity] 实体：关键名词、人名、地名或物体（仅提取核心实体）。
3. [Knowledge] 知识：客观事实、定义或常识（如"地球是圆的"）。
4. [Rule] 规则：用户的显式指令、偏好或约束条件（如"用户喜欢..."）。

约束：
- 去除口语冗余（如“那个”、“嗯”）。
- 保持原子的独立语义完整性。
- **必须**使用 Markdown 代码块输出 JSON。
"""

# === 2. 场景模板库 (Template Registry) ===
# 核心思想：利用 Python 的 Lambda 函数动态生成 Ground Truth，确保逻辑 100% 正确

TEMPLATES = [
    # -------------------------------------------------------------------------
    # Scenario A: [Hard Case] 环境限制与个人偏好的互补 (Environment vs Preference)
    # 目的：训练模型学会“互补保留”，不要因为语义相似就暴力去重
    # -------------------------------------------------------------------------
    {
        "name": "food_conflict",
        "text_template": "听说{city}的菜都比较{taste}。虽然{knowledge}，但我其实{preference}。以后帮我找餐厅时，{rule}。",
        "slots": {
            "city": ["杭州", "苏州", "无锡", "成都", "长沙"],
            "taste": ["甜", "清淡", "辣", "油腻"],
            "knowledge": ["那边也是美食荒漠", "是传统口味", "那边人很能吃辣"],
            "preference": ["不爱吃甜", "口味很重", "吃不了太辣", "在减肥"],
            "rule": ["避雷甜口的", "只找重口味的", "只要微辣的", "计算一下卡路里"]
        },
        # 逻辑生成器：定义了什么才是“完美的分解”
        "logic_generator": lambda s: {
            "event": [],  # 此句无具体 Event
            "entity": [s['city'], "餐厅", "菜"],
            "knowledge": [f"{s['city']}的菜比较{s['taste']}", s['knowledge']],  # 保留环境知识
            "rule": [s['preference'], f"找餐厅时{s['rule']}"]  # 保留个人规则
        }
    },

    # -------------------------------------------------------------------------
    # Scenario B: 代码纠错与知识沉淀 (Coding & Debugging)
    # 目的：训练模型准确区分 Event(报错) 和 Rule(修正策略)
    # -------------------------------------------------------------------------
    {
        "name": "code_debug",
        "text_template": "我在跑{lang}代码时遇到了{error}错误。查了一下，{error}通常是因为{reason}。以后再出现这个报错，直接{fix_strategy}，不用问我。",
        "slots": {
            "lang": ["Python", "PyTorch", "Java", "C++"],
            "error": ["OutOfMemory", "NullPointer", "SegFault", "Timeout"],
            "reason": ["显存溢出", "对象为空", "内存越界", "网络波动"],
            "fix_strategy": ["调小Batch Size", "加判空逻辑", "检查指针", "重试三次"]
        },
        "logic_generator": lambda s: {
            "event": [f"跑{s['lang']}代码遇到{s['error']}错误"],
            "entity": [s['lang'], s['error']],
            "knowledge": [f"{s['error']}通常是因为{s['reason']}"],
            "rule": [f"出现{s['error']}时直接{s['fix_strategy']}"]
        }
    },

    # -------------------------------------------------------------------------
    # Scenario C: 智能家居与条件触发 (Smart Home Rules)
    # 目的：训练模型提取复杂的条件规则 (If-Then Logic)
    # -------------------------------------------------------------------------
    {
        "name": "smart_home",
        "text_template": "把{device}打开。对了，设置一个联动规则：如果{condition}，就自动{action}，否则保持{state}。",
        "slots": {
            "device": ["客厅空调", "空气净化器", "加湿器", "窗帘"],
            "condition": ["室内温度高于30度", "PM2.5大于100", "湿度低于40%", "光照太强"],
            "action": ["开启强力模式", "最大风量", "喷雾", "拉上"],
            "state": ["睡眠模式", "静音模式", "待机", "半开"]
        },
        "logic_generator": lambda s: {
            "event": [f"打开{s['device']}"],
            "entity": [s['device'], "联动规则"],
            "knowledge": [],
            "rule": [f"如果{s['condition']}则自动{s['action']}，否则保持{s['state']}"]
        }
    },

    # -------------------------------------------------------------------------
    # Scenario D: 日程管理与负向约束 (Schedule & Constraints)
    # 目的：训练模型识别“不要做什么” (Negative Constraints)
    # -------------------------------------------------------------------------
    {
        "name": "schedule_neg",
        "text_template": "帮我约一下{person}开会，时间定在{time}。记住，{person}不喜欢{avoid}，所以会议室里千万别{rule_neg}。",
        "slots": {
            "person": ["李总", "王教授", "客户张经理"],
            "time": ["下周一上午", "明天下午三点", "周五晚上"],
            "avoid": ["烟味", "吵闹", "太冷"],
            "rule_neg": ["抽烟", "大声喧哗", "开太低空调"]
        },
        "logic_generator": lambda s: {
            "event": [f"约{s['person']}开会", f"时间定在{s['time']}"],
            "entity": [s['person'], "会议室"],
            "knowledge": [f"{s['person']}不喜欢{s['avoid']}"],
            "rule": [f"会议室里千万别{s['rule_neg']}"]
        }
    }
]


class DataFactory:
    """数据合成工厂类"""

    def __init__(self, seed=2025):
        random.seed(seed)
        self.base_time = 1766550000.0  # 模拟起始时间戳

    def _generate_single_sample(self) -> Dict[str, Any]:
        """生成单条样本"""
        # 1. 随机选择模板
        template = random.choice(TEMPLATES)

        # 2. 随机填充槽位
        filled_slots = {k: random.choice(v) for k, v in template['slots'].items()}

        # 3. 渲染原始文本
        raw_text = template['text_template'].format(**filled_slots)

        # 4. 执行逻辑生成器，得到完美的 JSON Ground Truth
        ground_truth_json = template['logic_generator'](filled_slots)

        # 5. 模拟时间戳
        self.base_time += random.uniform(60, 3600)
        input_content = f"Time: {self.base_time:.3f}\nText: {raw_text}"

        # 6. 构造 SFT 格式
        return {
            "instruction": SYSTEM_INSTRUCTION.strip(),
            "input": input_content,
            "output": json.dumps(ground_truth_json, ensure_ascii=False)
        }

    def produce(self, count: int, filepath: str):
        """批量生产并写入文件"""
        print(f">>> [DataFactory] 开始生产 {count} 条 Golden Data...")
        print(f">>> [Strategy] 采用基于规则的模板混合策略 (Rule-based Template Mixing)")

        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(count):
                sample = self._generate_single_sample()
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                if (i + 1) % 100 == 0:
                    print(f"    ...已生成 {i + 1} 条")

        print(f">>> [Success] 数据已保存至: {filepath}")
        print(f">>> [Check] 请检查文件内容以确保格式正确。")


# === 3. 执行入口 ===

if __name__ == "__main__":
    # 实例化工厂
    factory = DataFactory(seed=42)

    # 生产数据
    factory.produce(count=TOTAL_SAMPLES, filepath=OUTPUT_FILE)

    # 打印一条样例供核对
    print("\n>>> [Sample Review] 随机抽样一条数据展示：")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        print(f.readline())