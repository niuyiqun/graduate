# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：decoupler.py.py
@Author  ：niu
@Date    ：2025/12/24 12:33 
@Desc    ：
"""

# c1/decoupler.py
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass

# 导入通用结构
from general.decoupled_memory import DecoupledMemoryAtom
# 导入模型基类
from general.model import BaseModel


@dataclass
class RawInputObj:
    """
    [Data Object] 原始输入封装
    用于标准化进入 Pipeline 的非结构化文本流，保留时序信息以支持时序推理。
    """
    text: str
    timestamp: float = None
    source: str = "user"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SemanticDecoupler:
    """
    【研究内容一(1)：多粒度语义正交分解器 (Multi-Granularity Orthogonal Decoupler)】

    核心科研逻辑：
    该模块是整个记忆系统的“入水口”，负责解决非结构化对话流中“信息纠缠”与“信噪比低”的问题。

    架构设计 (Architecture Design):
    -------------------------------------------------------
    1. 理论基础：双流认知架构 (Dual-Stream Cognitive Architecture)
       - 模拟人脑将“情景记忆 (Episodic)”与“语义记忆 (Semantic)”分离的机制。
       - 将混杂文本流拆解为独立的特征子空间，防止时序信息与逻辑规则混淆。

    2. 核心机制：基于 Schema 的正交投影 (Schema-Based Orthogonal Projection)
       - 利用 LLM 的 In-Context Learning 能力，强约束模型将输入映射到四个预定义的正交槽位：
         [Event] / [Entity] / [Knowledge] / [Rule]。
       - 这种解耦是后续“高密度压缩”和“精确推理”的前提。
    -------------------------------------------------------
    """

    def __init__(self, llm_model: BaseModel):
        """
        初始化分解器
        :param llm_model: 传入初始化好的 Teacher Model (如 GLM-4/Llama-3)
        """
        self.llm = llm_model

        # --- Schema Definition (核心定义的语义投影空间) ---
        # 对应 PPT 中的“多视图正交槽位定义”
        # 这里的 Prompt 设计体现了“模式遵循 (Schema Following)”的技术路线
        self.system_prompt = """
        你是一个认知记忆系统的预处理模块。请将用户的输入文本进行“多粒度正交分解”，拆解为以下四类记忆原子：

        1. [Event] 情景/事件：包含时间、动作的动态过程（如"用户去了..."）。
           -> 对应 Episodic Stream (时序流)
        2. [Entity] 实体：关键名词、人名、地名或物体（仅提取核心实体）。
           -> 对应 Concept Graph Node (概念节点)
        3. [Knowledge] 知识：客观事实、定义或常识（如"地球是圆的"）。
           -> 对应 Semantic Stream (语义流-客观)
        4. [Rule] 规则：用户的显式指令、偏好或约束条件（如"用户喜欢..."）。
           -> 对应 Semantic Stream (语义流-主观)

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

    def decouple(self, raw_input: RawInputObj) -> List[DecoupledMemoryAtom]:
        """
        [主入口] 执行分解流水线 (Execution Pipeline)。

        流程：
        1. Context Injection: 注入时间戳上下文。
        2. Inference: 调用 LLM 进行正交投影。
        3. Fallback: 异常熔断与降级处理。
        4. Atomization: 归一化封装为记忆原子。
        """

        # 1. 构造上下文 (Context Construction)
        # 将时间维度显式注入，辅助模型判断 Event 的时序属性
        user_content = f"Time: {raw_input.timestamp}\nText: {raw_input.text}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        print(f"--- [Decoupler] 正在调用大模型分解: {raw_input.text[:20]}... ---")

        # 2. LLM 推理 (Inference)
        response_data = self.llm.chat(messages)

        # 3. 鲁棒性工程：错误处理与降级 (Robustness & Fallback)
        # 如果模型输出格式崩坏或 API 报错，执行“降级策略”：
        # 将整句话作为单一 Event 存储，保证系统可用性 (Availability) > 完美性 (Perfection)
        if "error" in response_data:
            print(f"[Error] 模型调用失败: {response_data.get('details')}")
            # Fallback: Treat the whole text as a single raw event
            return [DecoupledMemoryAtom(content=raw_input.text, atom_type="event", source_text=raw_input.text)]

        # 4. 原子化与归一化 (Atomization & Normalization)
        atoms = []
        parsed_data = response_data  # 假设 LLM 基类已处理 JSON 解析

        # 类型映射表：处理 LLM 可能输出的复数形式或大小写差异
        type_map = {
            'event': 'event', 'events': 'event',
            'entity': 'entity', 'entities': 'entity',
            'knowledge': 'knowledge',
            'rule': 'rule', 'rules': 'rule'
        }

        for key, content_list in parsed_data.items():
            # 过滤掉非 Schema 定义的幻觉 Key
            norm_type = type_map.get(key.lower())
            if not norm_type:
                continue

            if isinstance(content_list, list):
                for content in content_list:
                    # 实例化原子对象
                    # 这一步完成了从“非结构化文本”到“可计算对象”的质变
                    atom = DecoupledMemoryAtom(
                        content=content,
                        atom_type=norm_type,
                        source_text=raw_input.text,  # 保留溯源信息 (Provenance)
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(raw_input.timestamp))
                    )
                    atoms.append(atom)

        print(f"  -> 初步提取 {len(atoms)} 条原子")
        return atoms


# test_real_llm.py
import sys
import os

# 确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from general.model import ZhipuChat
from general.base_memory import AgenticMemorySystem


def main():
    # 1. 初始化大模型
    print(">>> 初始化大模型...")
    config_path = "../config/llm_config.yaml"
    try:
        llm = ZhipuChat(config_path)
    except Exception as e:
        print(f"模型初始化失败，请检查配置文件路径: {e}")
        return

    # 2. 初始化分解器和记忆系统
    decoupler = SemanticDecoupler(llm)
    memory_sys = AgenticMemorySystem()

    # 3. 构造一个复杂的测试输入
    input_text = "哎，那个，我之前不是说喜欢吃苹果吗？我现在改主意了，因为苹果太酸了。你知道吗，苹果其实属于蔷薇科。以后每天早上给我准备一根香蕉吧。"
    raw_input = RawInputObj(text=input_text)

    # 4. 执行分解 (调用真实 API)
    print(f"\n>>> 输入文本: {input_text}")
    atoms = decoupler.decouple(raw_input)

    # 5. 展示结果
    print(f"\n>>> 分解成功！共得到 {len(atoms)} 个记忆原子：")
    print("-" * 50)
    for atom in atoms:
        # 使用我们定义的 __repr__ 打印，查看类型和重要性
        print(f"{atom}")
    print("-" * 50)

    # 6. 存入记忆系统并检索测试
    print("\n>>> 存入记忆系统并测试检索 '水果偏好'...")
    for atom in atoms:
        memory_sys.add_note(atom.content)  # 注意：这里为了简单用了 add_note，实际可用 memory_manager.add_memory(atom)

    # 检索
    results = memory_sys.find_related_memories("用户现在喜欢吃什么水果？")
    for res in results:
        print(f"Retrieval Result: {res.content}")


if __name__ == "__main__":
    main()