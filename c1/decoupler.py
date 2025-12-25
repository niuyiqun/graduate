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
from typing import List, Dict, Any
from dataclasses import dataclass

# 导入你定义好的数据结构
from general.decoupled_memory import DecoupledMemoryAtom

# 导入你的模型基类用于类型提示
from general.model import BaseModel


@dataclass
class RawInputObj:
    """简单的原始输入封装"""
    text: str
    timestamp: float = None
    source: str = "user"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SemanticDecoupler:
    """
    【研究内容一核心算法】多粒度语义正交分解器
    接入 ZhipuChat 或 LlamaChat 进行真实推理
    """

    def __init__(self, llm_model: BaseModel):
        """
        :param llm_model: 传入初始化好的 ZhipuChat 或 LlamaChat 实例
        """
        self.llm = llm_model

        # System Prompt: 对应 PPT 中的“构建高维语义投影空间”
        # 注意：这里明确要求使用 ```json 包裹，适配 ZhipuChat 的解析逻辑
        self.system_prompt = """
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

    def decouple(self, raw_input: RawInputObj) -> List[DecoupledMemoryAtom]:
        """执行分解流程"""

        # 1. 构造 Messages (适配 ZhipuChat/LlamaChat 的输入格式)
        user_content = f"Time: {raw_input.timestamp}\nText: {raw_input.text}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        print(f"--- [Decoupler] 正在调用大模型分解: {raw_input.text[:20]}... ---")

        # 2. LLM 推理
        # 注意：你的 ZhipuChat.chat 返回的已经是解析好的 dict，或者包含 error 的 dict
        response_data = self.llm.chat(messages)

        # 3. 错误处理
        if "error" in response_data:
            print(f"[Error] 模型调用失败: {response_data.get('details')}")
            # 降级策略：把整句话当做一个 Event 存下来
            return [DecoupledMemoryAtom(content=raw_input.text, atom_type="event", source_text=raw_input.text)]

        # 4. 封装为记忆原子
        atoms = []

        # 你的 ZhipuChat 已经把 JSON 解析为字典了，直接用
        parsed_data = response_data

        # 定义类型映射，防止 key 大小写问题或复数问题
        type_map = {
            'event': 'event', 'events': 'event',
            'entity': 'entity', 'entities': 'entity',
            'knowledge': 'knowledge',
            'rule': 'rule', 'rules': 'rule'
        }

        for key, content_list in parsed_data.items():
            # 过滤掉非预定义的 key (比如 step_by_step_thinking)
            norm_type = type_map.get(key.lower())
            if not norm_type:
                continue

            if isinstance(content_list, list):
                for content in content_list:
                    atom = DecoupledMemoryAtom(
                        content=content,
                        atom_type=norm_type,
                        source_text=raw_input.text,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(raw_input.timestamp))
                    )
                    atoms.append(atom)

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