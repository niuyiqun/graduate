# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：decoupler.py.py
@Author  ：niu
@Date    ：2025/12/24 12:33 
@Desc    ：
"""

# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：decoupler.py
@Desc    ：研究内容一(1)：双流语义正交分解器 (Dual-Stream Orthogonal Decoupler)
          负责将非结构化对话流分解为 Profile, Knowledge, Activity, Thought 四个正交视图。
"""

import time
import json
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# --- 路径适配 (确保能导入根目录下的 general 模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 模块导入 ---
try:
    from general.decoupled_memory import DecoupledMemoryAtom
    from general.model import BaseModel
    # 导入 Prompt 定义，确保逻辑源头统一
    from c1.prompts import DecouplerPrompt
except ImportError as e:
    raise ImportError(f"模块导入失败，请确保项目结构正确 (c1/ 和 general/ 同级): {e}")


@dataclass
class RawInputObj:
    """
    [Data Object] 原始输入封装
    用于标准化进入 Pipeline 的非结构化文本流，保留时序信息。
    """
    text: str
    timestamp: float = None
    source: str = "user"  # 或 'dialogue_stream'

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SemanticDecoupler:
    """
    【核心组件】双流语义正交分解器

    架构升级说明 (Architecture Upgrade):
    -------------------------------------------------------
    基于“内外双流 (Inner-Outer Dual-Stream)”认知架构，将记忆分解为：

    1. Semantic Stream (静态/抽象):
       - [semantic_profile]: 用户模型 (User Model) - 属性、偏好
       - [semantic_knowledge]: 世界模型 (World Model) - 事实、常识

    2. Episodic Stream (动态/具体):
       - [episodic_activity]: 外部世界 (Outer World) - 行为、事件
       - [episodic_thought]: 内部世界 (Inner World) - 意图、情绪、动机
    -------------------------------------------------------
    """

    def __init__(self, llm_model: BaseModel):
        """
        初始化分解器
        :param llm_model: 已初始化的 LLM 实例 (如 ZhipuChat)
        """
        self.llm = llm_model
        # 直接使用 c1/prompts.py 中定义的 System Prompt
        self.system_prompt = DecouplerPrompt.SYSTEM

    def decouple(self, raw_input: RawInputObj) -> List[DecoupledMemoryAtom]:
        """
        [主入口] 执行分解流水线 (Execution Pipeline)

        Args:
            raw_input: 包含文本和时间戳的原始输入对象

        Returns:
            List[DecoupledMemoryAtom]: 分解后的记忆原子列表
        """

        # 1. 构造上下文 (Context Construction)
        # 使用 Prompt 类提供的静态方法构建用户输入 (可扩展性更好)
        user_content = DecouplerPrompt.build_user_input(raw_input.text)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 打印日志方便调试 (截取前30字符，去除换行)
        log_text = raw_input.text[:30].replace('\n', ' ')
        print(f"--- [Decoupler] 正在调用大模型分解: {log_text}... ---")

        # 2. LLM 推理 (Inference)
        response_data = self.llm.chat(messages)

        # 3. 鲁棒性工程：错误处理与熔断 (Error Handling)
        if "error" in response_data:
            print(f"[Error] 模型调用失败: {response_data.get('details')}")
            # 降级策略 (Fallback):
            # 如果解析失败，将整段话作为未分类的 'episodic_activity' 存入，保证不丢数据
            return [DecoupledMemoryAtom(
                content=raw_input.text,
                atom_type="episodic_activity",
                source_text=raw_input.text,
                timestamp=self._format_time(raw_input.timestamp)
            )]

        # 4. 原子化与归一化 (Atomization & Normalization)
        atoms = []
        parsed_json = response_data  # 假设 LLM 基类已处理 JSON 解析，返回 dict

        # 定义需要提取的目标 Key (对应四视图)
        target_keys = [
            "semantic_profile",
            "semantic_knowledge",
            "episodic_activity",
            "episodic_thought"
        ]

        formatted_time = self._format_time(raw_input.timestamp)

        for key in target_keys:
            # 容错处理：获取数据，防范 LLM 偶尔输出复数 key (e.g., semantic_profiles)
            content_list = parsed_json.get(key) or parsed_json.get(key + 's')

            if not content_list or not isinstance(content_list, list):
                continue

            for content in content_list:
                clean_content = str(content).strip()

                # --- [关键清洗步骤] 实体过滤 ---
                # 英文语境下，如果内容少于 3 个单词，极大概率是残留的单一实体 (如 "Dogs", "The sky")
                # 这种碎片对记忆检索无用且有害，直接丢弃
                if len(clean_content.split()) < 3:
                    # print(f"  [Filter] 丢弃过短原子(实体残留): {clean_content}")
                    continue

                # 实例化原子对象
                atom = DecoupledMemoryAtom(
                    content=clean_content,
                    atom_type=key,  # 直接使用新的类型名
                    source_text=raw_input.text,  # 保留溯源信息
                    timestamp=formatted_time
                )
                atoms.append(atom)

        print(f"  -> 初步提取 {len(atoms)} 条原子")
        return atoms

    def _format_time(self, timestamp: float) -> str:
        """辅助函数：格式化时间戳"""
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        except Exception:
            return str(timestamp)


# test_real_llm.py
import sys
import os

# 确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from general.model import ZhipuChat
from general.base_memory import AgenticMemorySystem

if __name__ == "__main__":
    print(">>> [Demo] 初始化双流语义解耦器...")

    # 1. 尝试初始化 LLM (需要确保 config 路径正确)
    try:
        from general.model import ZhipuChat

        # 假设 config 在项目根目录的 config 文件夹下
        config_path = os.path.join(root_dir, "config", "llm_config.yaml")
        llm = ZhipuChat(config_path)
    except Exception as e:
        print(f"[Fatal] 模型初始化失败: {e}")
        print("请检查 config/llm_config.yaml 是否存在且配置正确。")
        exit()

    # 2. 实例化解耦器
    decoupler = SemanticDecoupler(llm)

    # 3. 构造模拟数据 (LoCoMo 风格双人对话)
    # 场景设计：
    # - Activity: 去过爵士俱乐部 (Past)
    # - Profile: Alex 是夜猫子，吃素 (Static)
    # - Knowledge: 提到天气预报/常识 (Objective)
    # - Thought: 期待徒步 (Emotion)，打算买零食 (Intent)

    mock_dialogue = """
[Jordan]: Hey Alex, I went to that new jazz club last night. The live performance was amazing!
[Alex]: Oh nice. I actually stayed home. I've been trying to fix my sleep schedule, but being a night owl is really hard.
[Jordan]: I feel you. By the way, are we still on for hiking this Saturday? The weather forecast says it will be sunny and perfect for climbing.
[Alex]: Definitely. I'm really looking forward to getting out of the city. I need to buy some vegetarian snacks for the trip though, maybe tomorrow.
    """

    print(f"\n{'=' * 60}")
    print(f"Input Dialogue Stream (LoCoMo Style):")
    print(f"{'=' * 60}")
    print(mock_dialogue.strip())
    print(f"{'=' * 60}\n")

    # 4. 执行解耦
    raw_input = RawInputObj(text=mock_dialogue)
    start_time = time.time()
    atoms = decoupler.decouple(raw_input)
    cost_time = time.time() - start_time

    # 5. 格式化输出结果 (按四视图分类展示)
    print(f"\n>>> 解耦完成 (耗时: {cost_time:.2f}s) | 共提取 {len(atoms)} 个原子")

    # 简单的分类桶
    views = {
        "semantic_profile": [],
        "semantic_knowledge": [],
        "episodic_activity": [],
        "episodic_thought": []
    }

    for atom in atoms:
        if atom.atom_type in views:
            views[atom.atom_type].append(atom.content)
        else:
            # 防御性编程：如果有漏网之鱼
            print(f"[Warn] Unknown Type: {atom.atom_type} -> {atom.content}")

    # 打印结果矩阵
    print("\n" + "#" * 20 + " Inner-Outer Dual-Stream Architecture " + "#" * 20)

    # A. Semantic Stream
    print(f"\n【A. Semantic Stream (Static)】")
    print(f"  1. [Profile] (User Model):")
    for item in views["semantic_profile"]:
        print(f"     - {item}")

    print(f"  2. [Knowledge] (World Model):")
    for item in views["semantic_knowledge"]:
        print(f"     - {item}")

    # B. Episodic Stream
    print(f"\n【B. Episodic Stream (Dynamic)】")
    print(f"  3. [Activity] (Outer World / History):")
    for item in views["episodic_activity"]:
        print(f"     - {item}")

    print(f"  4. [Thought] (Inner World / Intent & Emotion):")
    for item in views["episodic_thought"]:
        print(f"     - {item}")

    print("\n" + "#" * 78)

