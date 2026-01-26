# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：decoupler.py
@Desc    ：研究内容一(1)：双流语义正交分解器 (Dual-Stream Orthogonal Decoupler)
          负责将非结构化对话流分解为 Profile, Knowledge, Activity, Thought 四个正交视图。

          【适配说明】
          已针对 Qwen2.5-GRPO 模型进行适配：
          1. 增强 JSON 解析能力 (支持 Markdown 包裹、裸 JSON、嵌套字典)。
          2. 增加结构化数据拍扁 (Flatten) 逻辑。
"""

import time
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from general.decoupled_memory import DecoupledMemoryAtom
    from general.model import BaseModel
    # 确保导入了您定义的 Prompt
    from c1.prompts import DecouplerPrompt
except ImportError as e:
    pass


@dataclass
class RawInputObj:
    """
    [Data Object] 原始输入封装
    """
    text: str  # 【当前轮次】的内容 (Target)
    context: str = ""  # 【历史对话】的内容 (Context)
    timestamp: float = None
    source: str = "user"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SemanticDecoupler:
    """
    【核心组件】双流语义正交分解器

    架构设计 (Architecture):
    -------------------------------------------------------
    基于“内外双流 (Inner-Outer Dual-Stream)”认知架构，将记忆分解为：
    1. Semantic Stream (静态/抽象): [semantic_profile], [semantic_knowledge]
    2. Episodic Stream (动态/具体): [episodic_activity], [episodic_thought]
    -------------------------------------------------------
    """

    def __init__(self, llm_model: BaseModel):
        self.llm = llm_model
        # 预编译正则，提升解析速度
        self.json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def decouple(self, raw_input: RawInputObj) -> List[Any]:
        """
        [主入口] 执行分解流水线
        Args:
            raw_input: 包含当前句和上下文的输入对象
        Returns:
            List[DecoupledMemoryAtom]: 提取出的记忆原子列表
        """

        # 1. 构造 Prompt (关键：使用 c1/prompts.py 定义的模板)
        system_content = DecouplerPrompt.SYSTEM
        user_content = DecouplerPrompt.build_user_input(
            history_text=raw_input.context,
            current_turn_text=raw_input.text
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # print(f"  >>> [Decoupler] Processing: {raw_input.text[:30]}...")

        # 2. LLM 推理
        # QwenGRPOChat 会自动尝试解析，但为了稳健性，这里包含二次解析逻辑
        response = self.llm.chat(messages)

        # 3. 数据清洗与标准化 (Adapter Logic)
        data = {}

        # 情况 A: 模型返回了字典
        if isinstance(response, dict):
            # 检查是否嵌套在 'content' 或 'answer' 字段中
            if "semantic_profile" in response:
                data = response
            elif "content" in response:
                content = response["content"]
                if isinstance(content, dict):
                    data = content
                elif isinstance(content, str):
                    data = self._try_parse_json(content)
            elif "raw_response" in response:
                data = self._try_parse_json(response["raw_response"])
            else:
                data = response

        # 情况 B: 模型返回了字符串 (Fallback)
        elif isinstance(response, str):
            data = self._try_parse_json(response)

        # 4. 提取并封装为 Atom 对象
        atoms = []
        target_keys = ["semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"]
        formatted_time = self._format_time(raw_input.timestamp)

        if isinstance(data, dict):
            for key in target_keys:
                items = data.get(key, [])
                if not items: continue

                # 处理 List 类型 (标准情况)
                if isinstance(items, list):
                    for item in items:
                        atom_content = self._format_atom_content(item)
                        if atom_content:
                            self._add_atom(atoms, atom_content, key, raw_input.text, formatted_time)

                # 处理 Dict 类型 (容错情况)
                elif isinstance(items, dict):
                    for k, v in items.items():
                        atom_content = f"{k}: {v}"
                        self._add_atom(atoms, atom_content, key, raw_input.text, formatted_time)

        # print(f"      L-> Extracted {len(atoms)} atoms.")
        return atoms

    def _add_atom(self, atoms_list, content, atom_type, source_text, timestamp):
        """辅助：创建并添加 Atom 对象"""
        # 简单的长度过滤
        if len(content.split()) < 2: return

        try:
            from general.decoupled_memory import DecoupledMemoryAtom
            atom = DecoupledMemoryAtom(
                content=content,
                atom_type=atom_type,
                source_text=source_text,
                timestamp=timestamp
            )
            atoms_list.append(atom)
        except ImportError:
            # 本地调试时的 fallback
            atoms_list.append({"content": content, "atom_type": atom_type})

    def _format_atom_content(self, item: Any) -> str:
        """辅助：将 item 转为字符串"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return ", ".join([f"{k}={v}" for k, v in item.items()])
        return str(item)

    def _try_parse_json(self, text: str) -> Dict:
        """辅助：强力 JSON 提取"""
        try:
            match = self.json_pattern.search(text)
            if match: return json.loads(match.group(1))
            start = text.find("{")
            end = text.rfind("}")
            if start != -1: return json.loads(text[start:end + 1])
            return json.loads(text)
        except:
            return {}

    def _format_time(self, timestamp: float) -> str:
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        except:
            return str(timestamp)