# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：decoupler.py
@Desc    ：研究内容一(1)：双流语义正交分解器 (Dual-Stream Orthogonal Decoupler)
          ✅ 更新：RawInputObj 支持传入自定义 timestamp (str 或 float)
          ✅ 更新：Atom 生成时携带完整的源数据 (Timestamp, Source Text)
"""

import time
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from general.decoupled_memory import DecoupledMemoryAtom
    from general.model import BaseModel
    from c1.prompts import DecouplerPrompt
except ImportError as e:
    pass


@dataclass
class RawInputObj:
    """
    [Data Object] 原始输入封装
    支持传入 Locomo 的字符串时间戳
    """
    text: str
    context: str = ""
    timestamp: Union[float, str, None] = None
    source: str = "user"

    def __post_init__(self):
        # 如果没传时间，用当前系统时间
        if self.timestamp is None:
            self.timestamp = time.time()


class SemanticDecoupler:
    def __init__(self, llm_model: BaseModel):
        self.llm = llm_model
        self.json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def decouple(self, raw_input: RawInputObj) -> List[Any]:
        """主提取逻辑"""

        # 1. Prompt
        system_content = DecouplerPrompt.SYSTEM
        user_content = DecouplerPrompt.build_user_input(
            history_text=raw_input.context,
            current_turn_text=raw_input.text
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # 2. Inference
        response = self.llm.chat(messages)

        # 3. Parsing
        data = {}
        if isinstance(response, dict):
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
        elif isinstance(response, str):
            data = self._try_parse_json(response)

        # 4. Atom Creation (带时间戳)
        atoms = []
        target_keys = ["semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"]

        # 处理时间：如果是 float 转 str，如果是 str (Locomo) 直接用
        final_time_str = raw_input.timestamp
        if isinstance(final_time_str, float):
            final_time_str = self._format_time(final_time_str)

        if isinstance(data, dict):
            for key in target_keys:
                items = data.get(key, [])
                if not items: continue

                if isinstance(items, list):
                    for item in items:
                        atom_content = self._format_atom_content(item)
                        if atom_content:
                            # 传入 source_text 和 timestamp
                            self._add_atom(atoms, atom_content, key, raw_input.text, final_time_str)

                elif isinstance(items, dict):
                    for k, v in items.items():
                        atom_content = f"{k}: {v}"
                        self._add_atom(atoms, atom_content, key, raw_input.text, final_time_str)

        return atoms

    def _add_atom(self, atoms_list, content, atom_type, source_text, timestamp):
        """创建原子，填充完整元数据"""
        # 噪音过滤：太短的不要
        if len(content.split()) < 2: return

        try:
            from general.decoupled_memory import DecoupledMemoryAtom
            atom = DecoupledMemoryAtom(
                content=content,
                atom_type=atom_type,
                source_text=source_text,  # 记录来源
                timestamp=timestamp,  # 记录时间
                confidence=1.0
            )
            atoms_list.append(atom)
        except ImportError:
            atoms_list.append({"content": content, "atom_type": atom_type})

    def _format_atom_content(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return ", ".join([f"{k}={v}" for k, v in item.items()])
        return str(item)

    def _try_parse_json(self, text: str) -> Dict:
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