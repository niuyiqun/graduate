# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：evolution.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""

# c2/builders/evolution.py
import sys
import os
from typing import List

# === 路径与配置导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入配置
try:
    from ..config import CONFLICT_RETRIEVAL_WINDOW, DECAY_FACTOR, LLM_CONFIG_PATH
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CONFLICT_RETRIEVAL_WINDOW, DECAY_FACTOR, LLM_CONFIG_PATH

# 导入 LLM
try:
    from general.model import ZhipuChat
except ImportError:
    from model import ZhipuChat

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import CONFLICT_DETECTION_PROMPT


class EvolutionBuilder(BaseGraphBuilder):
    """
    [Phase 2] 神经侧 (Neural Side) - 演化与冲突检测
    职责:
    1. 检测新记忆与旧记忆的冲突 (NLI)。
    2. 如果冲突，建立 VERSION 边，并对旧节点降权 (Decay)。
    """

    def __init__(self):
        print(f"  [Evolution] Loading LLM for Conflict Detection...")
        self.llm = ZhipuChat(LLM_CONFIG_PATH)

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Evolution] 正在调用 LLM 进行冲突检测...")

        all_nodes = graph.get_all_nodes()

        # 只与最近的 N 个节点比较 (由 config 控制窗口大小)
        candidates = all_nodes[-CONFLICT_RETRIEVAL_WINDOW:] if len(all_nodes) > CONFLICT_RETRIEVAL_WINDOW else all_nodes

        for new_node in new_nodes:
            for old_node in candidates:
                if new_node.id == old_node.id: continue

                # 调用 LLM 判断冲突
                if self._detect_conflict(new_node.content, old_node.content):
                    print(f"    ⚠️ [Conflict Detected] '{old_node.content[:15]}...' vs '{new_node.content[:15]}...'")

                    # 1. 建立演化边: New -> Old (取代关系)
                    graph.add_edge(new_node.id, old_node.id, EdgeType.VERSION)

                    # 2. 降低旧节点的激活值 (由 config 控制系数)
                    old_node.activation *= DECAY_FACTOR

    def _detect_conflict(self, text_a: str, text_b: str) -> bool:
        """
        利用 LLM 进行 NLI (Natural Language Inference) 推理
        """
        prompt = CONFLICT_DETECTION_PROMPT.format(text_a=text_a, text_b=text_b)
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat(messages)
            # 解析结果：如果包含 "YES"，则认为冲突
            if isinstance(result, dict):
                content = result.get("content", "NO").upper()
            else:
                content = str(result).upper()

            return "YES" in content
        except Exception as e:
            print(f"    [Evolution] Error: {e}")
            return False