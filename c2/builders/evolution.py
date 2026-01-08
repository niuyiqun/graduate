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

# === 路径设置：为了导入 model.py ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入 ZhipuChat
try:
    from general.model import ZhipuChat
except ImportError:
    sys.path.append("..")
    from model import ZhipuChat

from .base import BaseGraphBuilder
from ..definitions import EdgeType, GraphNode
from ..graph_storage import AtomGraph
from ..prompts import CONFLICT_DETECTION_PROMPT  # 导入我们在 prompts.py 定义的提示词


class EvolutionBuilder(BaseGraphBuilder):
    """
    [Phase 2] 演化侧 (Evolution Side)
    利用 LLM 进行 NLI (自然语言推理) 检测冲突，建立版本演进边。
    """

    def __init__(self):
        # 1. 初始化 LLM
        config_path = os.path.join(project_root, "config/llm_config.yaml")
        if not os.path.exists(config_path):
            config_path = "./config/llm_config.yaml"

        print(f"  [Evolution] Loading LLM for Conflict Detection...")
        self.llm = ZhipuChat(config_path)

    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        print("  [Evolution] 正在调用 LLM 进行冲突检测...")
        all_nodes = graph.get_all_nodes()

        for new_node in new_nodes:
            # === 1. 检索候选 (Retrieval) ===
            # TODO: 未来这里应该用 Vector DB (chroma/faiss) 做 Top-K 检索
            # 目前暂时暴力遍历最近的 10 个节点作为演示
            candidates = all_nodes[-10:]

            for old_node in candidates:
                # 跳过自己
                if new_node.id == old_node.id: continue

                # === 2. 真实 LLM 冲突检测 (NLI) ===
                # 只有当两个节点都有 Embedding 且相似度高时才检测冲突 (节省 Token)
                # 这里为了演示效果，先不做相似度过滤，直接检测

                if self._check_conflict(old_node, new_node):
                    print(f"    ⚠️ [Conflict Detected] '{old_node.content[:10]}...' vs '{new_node.content[:10]}...'")

                    # 3. 建立版本演进边 (Old -> New)
                    graph.add_edge(old_node.id, new_node.id, EdgeType.VERSION)

                    # 4. 旧节点降权
                    old_node.activation *= 0.5

    def _check_conflict(self, n1: GraphNode, n2: GraphNode) -> bool:
        """
        调用大模型判断两个节点内容是否冲突
        """
        # 1. 组装 Prompt (使用 prompts.py 中的模板)
        prompt = CONFLICT_DETECTION_PROMPT.format(old_text=n1.content, new_text=n2.content)
        messages = [{"role": "user", "content": prompt}]

        # 2. 调用 LLM
        try:
            # model.py 会自动解析 JSON 返回字典
            result = self.llm.chat(messages)

            if isinstance(result, dict):
                # 获取结果，默认为 NO
                is_conflict = result.get("is_conflict", "NO").upper()
                return "YES" in is_conflict

        except Exception as e:
            print(f"  [Evolution] LLM NLI Error: {e}")

        return False