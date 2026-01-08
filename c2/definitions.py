# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：definitions.py.py
@Author  ：niu
@Date    ：2026/1/8 13:24 
@Desc    ：
"""

# c2/definitions.py
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
import torch

# === 尝试导入第一章的定义 ===
# 确保你的 general 包在父目录下
sys.path.append("..")
try:
    from general.decoupled_memory import DecoupledMemoryAtom
except ImportError:
    # 如果没找到，定义一个简单的 Mock 类防止报错
    @dataclass
    class DecoupledMemoryAtom:
        content: str
        atom_type: str = "event"
        id: str = "mock_id"
        timestamp: str = "2023-01-01 00:00:00"


class EdgeType(Enum):
    """
    [Design Spec] 四维边属性定义
    """
    # 1. 神经侧 (LLM) 构建: 显式语义 (实体共现)
    SEMANTIC = "SEMANTIC"

    # 2. 规则侧 构建: 时序邻接 (对话流)
    TEMPORAL = "TEMPORAL"

    # 3. 演化侧 (LLM) 构建: 版本演进 (冲突修正)
    VERSION = "EVOLVES_TO"

    # 4. 符号侧 (GNN) 构建: 隐式逻辑 (结构推理)
    IMPLICIT = "IMPLICIT"


@dataclass
class MemoryEdge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphNode:
    """
    [Graph Node] 图节点容器
    封装 DecoupledMemoryAtom，增加图计算属性
    """

    def __init__(self, atom: DecoupledMemoryAtom):
        self.atom = atom
        self.edges: List[MemoryEdge] = []

        # --- 图计算专用属性 ---
        self.entities: Set[str] = set()  # 显式实体 (LLM 提取)
        self.embedding: Optional[torch.Tensor] = None  # 语义向量 (用于 GNN)
        self.activation: float = 1.0  # 激活权重

    @property
    def id(self): return self.atom.id

    @property
    def content(self): return self.atom.content

    @property
    def timestamp(self): return self.atom.timestamp

    def __repr__(self):
        return f"<Node {self.id[-4:]}: {self.content[:10]}...>"
