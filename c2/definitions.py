# -*- coding: UTF-8 -*-
# c2/definitions.py

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class NodeType(Enum):
    """
    [THESIS] 对应论文 3.2 节的数据拓扑定义
    🔵 EPISODIC (情景原子): 对应“海马体”的快速流式存储，记录具体事件
    🔴 CONCEPTUAL (概念原子): 对应“新皮层”的慢速结构化存储，记录抽象知识
    """
    EPISODIC = "episodic"
    CONCEPTUAL = "conceptual"


class AtomCategory(Enum):
    """
    细分的原子类别，对应 C1 模块输出的四维正交槽位
    """
    # === 情景流 (Episodic Stream) ===
    ACTIVITY = "episodic_activity"  # 外部行为 (What happened)
    THOUGHT = "episodic_thought"  # 内部思维 (Why it happened)

    # === 语义流 (Semantic Stream) ===
    PROFILE = "semantic_profile"  # 用户画像 (User Attributes)
    KNOWLEDGE = "semantic_knowledge"  # 世界模型 (Fact & Common Sense)

    UNKNOWN = "unknown"


class EdgeType(Enum):
    """
    [THESIS] 对应论文 3.2 节定义的五种边类型
    """
    SEMANTIC = "SEMANTIC"  # [显式] 语义共现：实体共享或逻辑强关联 (Thought <-> Activity)
    TEMPORAL = "TEMPORAL"  # [显式] 时间流：仅连接 Activity，构成时间轴 (Tn -> Tn+1)

    VERSION = "VERSION"  # [演化] 版本更替：当 NLI 检测到冲突时，新节点指向旧节点

    IMPLICIT = "IMPLICIT"  # [隐式] 神经符号推理：由 GNN 召回 + LLM 验证生成的“直觉边”
    ABSTRACT = "ABSTRACT"  # [涌现] 层次整合：由 Concept 节点指向底层的 Event 簇


@dataclass
class MemoryNode:
    """
    记忆图谱的基本单元
    """
    node_id: str
    content: str
    category: AtomCategory
    node_type: NodeType

    # 时间戳 (用于构建 TEMPORAL 边)
    timestamp: Optional[Any] = None

    # [SIMPLIFIED] 生产环境应使用 Vector DB (如 Milvus/Faiss) 存储
    # 这里为了演示直接挂在对象上
    embedding: Optional[List[float]] = None

    # 元数据 (存储来源、置信度等)
    meta: Dict = field(default_factory=dict)

    # [THESIS] 能量值 (Energy Level)
    # 用于 Chapter 3 的“刺激扩散”。Concept 默认高能量(稳态)，Event 默认低能量(需激活)。
    energy_level: float = 1.0

    @staticmethod
    def map_category_to_type(cat_str: str) -> NodeType:
        """辅助函数：根据字符串类别判断是情景还是概念"""
        if "episodic" in str(cat_str):
            return NodeType.EPISODIC
        return NodeType.CONCEPTUAL

    def to_dict(self):
        return {
            "id": self.node_id,
            "content": self.content,
            "category": self.category.value,
            "type": self.node_type.value,
            "timestamp": self.timestamp,
            "energy": self.energy_level
        }