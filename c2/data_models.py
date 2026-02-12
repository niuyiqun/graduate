"""
c2.data_models - 核心数据结构定义
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import numpy as np


class NodeType(Enum):
    """
    定义图谱节点的认知类型，对应双流架构。
    """
    # Episodic Stream (海马体 - 快速流)
    EVENT_ACTIVITY = "episodic_activity"  # 外部客观行为
    EVENT_THOUGHT = "episodic_thought"  # 内部主观思维

    # Semantic Stream (新皮层 - 慢速流)
    CONCEPT_PROFILE = "semantic_profile"  # 用户画像/偏好
    CONCEPT_KNOWLEDGE = "semantic_knowledge"  # 世界知识/常识
    CONCEPT_ABSTRACT = "emergent_concept"  # [Phase 5] 涌现的高层概念


class EdgeType(Enum):
    """
    定义边的逻辑类型，决定能量扩散的路径。
    """
    TEMPORAL_NEXT = "temporal_next"  # 时序后继 (t -> t+1)
    SEMANTIC_REL = "semantic_rel"  # 语义关联 (Co-occurrence)
    VERSION_UPDATE = "version_update"  # 演化更新 (Old -> New)
    IMPLICIT_LINK = "implicit_link"  # 隐式推理 (GNN Discovered)
    ABSTRACT_UP = "abstraction_up"  # 归纳 (Event -> Concept)


@dataclass
class MemoryNode:
    """图谱中的节点对象"""
    node_id: str
    content: str
    node_type: NodeType
    embedding: np.ndarray  # 高维语义向量
    timestamp: float

    # 认知动力学属性 (Cognitive Dynamics)
    activation_energy: float = 0.0  # 当前激活水平
    base_importance: float = 1.0  # 基础重要性 (Information Weight)
    access_count: int = 0  # 访问次数

    def to_dict(self):
        return {
            "content": self.content,
            "type": self.node_type.value,
            "time": self.timestamp,
            "importance": self.base_importance
        }