from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

# --- Core Enums based on Methodology ---

class NodeType(Enum):
    """
    映射论文中的两类核心原子:
    🔵 情景原子 (Episodic Atom): 承载动态流 (Activity, Thought)
    🔴 概念原子 (Conceptual Atom): 承载静态流 (Profile, Knowledge)
    """
    EPISODIC = "episodic"
    CONCEPTUAL = "conceptual"

class AtomCategory(Enum):
    """C1 输出的四维正交槽位"""
    PROFILE = "semantic_profile"
    KNOWLEDGE = "semantic_knowledge"
    ACTIVITY = "episodic_activity"
    THOUGHT = "episodic_thought"

class EdgeType(Enum):
    """
    映射论文定义的五种边类型:
    """
    SEMANTIC = "SEMANTIC"   # 语义共现 / 知行合一 (Thought <-> Activity)
    TEMPORAL = "TEMPORAL"   # 时间流 (Activity t -> Activity t+1)
    VERSION = "VERSION"     # 演化更替 (Old -> New)
    IMPLICIT = "IMPLICIT"   # 隐式推理 (GNN Discovery)
    ABSTRACT = "ABSTRACT"   # 层次整合 (Concept -> Event Cluster)

@dataclass
class MemoryNode:
    """
    图谱中的节点对象
    """
    node_id: str
    content: str
    category: AtomCategory
    node_type: NodeType
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def map_category_to_type(category_str: str) -> NodeType:
        # 处理可能的前缀或直接匹配
        if "episodic" in category_str:
            return NodeType.EPISODIC
        return NodeType.CONCEPTUAL