from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


class AtomType(Enum):
    # 情景流 (Episodic)
    ACTIVITY = "episodic_activity"  # 外部行为
    THOUGHT = "episodic_thought"  # 内部思维
    # 语义流 (Semantic)
    PROFILE = "semantic_profile"  # 用户画像
    KNOWLEDGE = "semantic_knowledge"  # 世界模型


class EdgeType(Enum):
    TEMPORAL = "temporal"  # 时间流 Tn -> Tn+1
    SEMANTIC = "semantic"  # 语义强关联 (知行合一)
    VERSION = "version"  # 演化更替 (旧 -> 新)
    IMPLICIT = "implicit"  # 隐式推理 (GNN 发现)
    ABSTRACT = "abstract"  # 概念归纳 (自底向上)


@dataclass
class MemoryAtom:
    content: str
    atom_type: AtomType
    embedding: List[float] = field(default_factory=list)  # 模拟向量
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # C1 阶段计算的信息权重 (Chapter 1 Output)
    info_weight: float = 1.0


@dataclass
class GraphNode:
    id: str
    atom: MemoryAtom
    mass: float = 0.0  # 当前激活能量 (C3 使用)
    base_mass: float = 0.0  # 静态质量 (由 info_weight 决定)