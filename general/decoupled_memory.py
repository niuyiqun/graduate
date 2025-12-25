# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：decoupled_memory.py
@Author  ：niu
@Date    ：2025/12/24 12:37 
@Desc    ：
"""

# general/decoupled_memory.py
from dataclasses import dataclass
from typing import Optional
from .base_memory import MemoryNote  # 导入基类


class DecoupledMemoryAtom(MemoryNote):
    """
    研究内容一：经过正交分解后的记忆原子
    继承自 base_memory.MemoryNote，复用其 ID、timestamp 等基础字段
    """

    def __init__(self,
                 content: str,
                 atom_type: str,  # 类型: 'event', 'entity', 'knowledge', 'rule'
                 source_text: str = "",  # 来源的原始文本，用于溯源
                 confidence: float = 1.0,  # 置信度
                 **kwargs):

        # 调用父类初始化
        # 注意：MemoryNote 的 __init__ 可能需要根据您 base_memory.py 的最新版本调整参数传递
        super().__init__(content=content, **kwargs)

        self.atom_type = atom_type
        self.source_text = source_text
        self.confidence = confidence

        # 策略模式：不同类型的记忆原子，初始重要性不同
        # 这对应 PPT 中“自适应压缩”的前置权重
        if atom_type == 'rule':
            self.importance_score = 1.5  # 规则/偏好最重要，不轻易遗忘
        elif atom_type == 'knowledge':
            self.importance_score = 1.2  # 事实知识次之
        elif atom_type == 'event':
            self.importance_score = 1.0  # 普通事件正常
        else:
            self.importance_score = 0.8  # 实体提及如果没上下文，权重较低

    def extra_info(self) -> str:
        """重写父类抽象方法，提供给检索器的额外上下文"""
        return f"[{self.atom_type.upper()}] Conf:{self.confidence:.2f} | Src:{self.source_text[:10]}..."

    def __repr__(self):
        # 调试打印时更直观
        return f"<{self.atom_type.upper()}> {self.content} (Imp:{self.importance_score})"
