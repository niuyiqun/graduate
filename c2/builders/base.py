# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：base.py
@Author  ：niu
@Date    ：2026/1/8 13:25 
@Desc    ：
"""

# c2/builders/base.py
from abc import ABC, abstractmethod
from typing import List
from ..graph_storage import AtomGraph
from ..definitions import GraphNode

class BaseGraphBuilder(ABC):
    """
    图构建器标准接口
    """
    @abstractmethod
    def process(self, new_nodes: List[GraphNode], graph: AtomGraph):
        pass