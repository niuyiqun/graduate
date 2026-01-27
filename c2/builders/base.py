# -*- coding: UTF-8 -*-
# c2/builders/base.py
import logging

logger = logging.getLogger(__name__)

class BaseGraphBuilder:
    """
    [THESIS] 图构建器的基类 (Abstract Base Class)
    所有的构建器（Semantic, Temporal, Structural, Evolution, Emergence）都继承此类。
    """
    def __init__(self, *args, **kwargs):
        pass

    def process(self, new_nodes, graph):
        """
        [THESIS] 核心处理逻辑
        :param new_nodes: 本批次新增的节点列表 (List[MemoryNode])
        :param graph: 全局记忆图谱对象 (MemoryGraph)
        """
        raise NotImplementedError("每个 Builder 子类必须实现 process 方法")