from c2.graph_storage import MemoryGraph

class GraphBuilder:
    def __init__(self, graph: MemoryGraph):
        self.graph = graph

    def build(self):
        """核心构建逻辑"""
        raise NotImplementedError