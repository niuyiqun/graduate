"""
c2.processors.inference - 结构化隐式推理算子
"""
import networkx as nx
from ..data_models import EdgeType, MemoryNode
from ..config import SystemConfig


class TopologyInferenceProcessor:
    def process_implicit_links(self, graph_kernel, new_nodes: list):
        """
        基于图拓扑结构发现隐式关联 (Implicit Recall)。
        算法：使用 Adamic-Adar Index 预测潜在链接。
        """
        # 转换为无向图视图以计算拓扑指标
        undirected_view = graph_kernel.graph.to_undirected()

        new_ids = [n.node_id for n in new_nodes]
        all_ids = list(graph_kernel.graph.nodes())

        # 生成候选对：(新节点, 任意非邻接节点)
        candidates = []
        for u in new_ids:
            for v in all_ids:
                if u != v and not graph_kernel.graph.has_edge(u, v):
                    candidates.append((u, v))

        if not candidates: return

        # 计算 Adamic-Adar 指标 (Common Neighbor 的加权变体)
        # 这一步体现了 "符号侧 (Symbolic Side)" 的推理能力
        preds = nx.adamic_adar_index(undirected_view, candidates)

        for u, v, score in preds:
            if score > SystemConfig.TOPOLOGY_INFERENCE_THRESHOLD:
                # 写入隐式边
                graph_kernel.add_edge(
                    u, v,
                    EdgeType.IMPLICIT_LINK,
                    weight=score * 0.5  # 隐式边的权重通常低于显式边
                )