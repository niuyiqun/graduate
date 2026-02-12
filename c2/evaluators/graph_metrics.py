"""
c2.evaluators.graph_metrics - 记忆图谱质量评测
职责: 计算拓扑指标，验证 "高信噪比" 和 "结构化程度"。
"""
import networkx as nx
import numpy as np
import logging
from ..data_models import EdgeType, NodeType


class GraphEvaluator:
    def __init__(self, graph_kernel):
        self.G = graph_kernel.graph
        self.logger = logging.getLogger("GraphEvaluator")

    def evaluate_all(self):
        """运行所有评测指标并返回字典"""
        metrics = {
            "1. Topology Metrics": self._topology_metrics(),
            "2. Cognitive Density": self._cognitive_density(),
            "3. Evolution Quality": self._evolution_metrics(),
            "4. Abstraction Level": self._abstraction_metrics()
        }
        self._print_report(metrics)
        return metrics

    def _topology_metrics(self):
        """计算基础拓扑指标"""
        # 图密度 (Graph Density): 衡量记忆关联的致密程度
        density = nx.density(self.G)

        # 平均路径长度 (Avg Path Length): 衡量推理跳跃的效率
        # 仅在最大连通分量上计算
        if len(self.G) > 0:
            undirected = self.G.to_undirected()
            largest_cc = max(nx.connected_components(undirected), key=len)
            subgraph = undirected.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(subgraph)
        else:
            avg_path = 0

        # 聚类系数 (Clustering Coefficient): 衡量知识的局部聚集性
        clustering = nx.average_clustering(self.G.to_undirected())

        return {
            "Density": f"{density:.4f}",
            "Avg Path Length": f"{avg_path:.2f}",
            "Clustering Coef": f"{clustering:.4f}"
        }

    def _cognitive_density(self):
        """认知密度：有效信息与节点总数的比率"""
        total_nodes = self.G.number_of_nodes()
        total_edges = self.G.number_of_edges()
        if total_nodes == 0: return 0

        # 假设：拥有 Semantic 或 Implicit 边的节点参与了高层认知
        active_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                        if d.get('type') in [EdgeType.SEMANTIC_REL, EdgeType.IMPLICIT_LINK]]

        return {
            "Cognitive Connection Ratio": f"{len(active_edges) / total_edges:.2%}" if total_edges else "0",
            "Mean Degree": f"{total_edges / total_nodes:.2f}"
        }

    def _evolution_metrics(self):
        """演化质量：冲突消解的比例"""
        version_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                         if d.get('type') == EdgeType.VERSION_UPDATE]

        profiles = [n for n, d in self.G.nodes(data=True)
                    if d['data'].node_type == NodeType.CONCEPT_PROFILE]

        return {
            "Total Conflicts Resolved": len(version_edges),
            "Evolution Rate": f"{len(version_edges) / len(profiles):.2%}" if profiles else "0"
        }

    def _abstraction_metrics(self):
        """抽象水平：概念节点与事件节点的比例 (Simulate Neocortex/Hippocampus ratio)"""
        concepts = [n for n, d in self.G.nodes(data=True)
                    if d['data'].node_type in [NodeType.CONCEPT_KNOWLEDGE, NodeType.CONCEPT_ABSTRACT]]
        events = [n for n, d in self.G.nodes(data=True)
                  if d['data'].node_type in [NodeType.EVENT_ACTIVITY, NodeType.EVENT_THOUGHT]]

        ratio = len(concepts) / len(events) if events else 0

        return {
            "Concept Nodes": len(concepts),
            "Event Nodes": len(events),
            "C/E Ratio (Abstraction)": f"{ratio:.2f}"
        }

    def _print_report(self, metrics):
        print("\n" + "=" * 50)
        print("MEMORY GRAPH EVALUATION REPORT")
        print("=" * 50)
        for category, values in metrics.items():
            print(f"\n[{category}]")
            for k, v in values.items():
                print(f"  - {k:<25}: {v}")
        print("=" * 50 + "\n")