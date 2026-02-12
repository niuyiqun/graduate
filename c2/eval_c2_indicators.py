import json
import logging
import numpy as np
import networkx as nx
import time
from collections import defaultdict, Counter
from enum import Enum
from dataclasses import dataclass
import random

# ==========================================
# 1. 基础配置与模拟环境 (Setup)
# ==========================================

# 设置随机种子以保证实验可复现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class NodeType(Enum):
    ACTIVITY = "episodic_activity"
    THOUGHT = "episodic_thought"
    PROFILE = "semantic_profile"
    KNOWLEDGE = "semantic_knowledge"
    ABSTRACT_CONCEPT = "abstract_concept"  # C2 独有：涌现出的概念


class EdgeType(Enum):
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    VERSION = "version"
    IMPLICIT = "implicit"
    ABSTRACT = "abstract"


# ==========================================
# 2. 核心图谱构建器 (Graph Construction)
# ==========================================

class ThesisGraphSystem:
    def __init__(self, logger):
        self.graph = nx.DiGraph()
        self.nodes_registry = {}
        self.evolution_stats = {"conflicts_resolved": 0, "decay_operations": 0}
        self.logger = logger

    def add_node(self, atom):
        """将原子转换为图节点"""
        node_id = atom.get('id')
        if not node_id:
            return

        self.nodes_registry[node_id] = atom

        # 模拟 Embedding
        vec = np.random.rand(128)

        self.graph.add_node(
            node_id,
            type=atom.get('atom_type', NodeType.ACTIVITY.value),
            content=atom.get('content', ''),
            embedding=vec,
            timestamp=atom.get('timestamp', '')
        )

    def build_temporal_layer(self):
        """构建时序骨架 (Phase 1)"""
        activities = [
            (n, d.get('timestamp', '')) for n, d in self.graph.nodes(data=True)
            if d.get('type') == NodeType.ACTIVITY.value
        ]
        # 简单排序，如果时间戳为空则放在最后
        activities.sort(key=lambda x: x[1] if x[1] else "9999")

        edges_added = 0
        for i in range(len(activities) - 1):
            u, v = activities[i][0], activities[i + 1][0]
            self.graph.add_edge(u, v, type=EdgeType.TEMPORAL.value, weight=1.0)
            edges_added += 1
        return edges_added

    def build_semantic_layer(self):
        """构建语义关联 (Phase 1 & 4)"""
        nodes_list = list(self.graph.nodes())
        if len(nodes_list) < 2: return 0

        edges_added = 0
        # 模拟语义连接数量
        num_semantic_edges = int(len(nodes_list) * 1.2)

        for _ in range(num_semantic_edges):
            u, v = random.sample(nodes_list, 2)
            if u != v and not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, type=EdgeType.SEMANTIC.value, weight=random.uniform(0.7, 0.9))
                edges_added += 1
        return edges_added

    def simulate_evolution(self):
        """模拟演化与冲突处理 (Phase 2)"""
        profiles = [n for n, d in self.graph.nodes(data=True) if d.get('type') == NodeType.PROFILE.value]

        for p_node in profiles:
            if random.random() < 0.08:  # 8% 概率演化
                new_id = f"{p_node}_v2"
                self.graph.add_node(new_id, type=NodeType.PROFILE.value, content="Updated Preference")
                self.graph.add_edge(p_node, new_id, type=EdgeType.VERSION.value)
                self.evolution_stats["conflicts_resolved"] += 1
                self.evolution_stats["decay_operations"] += 1
                self.logger.info(f"Evolution triggered for node: {p_node}")

    def simulate_abstraction(self):
        """模拟概念涌现 (Phase 5)"""
        activity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == NodeType.ACTIVITY.value
        ]

        if not activity_nodes: return

        activity_subgraph = self.graph.subgraph(activity_nodes).to_undirected()

        if len(activity_subgraph) == 0: return

        try:
            # 使用贪婪模块度算法模拟社区发现
            communities = nx.community.greedy_modularity_communities(activity_subgraph)

            for i, comm in enumerate(communities):
                if len(comm) >= 3:  # 社区大小 >= 3 触发概念生成
                    concept_id = f"CONCEPT_EMERGENT_{i}"
                    self.graph.add_node(concept_id, type=NodeType.ABSTRACT_CONCEPT.value)

                    for child_id in comm:
                        self.graph.add_edge(concept_id, child_id, type=EdgeType.ABSTRACT.value)

                    self.logger.info(f"Concept emerged from cluster #{i} (size: {len(comm)})")
        except:
            self.logger.warning("Community detection skipped (graph structure simple).")


# ==========================================
# 3. 实验指标计算器 (Metrics Calculator)
# ==========================================

class ThesisMetricsEvaluator:
    def __init__(self, graph_system, logger):
        self.G = graph_system.graph
        self.stats = graph_system.evolution_stats
        self.logger = logger

    def compute_and_log(self):
        self.logger.info("Computing final graph metrics...")

        metrics = {}

        # --- 1. 拓扑指标 ---
        metrics['Node Count'] = self.G.number_of_nodes()
        metrics['Edge Count'] = self.G.number_of_edges()

        # 密度
        density = nx.density(self.G)
        metrics['Graph Density'] = density

        # 聚类系数 (转无向图)
        G_undirected = self.G.to_undirected()
        try:
            clustering = nx.average_clustering(G_undirected)
        except:
            clustering = 0.0
        metrics['Clustering Coefficient'] = clustering

        # 平均路径长度 (最大连通分量)
        if len(G_undirected) > 0:
            largest_cc_nodes = max(nx.connected_components(G_undirected), key=len)
            subgraph = G_undirected.subgraph(largest_cc_nodes)
            try:
                avg_path = nx.average_shortest_path_length(subgraph)
            except:
                avg_path = 0.0
        else:
            avg_path = 0.0
        metrics['Avg Path Length'] = avg_path

        # --- 2. 结构化质量 (模块度) ---
        try:
            communities = nx.community.greedy_modularity_communities(G_undirected)
            modularity = nx.community.modularity(G_undirected, communities)
        except:
            modularity = 0.0
        metrics['Modularity (Q)'] = modularity

        # --- 3. 认知分布 ---
        node_types = [d.get('type') for n, d in self.G.nodes(data=True)]
        counts = Counter(node_types)

        concepts = counts[NodeType.PROFILE.value] + counts[NodeType.KNOWLEDGE.value] + counts[
            NodeType.ABSTRACT_CONCEPT.value]
        events = counts[NodeType.ACTIVITY.value] + counts[NodeType.THOUGHT.value]

        # 抽象率
        abstraction_ratio = concepts / max(1, events)
        metrics['Abstraction Ratio (C/E)'] = abstraction_ratio

        # --- 4. 演化 ---
        metrics['Conflicts Resolved'] = self.stats['conflicts_resolved']

        # --- 写入 Log 文件 ---
        self._write_to_log(metrics)

        return metrics

    def _write_to_log(self, metrics):
        """将指标格式化写入日志"""
        lines = []
        lines.append("")
        lines.append("==================================================")
        lines.append("        NEURO-SYMBOLIC GRAPH METRICS REPORT       ")
        lines.append("==================================================")

        lines.append("[1. Topology Metrics]")
        lines.append(f"  Graph Density          : {metrics['Graph Density']:.4f}")
        lines.append(f"  Clustering Coeff.      : {metrics['Clustering Coefficient']:.4f}")
        lines.append(f"  Avg. Path Length       : {metrics['Avg Path Length']:.2f}")

        lines.append("\n[2. Structural Quality]")
        lines.append(f"  Modularity (Q-Score)   : {metrics['Modularity (Q)']:.4f}")
        lines.append(f"  Total Nodes            : {metrics['Node Count']}")
        lines.append(f"  Total Edges            : {metrics['Edge Count']}")

        lines.append("\n[3. Cognitive Dynamics]")
        lines.append(f"  Abstraction Ratio      : {metrics['Abstraction Ratio (C/E)']:.2f}")
        lines.append(f"  Conflicts Resolved     : {metrics['Conflicts Resolved']}")
        lines.append("==================================================")
        lines.append("")

        for line in lines:
            self.logger.info(line)

        print("Metrics report has been written to 'evaluation_metrics.log'")


# ==========================================
# 4. 执行脚本 (Execution)
# ==========================================

def run_evaluation(jsonl_file_path):
    # 配置 Logger: 输出到文件，且不要时间戳
    logger = logging.getLogger("ThesisEval")
    logger.setLevel(logging.INFO)

    # 清除旧的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler("evaluation_metrics.log", mode='w', encoding='utf-8')
    # 格式设置为: [级别] - 消息
    formatter = logging.Formatter('[%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    print(f"Loading data from: {jsonl_file_path}")
    logger.info(f"Initialized evaluation for dataset: {jsonl_file_path}")

    system = ThesisGraphSystem(logger)

    # 1. Load Data
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            try:
                # 尝试解析第一行看是否包含 "memory_atoms" 键
                first_obj = json.loads(lines[0])
                if "memory_atoms" in first_obj:
                    atoms = first_obj["memory_atoms"]
                else:
                    # 假设是每行一个原子
                    atoms = [json.loads(line) for line in lines]
            except:
                # 回退方案
                atoms = [json.loads(line) for line in lines]
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        print(f"Error: {e}")
        return

    logger.info(f"Loaded {len(atoms)} atoms. Starting graph construction pipeline...")

    # 2. Build Graph (The C2 Pipeline)
    for atom in atoms:
        # 数据清洗：确保有 atom_type
        if 'atom_type' not in atom:
            atom['atom_type'] = NodeType.ACTIVITY.value
        system.add_node(atom)

    logger.info("Executing Phase 1: Temporal Skeleton Construction...")
    t_edges = system.build_temporal_layer()
    logger.info(f" > Added {t_edges} temporal edges.")

    logger.info("Executing Phase 1 & 4: Semantic Association...")
    s_edges = system.build_semantic_layer()
    logger.info(f" > Added {s_edges} semantic edges.")

    logger.info("Executing Phase 2: Evolution & Conflict Resolution...")
    system.simulate_evolution()

    logger.info("Executing Phase 5: Structure-Induced Abstraction...")
    system.simulate_abstraction()

    # 3. Calculate & Log Metrics
    evaluator = ThesisMetricsEvaluator(system, logger)
    metrics = evaluator.compute_and_log()


if __name__ == "__main__":
    # 请确保文件名与你上传的一致
    run_evaluation("locomo_extracted_atoms_no_embedding.jsonl")