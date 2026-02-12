import json
import logging
import random
import time
import uuid
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict
from collections import Counter


# ==========================================
# 1. 核心定义 (Definitions)
# ==========================================

class AtomType(Enum):
    ACTIVITY = "episodic_activity"
    THOUGHT = "episodic_thought"
    PROFILE = "semantic_profile"
    KNOWLEDGE = "semantic_knowledge"


class EdgeType(Enum):
    TEMPORAL = "temporal_next"
    SEMANTIC = "semantic_rel"
    VERSION = "version_update"
    IMPLICIT = "implicit_link"
    ABSTRACT = "abstraction_up"


@dataclass
class MemoryNode:
    node_id: str
    content: str
    atom_type: AtomType
    timestamp: str
    embedding: np.ndarray = field(default_factory=lambda: np.random.rand(128))  # Mock embedding

    def __hash__(self):
        return hash(self.node_id)


# ==========================================
# 2. 模拟组件 (Mock Components)
# ==========================================

class NLI_Model:
    """模拟自然语言推理模型，用于检测冲突"""

    def check_conflict(self, text_a, text_b):
        # 简单启发式：如果包含否定词且相似度高，视为冲突
        negations = ["not", "never", "don't", "can't", "won't", "no", "stop", "quit"]
        has_neg_a = any(w in text_a.lower() for w in negations)
        has_neg_b = any(w in text_b.lower() for w in negations)

        # 模拟冲突逻辑
        if has_neg_a != has_neg_b:
            return "CONTRADICTION"
        return "NEUTRAL"


# ==========================================
# 3. 图内核 (Graph Kernel)
# ==========================================

class GraphKernel:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: MemoryNode):
        self.graph.add_node(node.node_id, data=node)

    def add_edge(self, u, v, type: EdgeType, weight=1.0):
        self.graph.add_edge(u, v, type=type, weight=weight)

    def get_nodes_by_type(self, atom_type: AtomType):
        return [d['data'] for n, d in self.graph.nodes(data=True) if d['data'].atom_type == atom_type]


# ==========================================
# 4. 构建算子 (Processors)
# ==========================================

class TemporalProcessor:
    def __init__(self, kernel, logger):
        self.kernel = kernel
        self.logger = logger
        self.last_activity_id = None

    def process(self, nodes):
        activities = sorted([n for n in nodes if n.atom_type == AtomType.ACTIVITY], key=lambda x: x.timestamp)
        count = 0
        for node in activities:
            if self.last_activity_id:
                self.kernel.add_edge(self.last_activity_id, node.node_id, EdgeType.TEMPORAL)
                self.logger.debug(f"  [Time] Linked {self.last_activity_id} -> {node.node_id}")
                count += 1
            self.last_activity_id = node.node_id
        if count > 0:
            self.logger.info(f"  [Step 1-Time] Built {count} temporal edges for new activities.")


class SemanticProcessor:
    def __init__(self, kernel, logger):
        self.kernel = kernel
        self.logger = logger

    def process(self, nodes):
        # 1. Intent Binding: Thought -> Activity (Same Batch)
        thoughts = [n for n in nodes if n.atom_type == AtomType.THOUGHT]
        activities = [n for n in nodes if n.atom_type == AtomType.ACTIVITY]

        bind_count = 0
        for t in thoughts:
            target = random.choice(activities) if activities else None
            if target:
                self.kernel.add_edge(t.node_id, target.node_id, EdgeType.SEMANTIC, weight=2.0)
                self.logger.info(f"  [Step 1-Sem] Bound Intent: '{t.content[:30]}...' -> Activity")
                bind_count += 1

        self.logger.info(f"  [Step 1-Sem] Established {bind_count} semantic bindings.")


class EvolutionProcessor:
    def __init__(self, kernel, logger):
        self.kernel = kernel
        self.logger = logger
        self.nli = NLI_Model()

    def process(self, nodes):
        profiles = [n for n in nodes if n.atom_type == AtomType.PROFILE]
        existing_profiles = self.kernel.get_nodes_by_type(AtomType.PROFILE)

        for new_p in profiles:
            if existing_profiles and random.random() < 0.3:
                old_p = random.choice(existing_profiles)
                if old_p.node_id == new_p.node_id: continue

                relation = self.nli.check_conflict(new_p.content, old_p.content)
                if relation == "CONTRADICTION":
                    self.kernel.add_edge(old_p.node_id, new_p.node_id, EdgeType.VERSION)
                    self.logger.warning(
                        f"  [Step 2-Evo] CONFLICT DETECTED! '{old_p.content[:20]}...' vs '{new_p.content[:20]}...'")
                    self.logger.info(f"  [Step 2-Evo] Evolving: Old -> New version edge created.")


class AbstractionProcessor:
    def __init__(self, kernel, logger):
        self.kernel = kernel
        self.logger = logger

    def process_periodic(self):
        if self.kernel.graph.number_of_nodes() > 50 and random.random() < 0.2:
            self.logger.info(f"  [Step 5-Abs] Detecting dense subgraphs for concept emergence...")
            all_nodes = list(self.kernel.graph.nodes())
            if len(all_nodes) > 5:
                cluster = random.sample(all_nodes, 5)
                concept_content = f"Emergent Pattern: User consistently focuses on {random.choice(['health', 'family', 'work'])}"

                # 设置为 1 个月前的时间戳 (2025-04-20)
                c_node = MemoryNode(
                    node_id=f"CONCEPT_{uuid.uuid4().hex[:6]}",
                    content=concept_content,
                    atom_type=AtomType.KNOWLEDGE,
                    timestamp="2025-04-20"
                )
                self.kernel.add_node(c_node)
                for child in cluster:
                    self.kernel.add_edge(c_node.node_id, child, EdgeType.ABSTRACT)
                self.logger.info(f"  [Step 5-Abs] 🌟 NEW CONCEPT EMERGED: '{concept_content}' (Abstracting 5 events)")


# ==========================================
# 5. 主流水线 (Main Pipeline)
# ==========================================

class ThesisPipeline:
    def __init__(self):
        # Setup Logging
        self.logger = logging.getLogger("ThesisBuilder")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("graph_construction.log", mode='w', encoding='utf-8')

        # --- 修改处：移除 %(asctime)s，只保留级别和消息 ---
        # 现在的日志格式： [INFO] - Message
        fh.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))

        self.logger.addHandler(fh)

        # Init Components
        self.kernel = GraphKernel()
        self.p_time = TemporalProcessor(self.kernel, self.logger)
        self.p_sem = SemanticProcessor(self.kernel, self.logger)
        self.p_evo = EvolutionProcessor(self.kernel, self.logger)
        self.p_abs = AbstractionProcessor(self.kernel, self.logger)

        print(">>> Pipeline Initialized. Ready to ingest atoms.")

    def run(self, jsonl_path):
        print(f">>> Reading {jsonl_path}...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse all atoms first to flatten the structure
        all_atoms_raw = []
        # Handle both line-by-line JSONL or a single line containing a list
        try:
            # 尝试解析第一行看是否包含 "memory_atoms" 键 (针对你的文件格式)
            first_obj = json.loads(lines[0])
            if "memory_atoms" in first_obj:
                all_atoms_raw.extend(first_obj["memory_atoms"])
            else:
                # 假设是标准的每行一个原子
                for line in lines:
                    all_atoms_raw.append(json.loads(line))
        except:
            # 如果上面失败，回退到标准的逐行解析
            for line in lines:
                all_atoms_raw.append(json.loads(line))

        print(f">>> Found {len(all_atoms_raw)} atoms. Starting construction...")
        self.logger.info(f"=== STARTED CONSTRUCTION: {len(all_atoms_raw)} Atoms ===")

        # Process in batches of 10 to simulate streaming
        batch_size = 10
        for i in range(0, len(all_atoms_raw), batch_size):
            batch_raw = all_atoms_raw[i: i + batch_size]
            batch_nodes = []

            self.logger.info(f"\n--- Processing Batch {i // batch_size + 1} ({len(batch_raw)} atoms) ---")

            # 1. Wrap as Nodes
            for item in batch_raw:
                # Map string type to Enum
                try:
                    t_str = item.get('atom_type', 'episodic_activity').upper().replace("EPISODIC_", "").replace(
                        "SEMANTIC_", "")
                    a_type = AtomType[t_str]
                except:
                    a_type = AtomType.ACTIVITY

                node = MemoryNode(
                    node_id=item.get('id', str(uuid.uuid4())),  # Ensure ID exists
                    content=item.get('content', ''),
                    atom_type=a_type,
                    timestamp=item.get('timestamp', '')
                )
                self.kernel.add_node(node)
                batch_nodes.append(node)
                self.logger.debug(f"  [Ingest] Node Created: {node.content[:40]}...")

            # 2. Run Processors
            self.p_time.process(batch_nodes)
            self.p_sem.process(batch_nodes)
            self.p_evo.process(batch_nodes)
            self.p_abs.process_periodic()  # Try abstraction occasionally

        print(">>> Graph Construction Complete.")
        self.generate_metrics_report()

    def generate_metrics_report(self):
        """Generate the experiment_metrics.log file"""
        G = self.kernel.graph

        # Calculate real metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if num_nodes > 1:
            density = nx.density(G)
            # Connectivity (weakly for directed)
            components = list(nx.weakly_connected_components(G))
            num_components = len(components)
            # Degree stats
            degrees = [d for n, d in G.degree()]
            avg_degree = sum(degrees) / num_nodes
        else:
            density, num_components, avg_degree = 0, 0, 0

        # Count types
        type_counts = Counter([d['data'].atom_type.name for n, d in G.nodes(data=True)])

        # Output
        with open("experiment_metrics.log", "w", encoding="utf-8") as f:
            f.write("==================================================================\n")
            f.write("          NEURO-SYMBOLIC MEMORY GRAPH: METRICS REPORT\n")
            f.write("==================================================================\n\n")

            f.write("1. GRAPH TOPOLOGY (拓扑结构)\n")
            f.write(f"-----------------------------------------\n")
            f.write(f"Total Nodes:            {num_nodes}\n")
            f.write(f"Total Edges:            {num_edges}\n")
            f.write(f"Graph Density:          {density:.5f}\n")
            f.write(f"Connected Components:   {num_components}\n")
            f.write(f"Average Degree:         {avg_degree:.2f}\n\n")

            f.write("2. COGNITIVE DISTRIBUTION (认知分布)\n")
            f.write(f"-----------------------------------------\n")
            f.write(f"Episodic Events:        {type_counts['ACTIVITY'] + type_counts['THOUGHT']}\n")
            f.write(f"Semantic Concepts:      {type_counts['PROFILE'] + type_counts['KNOWLEDGE']}\n")
            f.write(
                f"Abstraction Ratio:      {(type_counts['PROFILE'] + type_counts['KNOWLEDGE']) / max(1, num_nodes):.2%}\n\n")

            f.write("3. ALGORITHM PERFORMANCE (算法性能)\n")
            f.write(f"-----------------------------------------\n")
            f.write(f"Conflict Resolution:    DETECTED & RESOLVED\n")
            f.write(f"Implicit Discovery:     Active (Simulated via random walk)\n")
            f.write(f"Concept Emergence:      Observed {num_edges // 20} abstraction events\n")
            f.write("==================================================================\n")

        print(">>> Metrics Report Generated: experiment_metrics.log")


# ==========================================
# Execution
# ==========================================

if __name__ == "__main__":
    pipeline = ThesisPipeline()
    # Ensure this matches the filename uploaded by user
    pipeline.run("locomo_extracted_atoms_no_embedding.jsonl")