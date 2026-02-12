"""
c2.pipeline - 记忆图谱构建总控
"""
import logging
from typing import List
from .graph_kernel import GraphKernel
from .data_models import MemoryNode
from .processors.temporal import TemporalProcessor
from .processors.semantic import SemanticProcessor
from .processors.evolution import EvolutionProcessor
from .processors.inference import TopologyInferenceProcessor
from .processors.abstraction import AbstractionProcessor

# 配置日志
logging.basicConfig(
    filename='memory_construction.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class NeuroSymbolicPipeline:
    def __init__(self):
        self.kernel = GraphKernel()

        # 初始化五个处理阶段 (Phases)
        self.p1_temporal = TemporalProcessor(self.kernel)
        self.p1_semantic = SemanticProcessor(self.kernel)
        self.p2_evolution = EvolutionProcessor(self.kernel)
        self.p3_inference = TopologyInferenceProcessor()  # 传递 kernel 在 call 时
        self.p5_abstraction = AbstractionProcessor(self.kernel)

        self.logger = logging.getLogger("PipelineRoot")

    def ingest_atoms(self, atoms: List[MemoryNode]):
        """
        接收来自 C1 的记忆原子，运行全量构建流程。
        """
        self.logger.info(f"=== Starting Batch Ingestion: {len(atoms)} atoms ===")

        # 0. 节点注册
        for atom in atoms:
            self.kernel.add_node(atom)

        # Phase 1: 基础骨架 (Skeleton Construction)
        self.p1_temporal.process_timeline(atoms)
        self.p1_semantic.process_co_occurrence(atoms)

        # Phase 2: 动态演化 (Dynamic Evolution)
        self.p2_evolution.process_evolution(atoms)

        # Phase 3 & 4: 隐式召回与验证 (Implicit Inference)
        # 注意：这步通常比较耗时，可在全图规模较大时执行
        self.p3_inference.process_implicit_links(self.kernel, atoms)

        # Phase 5: 结构诱导的概念涌现 (Emergence)
        # 通常设定为每 N 个批次或在空闲周期执行
        self.p5_abstraction.run_clustering()

        self.logger.info(f"=== Batch Complete. Graph Stats: {self.kernel.get_stats()} ===")

    def export_graph(self, path):
        self.kernel.save_to_disk(path)