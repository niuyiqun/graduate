import json
import logging
import os
import sys

# 确保能找到项目根目录
sys.path.append(os.getcwd())

from c2.definitions import MemoryNode, AtomCategory, NodeType
from c2.graph_storage import MemoryGraph
from c2.builders.temporal import TemporalBuilder, BasicSemanticBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryGraphPipeline:
    def __init__(self, c1_output_path: str, graph_save_path: str):
        self.c1_path = c1_output_path
        self.save_path = graph_save_path
        self.graph = MemoryGraph()

    def load_atoms(self):
        """
        Step 0: 从 C1 加载原子并转化为 MemoryNode 对象
        """
        if not os.path.exists(self.c1_path):
            raise FileNotFoundError(f"C1 output not found at {self.c1_path}")

        logger.info(f"Loading atoms from {self.c1_path}...")

        loaded_count = 0
        with open(self.c1_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # C1 的输出字段映射
                # 假设 C1 输出包含: category, content, metadata
                cat_str = data.get('category')
                if not cat_str: continue

                try:
                    category = AtomCategory(cat_str)
                except ValueError:
                    # 如果 C1 输出的类别名和 C2 定义不一致，这里做兼容处理
                    # 比如 C1 输出了简写，这里可以加映射逻辑
                    logger.warning(f"Unknown category: {cat_str}, skipping.")
                    continue

                node_type = MemoryNode.map_category_to_type(cat_str)

                # 构造唯一 ID
                # 优先使用 C1 提供的 ID，否则生成一个
                node_id = data.get('id')
                if not node_id:
                    node_id = f"{cat_str}_{loaded_count}_{data.get('timestamp', 0)}"

                node = MemoryNode(
                    node_id=str(node_id),
                    content=data.get('content', ''),
                    category=category,
                    node_type=node_type,
                    timestamp=data.get('timestamp', 0),
                    meta=data.get('metadata', {})
                )

                self.graph.add_node(node)
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} atoms into the graph.")

    def run_phase1_skeleton(self):
        """
        Phase 1: 基础骨架构建 (Skeleton Construction)
        """
        logger.info(">>> Starting Phase 1: Skeleton Construction <<<")

        # 1. 知行合一：连接 Thought 和 Activity
        sem_builder = BasicSemanticBuilder(self.graph)
        sem_builder.build()

        # 2. 时序连接：串联 Activity
        temp_builder = TemporalBuilder(self.graph)
        temp_builder.build()

        logger.info("Phase 1 Complete.")

    def run_full_pipeline(self):
        self.load_atoms()

        if self.graph.graph.number_of_nodes() == 0:
            logger.error("Graph is empty. Check C1 output.")
            return

        # Phase 1
        self.run_phase1_skeleton()

        # Phase 2: Evolution (TODO)
        # Phase 3: GNN (TODO)

        # Save
        self.graph.save_graph(self.save_path)


if __name__ == "__main__":
    # 配置路径
    # 假设我们在项目根目录下运行
    C1_OUTPUT = "c1/output/locomo_extracted_atoms.jsonl"
    C2_GRAPH_OUTPUT = "c2/memory_graph.json"

    pipeline = MemoryGraphPipeline(C1_OUTPUT, C2_GRAPH_OUTPUT)
    pipeline.run_full_pipeline()