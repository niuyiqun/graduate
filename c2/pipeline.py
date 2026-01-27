# -*- coding: UTF-8 -*-
# c2/pipeline.py

import json
import logging
import os
import sys

# === 路径配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.getcwd())

# === 模块导入 ===
from general.model import QwenChat
from c2.definitions import MemoryNode, AtomCategory, NodeType
from c2.graph_storage import MemoryGraph

# 导入各个构建器
from c2.builders.temporal import TemporalBuilder
from c2.builders.semantic import BasicSemanticBuilder
from c2.builders.structural import StructuralBuilder
from c2.builders.evolution import EvolutionBuilder
from c2.builders.emergence import EmergenceBuilder

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryGraphPipeline:
    def __init__(self, c1_output_path: str, graph_save_path: str, config_path: str):
        self.c1_path = c1_output_path
        self.save_path = graph_save_path
        self.graph = MemoryGraph()

        logger.info("正在初始化 QwenChat (Local vLLM)...")
        try:
            self.llm = QwenChat(config_path=config_path)
        except Exception as e:
            logger.warning(f"LLM init failed: {e}. Some builders may not work.")
            self.llm = None

    def load_atoms(self):
        """Step 0: 正确加载 C1 的层级化数据"""
        if not os.path.exists(self.c1_path):
            logger.error(f"❌ 找不到输入文件: {self.c1_path}")
            return False

        logger.info(f"正在加载记忆原子: {self.c1_path}...")
        total_atoms = 0

        with open(self.c1_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                try:
                    # 1. 解析每一行的 Sample 对象
                    sample_data = json.loads(line)

                    # 2. 提取内部的 atom 列表
                    # 兼容处理：有的文件可能是直接的列表，有的是包含 memory_atoms 键的对象
                    atoms_list = []
                    if isinstance(sample_data, list):
                        atoms_list = sample_data
                    elif isinstance(sample_data, dict):
                        atoms_list = sample_data.get("memory_atoms", [])

                    # 3. 遍历列表创建节点
                    for atom_data in atoms_list:
                        # 字段映射: atom_type (File) -> category (Code)
                        cat_str = atom_data.get('atom_type', 'unknown')
                        try:
                            category = AtomCategory(cat_str)
                        except ValueError:
                            # 尝试兼容处理，比如去掉前缀等，或者默认为 unknown
                            category = AtomCategory.UNKNOWN

                        node = MemoryNode(
                            node_id=atom_data.get('id', f"node_{line_idx}_{total_atoms}"),
                            content=atom_data.get('content', ''),
                            category=category,
                            node_type=MemoryNode.map_category_to_type(cat_str),
                            timestamp=atom_data.get('timestamp', 0),
                            meta=atom_data  # 保留原始数据作为 meta
                        )
                        self.graph.add_node(node)
                        total_atoms += 1

                except Exception as e:
                    logger.warning(f"解析第 {line_idx} 行失败: {e}")

        logger.info(f"✅ 成功加载 {total_atoms} 个原子 (来自 C1 输出)。")
        return total_atoms > 0

    def run(self):
        if not self.load_atoms():
            return

        nodes_batch = self.graph.get_all_nodes()
        if not nodes_batch:
            logger.error("没有加载到任何有效节点，终止运行。")
            return

        # === Phase 1: 基础骨架构建 ===
        logger.info(">>> Phase 1: Skeleton Construction")
        try:
            sem_builder = BasicSemanticBuilder(self.llm)
            sem_builder.process(nodes_batch, self.graph)
        except Exception as e:
            logger.error(f"SemanticBuilder Error: {e}", exc_info=True)

        try:
            temp_builder = TemporalBuilder()
            temp_builder.process(nodes_batch, self.graph)
        except Exception as e:
            logger.error(f"TemporalBuilder Error: {e}")

        # === Phase 2: 演化 ===
        logger.info(">>> Phase 2: Evolution")
        try:
            evo_builder = EvolutionBuilder(self.llm)
            evo_builder.process(nodes_batch, self.graph)
        except Exception as e:
            logger.error(f"EvolutionBuilder Error: {e}")

        # === Phase 3 & 4: 神经符号隐式召回 ===
        logger.info(">>> Phase 3/4: Neuro-Symbolic Recall")
        try:
            struct_builder = StructuralBuilder(self.llm)
            struct_builder.process(nodes_batch, self.graph)
        except Exception as e:
            logger.error(f"StructuralBuilder Error: {e}", exc_info=True)

        # === Phase 5: 概念涌现 ===
        logger.info(">>> Phase 5: Concept Emergence")
        try:
            emerge_builder = EmergenceBuilder(self.llm)
            emerge_builder.process(nodes_batch, self.graph)
        except Exception as e:
            logger.error(f"EmergenceBuilder Error: {e}")

        # 3. 保存
        self.graph.save_graph(self.save_path)
        logger.info(f"🎉 图谱构建完成！已保存至: {self.save_path}")


if __name__ == "__main__":
    CONFIG_PATH = "config/llm_config.yaml"
    C1_OUTPUT = "c1/output/locomo_extracted_atoms_no_embedding.jsonl"
    C2_OUTPUT = "c2/output/memory_graph.json"

    os.makedirs(os.path.dirname(C2_OUTPUT), exist_ok=True)

    pipeline = MemoryGraphPipeline(C1_OUTPUT, C2_OUTPUT, CONFIG_PATH)
    pipeline.run()