# -*- coding: UTF-8 -*-
# c2/pipeline.py

import json
import logging
import os
import sys
from datetime import datetime  # 🔥 [新增]

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
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MemoryGraphPipeline:
    def __init__(self, c1_output_path: str, output_path: str, config_path: str):
        self.c1_path = c1_output_path
        self.output_path = output_path

        print("\n" + "=" * 50)
        print("🚀 [Pipeline] 初始化 QwenChat (Local vLLM)...")
        try:
            self.llm = QwenChat(config_path=config_path)
        except Exception as e:
            logger.warning(f"❌ LLM init failed: {e}")
            self.llm = None

        self.semantic_builder = BasicSemanticBuilder(self.llm)
        print("✅ [Pipeline] 初始化完成")
        print("=" * 50 + "\n")

    def _parse_timestamp(self, ts_val) -> float:
        """🔥 [新增] 鲁棒的时间戳解析函数"""
        if ts_val is None:
            return 0.0
        if isinstance(ts_val, (int, float)):
            return float(ts_val)
        if isinstance(ts_val, str):
            try:
                # 尝试解析标准格式 "2023-05-08 13:56:00"
                dt = datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
            except ValueError:
                # 如果格式不对，尝试其他格式或直接返回 0
                return 0.0
        return 0.0

    def process_single_sample(self, sample_data: dict) -> dict:
        source_id = sample_data.get("source_id", "unknown")

        # 1. 创建图
        graph = MemoryGraph()

        # 2. 加载原子
        atoms_list = sample_data.get("memory_atoms", [])
        if not atoms_list and isinstance(sample_data, list):
            atoms_list = sample_data

        for idx, atom_data in enumerate(atoms_list):
            cat_str = atom_data.get('atom_type', 'unknown')
            try:
                category = AtomCategory(cat_str)
            except ValueError:
                category = AtomCategory.UNKNOWN

            # 🔥 [修正] 使用解析后的 float 时间戳
            ts_float = self._parse_timestamp(atom_data.get('timestamp'))

            node = MemoryNode(
                node_id=atom_data.get('id', f"node_{idx}"),
                content=atom_data.get('content', ''),
                category=category,
                node_type=MemoryNode.map_category_to_type(cat_str),
                timestamp=ts_float,  # 这里传入 float
                embedding=atom_data.get('embedding'),
                meta=atom_data
            )
            graph.add_node(node)

        nodes = graph.get_all_nodes()
        if not nodes: return None

        print(f"\n🔷 Processing Sample: {source_id} | Atoms: {len(nodes)}")

        # === 3. 执行 Phase ===

        # Phase 1
        self.semantic_builder.process(nodes, graph)
        try:
            TemporalBuilder().process(nodes, graph)
        except Exception as e:
            print(f"❌ [Temporal] Error: {e}")

        # Phase 2
        try:
            EvolutionBuilder(self.llm).process(nodes, graph)
        except Exception as e:
            print(f"❌ [Evolution] Error: {e}")

        # Phase 3 & 4
        try:
            StructuralBuilder(self.llm).process(nodes, graph)
        except Exception as e:
            print(f"❌ [Structural] Error: {e}")

        # Phase 5
        try:
            EmergenceBuilder(self.llm).process(nodes, graph)
        except Exception as e:
            print(f"❌ [Emergence] Error: {e}")

        # 4. 统计结果
        n_count = graph.graph.number_of_nodes()
        e_count = graph.graph.number_of_edges()
        print(f"✅ [Done] Stats: Nodes={n_count}, Edges={e_count}")

        # 序列化
        graph_data = graph.get_nx_graph()
        import networkx.readwrite.json_graph as json_graph
        json_data = json.loads(json.dumps(json_graph.node_link_data(graph_data)))

        return {
            "source_id": source_id,
            "graph_data": json_data
        }

    def run(self):
        if not os.path.exists(self.c1_path):
            print(f"❌ Input file not found: {self.c1_path}")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        processed_count = 0
        with open(self.c1_path, 'r', encoding='utf-8') as fin, \
                open(self.output_path, 'w', encoding='utf-8') as fout:

            for line in fin:
                if not line.strip(): continue
                try:
                    sample = json.loads(line)
                    result = self.process_single_sample(sample)

                    if result:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        processed_count += 1

                except Exception as e:
                    print(f"❌ Critical Error processing line: {e}")

        print(f"\n🎉 全部完成！共生成 {processed_count} 张图谱。")
        print(f"结果已保存至: {self.output_path}")


if __name__ == "__main__":
    CONFIG_PATH = "config/llm_config.yaml"
    # 🔥 确保这里指向你打过 Embedding 补丁的文件
    C1_OUTPUT = "c1/output/locomo_extracted_atoms_with_emb.jsonl"
    C2_OUTPUT = "c2/output/memory_graphs.jsonl"

    pipeline = MemoryGraphPipeline(C1_OUTPUT, C2_OUTPUT, CONFIG_PATH)
    pipeline.run()