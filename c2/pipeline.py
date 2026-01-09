# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：pipeline.py
@Author  ：niu
@Date    ：2026/1/8 13:26 
@Desc    ：
"""

# c2/pipeline.py
import sys
import os
from typing import List

# 路径修复 (Standard Project Setup)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入配置
from c2.config import GRAPH_SAVE_PATH

# 导入核心模块
try:
    from general.decoupled_memory import DecoupledMemoryAtom
except ImportError:
    # Mock Class for standalone testing
    from dataclasses import dataclass


    @dataclass
    class DecoupledMemoryAtom:
        content: str
        atom_type: str = "event"
        id: str = "0"
        timestamp: str = "2023-01-01 10:00:00"

from c2.graph_storage import AtomGraph
from c2.builders.semantic import SemanticBuilder
from c2.builders.temporal import TemporalBuilder
from c2.builders.evolution import EvolutionBuilder
from c2.builders.structural import StructuralBuilder
from c2.definitions import GraphNode


class NeuroSymbolicPipeline:
    def __init__(self):
        # 1. 初始化图存储
        self.graph = AtomGraph()

        # 2. 尝试加载旧存档 (增量更新的关键)
        self.graph.load(GRAPH_SAVE_PATH)

        # 3. 初始化构建器
        self.semantic = SemanticBuilder()
        self.temporal = TemporalBuilder()
        self.evolution = EvolutionBuilder()
        self.structural = StructuralBuilder()

    def run(self, atoms: List[DecoupledMemoryAtom]):
        """
        运行 Pipeline 的主入口
        """
        print(f"\n=== Pipeline Start: Input {len(atoms)} Atoms ===")

        # Step 1: 转换原子 (MemoryAtom -> GraphNode)
        # 只处理图中不存在的新节点
        new_nodes = []
        for atom in atoms:
            # 如果这id已经处理过了，就跳过 (去重)
            if self.graph.get_node(str(atom.id)):
                continue

            node = GraphNode(
                id=str(atom.id),
                content=atom.content,
                timestamp=atom.timestamp,
                type=atom.atom_type
            )
            self.graph.add_node(node)
            new_nodes.append(node)

        if not new_nodes:
            print("⚠️ No new unique nodes to process.")
            return

        print(f"🔄 Processing {len(new_nodes)} new unique nodes...")

        # Step 2: 语义构建 (Semantic) - 提取实体 & Embedding
        self.semantic.process(new_nodes, self.graph)

        # Step 3: 时序构建 (Temporal) - 连接时间线
        self.temporal.process(new_nodes, self.graph)

        # Step 4: 演化构建 (Evolution) - 冲突检测 & 版本控制
        self.evolution.process(new_nodes, self.graph)

        # Step 5: 结构构建 (Structural) - GNN 自监督训练 & 推理
        self.structural.process(new_nodes, self.graph)

        # Step 6: 自动保存 (Auto-Save)
        self.graph.save(GRAPH_SAVE_PATH)

        self._print_stats()

    def _print_stats(self):
        nodes = self.graph.get_all_nodes()
        edge_count = sum(len(n.edges) for n in nodes)
        print(f"\n=== Pipeline End: Total Nodes={len(nodes)}, Total Edges={edge_count} ===")


# ==========================================
# 测试入口 (Mock Data)
# ==========================================
if __name__ == "__main__":
    # 模拟数据
    atoms = [
        DecoupledMemoryAtom(id="A01", content="Andy 也就是我，非常喜欢户外徒步运动。", timestamp="2023-10-01 09:00"),
        DecoupledMemoryAtom(id="A02", content="Andy 周末去了 Fox Hollow 公园。", timestamp="2023-10-02 14:00"),
        DecoupledMemoryAtom(id="A03", content="Andy 现在非常讨厌徒步，再也不去了。", timestamp="2023-10-05 10:00"),
        DecoupledMemoryAtom(id="A04", content="买了一双昂贵的专业登山靴。", timestamp="2023-10-06 11:00"),
    ]

    pipeline = NeuroSymbolicPipeline()
    pipeline.run(atoms)


# -*- coding: utf-8 -*-
"""
================================================================================
📝 TODO LIST: Chapter 2 神经符号协同演化系统 - 挂起状态
================================================================================
📅 日期: 2026-01-07
🚩 当前进度: 
   - [x] 框架 (Pipeline) 已跑通。
   - [x] 语义侧 (Semantic): 已接入 ZhipuAI + 本地 MiniLM 模型。
   - [x] 演化侧 (Evolution): 已接入 ZhipuAI 进行真实 NLI 冲突检测。
   - [ ] 符号侧 (Structural): 目前仍为 Mock 版本 (z=x)，尚未应用 GNN 训练逻辑。

--------------------------------------------------------------------------------
🚀 下次启动时的任务清单 (按顺序执行):

[1] 🛠️ 环境依赖 (Environment)
    - [ ] 安装 PyTorch Geometric (PyG)。
          这是下一版 GNN 代码运行的基础。
          命令: pip install torch-geometric

[2] 💻 代码升级 (Code Update)
    - [ ] 修改 c2/builders/structural.py。
          将当前的 Mock 逻辑替换为【自监督训练版】代码 (包含 NeuroSymbolicGNN 类和 _train_gnn 循环)。
          (代码见聊天记录 "上策：正统流")

[3] 🧪 验证与调优 (Verify)
    - [ ] 运行 pipeline.py。
          观察控制台是否出现 "Training GNN for 50 epochs..." 日志。
    - [ ] 检查 "Final Loss" 是否收敛。
    - [ ] 观察新的 A04 <-> A01 隐式连接是否被正确召回。

[4] 🔮 未来优化 (Future)
    - [ ] 引入 Vector Database (Chroma/FAISS) 替换列表遍历。
    - [ ] 实现图谱的保存与加载 (Persistance)。
================================================================================
"""

