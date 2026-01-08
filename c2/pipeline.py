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
from typing import List

# 引入 Chapter 1 定义
sys.path.append("..")
try:
    from general.decoupled_memory import DecoupledMemoryAtom
except ImportError:
    from .definitions import DecoupledMemoryAtom  # Fallback

from c2.graph_storage import AtomGraph
from c2.builders.semantic import SemanticBuilder
from c2.builders.temporal import TemporalBuilder
from c2.builders.evolution import EvolutionBuilder
from c2.builders.structural import StructuralBuilder


class NeuroSymbolicPipeline:
    """
    [Chapter 2 System] 神经符号协同演化系统
    """

    def __init__(self):
        self.graph = AtomGraph()
        self.builders = [
            SemanticBuilder(),  # 1. 神经侧语义 (LLM)
            TemporalBuilder(),  # 2. 规则侧时序
            EvolutionBuilder(),  # 3. 演化侧冲突
            StructuralBuilder()  # 4. 符号侧结构 (GNN)
        ]

    def run(self, new_atoms: List[DecoupledMemoryAtom]):
        print(f"\n=== Pipeline Start: 处理 {len(new_atoms)} 个原子 ===")

        # 1. 接入标准化
        new_nodes = [self.graph.add_atom(atom) for atom in new_atoms]

        # 2. 依次执行构建
        for builder in self.builders:
            builder.process(new_nodes, self.graph)

        self._stats()

    def _stats(self):
        nodes = self.graph.get_all_nodes()
        edges = sum(len(n.edges) for n in nodes)
        print(f"=== Pipeline End: Nodes={len(nodes)}, Edges={edges} ===")


if __name__ == "__main__":
    # 0. 准备环境
    # 如果 DecoupledMemoryAtom 是 Mock 的，确保它有必要的属性
    try:
        from general.decoupled_memory import DecoupledMemoryAtom
    except ImportError:
        # 本地测试用的 Mock 类
        from dataclasses import dataclass


        @dataclass
        class DecoupledMemoryAtom:
            content: str
            atom_type: str = "event"
            id: str = "0"
            timestamp: str = "2023-01-01 10:00:00"

    print("\n" + "=" * 50)
    print("🧪 启动 Neuro-Symbolic Pipeline 集成测试")
    print("=" * 50)

    # 1. 构造剧本数据 (Scenario Data)
    # 我们设计 4 个原子，旨在触发所有类型的边
    atoms = [
        # Atom A: 基础事件
        DecoupledMemoryAtom(
            id="A01",
            content="Andy 也就是我，非常喜欢户外徒步运动。",
            atom_type="profile",
            timestamp="2023-10-01 09:00:00"
        ),

        # Atom B: 应该与 A01 产生 [SEMANTIC] 关联 (共享实体: Andy, 徒步)
        # 且应该与 A01 产生 [TEMPORAL] 关联 (时间晚 5 分钟)
        DecoupledMemoryAtom(
            id="A02",
            content="Andy 周末去了 Fox Hollow 国家公园。",
            atom_type="event",
            timestamp="2023-10-01 09:05:00"
        ),

        # Atom C: 这是一个冲突信息，应该触发 [VERSION] 演化
        # (原本喜欢徒步，现在说讨厌，模拟冲突)
        DecoupledMemoryAtom(
            id="A03",
            content="Andy 现在非常讨厌徒步，再也不去了。",
            atom_type="update",
            timestamp="2023-10-02 10:00:00"
        ),

        # Atom D: 隐式关联，应该由 GNN 触发 [IMPLICIT]
        # (虽然没提 Andy，但买了登山靴，逻辑上与徒步相关)
        DecoupledMemoryAtom(
            id="A04",
            content="买了一双昂贵的专业登山靴。",
            atom_type="event",
            timestamp="2023-10-03 10:00:00"
        )
    ]

    # 2. 初始化流水线
    pipeline = NeuroSymbolicPipeline()

    # 3. 运行处理
    pipeline.run(atoms)

    # 4. 结果可视化验证
    print("\n📊 图谱构建结果分析:")
    all_nodes = pipeline.graph.get_all_nodes()

    for node in all_nodes:
        print(f"\n📍 节点 [{node.id}] (Act:{node.activation:.1f}): {node.content[:20]}...")
        if not node.edges:
            print("   (孤立节点)")
        for edge in node.edges:
            # 打印边的类型和目标
            target_node = pipeline.graph.nodes.get(edge.target)
            target_content = target_node.content[:10] if target_node else "Unknown"

            icon = "🔗"
            if edge.type.value == "SEMANTIC":
                icon = "🧠 [显式语义]"
            elif edge.type.value == "TEMPORAL":
                icon = "⏱️ [时序流]"
            elif edge.type.value == "EVOLVES_TO":
                icon = "🔄 [版本演化]"
            elif edge.type.value == "IMPLICIT":
                icon = "🤖 [神经推理]"

            print(f"   |-- {icon} --> [{edge.target}] {target_content}...")

    print("\n" + "=" * 50)
    print("✅ 测试完成！请检查日志中是否包含了所有四种类型的边。")
    print("=" * 50)


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

