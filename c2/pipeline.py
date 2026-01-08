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
📝 TODO LIST: Chapter 2 神经符号协同演化系统完善计划
================================================================================
当前状态：框架逻辑已跑通，处于 Mock（模拟）模式。
下一步目标：接入真实组件（LLM, VectorDB, PyG）。

[1] 🧠 LLM 接入 (The Neural Side)
    - [ ] utils/llm_client.py: 
          封装统一的 LLM 客户端 (Ollama/OpenAI)，支持 qwen2.5/deepseek 等模型。
    - [ ] builders/semantic.py -> _llm_extract(): 
          替换 Mock 规则，使用 Prompt 让 LLM 提取文本中的【实体】和【显式关系】。
    - [ ] builders/evolution.py -> _check_conflict():
          替换 Mock 规则，使用 NLI Prompt 让 LLM 判断新旧原子是否【事实冲突】。
    - [ ] builders/structural.py -> _llm_verify():
          替换 Mock 规则，让 LLM 验证 GNN 推荐的两个节点是否存在【逻辑/因果关联】。

[2] 🔢 向量检索 (The Memory Side)
    - [ ] graph_storage.py: 
          集成 Embedding 模型 (如 bge-m3)，在节点入库时生成 vector。
    - [ ] builders/evolution.py -> _mock_vector_search():
          接入 ChromaDB 或 FAISS，实现真实的 Top-K 语义检索，而非遍历列表。

[3] 🕸️ GNN 深度实现 (The Symbolic Side)
    - [ ] builders/structural.py:
          完善 AtomGraph 到 PyG Data (x, edge_index, edge_attr) 的转换逻辑。
    - [ ] builders/structural.py -> SimpleRGCN:
          定义标准的 RGCN/GAT 网络结构，加载预训练权重或设计自监督 Loss 进行微调。

[4] ⚙️ 工程化配置
    - [ ] c2/config.py: 
          提取硬编码的阈值 (冲突阈值 0.7, 推荐阈值 0.85) 到配置文件。
    - [ ] 日志系统: 
          将 print() 替换为 logger.info/debug，方便记录实验数据。
================================================================================
"""

