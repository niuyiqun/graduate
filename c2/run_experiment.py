import numpy as np
import time
from c2.data_models import MemoryNode, NodeType
from c2.pipeline import NeuroSymbolicPipeline
from c2.evaluators.graph_metrics import GraphEvaluator


def generate_mock_embedding(dim=768):
    """
    辅助函数：生成符合正态分布的随机向量，用于验证代码流程。
    实际使用时，请接入 C1 的 Embedding 输出。
    """
    return np.random.normal(loc=0.0, scale=0.1, size=dim)


def run_thesis_experiment():
    print(">>> Initializing Chapter 2 Experiment: Neuro-Symbolic Memory Graph...")

    # 1. 初始化流水线
    pipeline = NeuroSymbolicPipeline()

    # 2. 构造实验数据 (模拟 C1 输出的原子流)
    # 场景：用户 Andy 的饮食偏好演变

    # --- Batch 1: 早期记忆 (喜辣) ---
    batch_t1 = [
        MemoryNode(
            node_id="p_001",
            content="Andy 特别喜欢吃辣，尤其是川菜",
            node_type=NodeType.CONCEPT_PROFILE,
            embedding=generate_mock_embedding(),
            timestamp=1600000000
        ),
        MemoryNode(
            node_id="e_001",
            content="Andy 中午去吃了一顿麻辣火锅",
            node_type=NodeType.EVENT_ACTIVITY,
            embedding=generate_mock_embedding(),
            timestamp=1600000100
        ),
        MemoryNode(
            node_id="t_001",
            content="觉得非常解压，心情变好了",
            node_type=NodeType.EVENT_THOUGHT,
            embedding=generate_mock_embedding(),
            timestamp=1600000105
        )
    ]

    # --- Batch 2: 后期记忆 (偏好改变 - 冲突发生) ---
    # 我们故意让 p_002 的向量与 p_001 很近，但内容冲突
    vec_p1 = batch_t1[0].embedding
    vec_p2 = vec_p1 + np.random.normal(0, 0.01, 768)  # 强相关

    batch_t2 = [
        MemoryNode(
            node_id="p_002",
            content="Andy 最近胃不舒服，医生禁止吃辣",
            node_type=NodeType.CONCEPT_PROFILE,
            embedding=vec_p2,
            timestamp=1700000000
        ),
        MemoryNode(
            node_id="e_002",
            content="Andy 拒绝了朋友的火锅邀请，点了一碗粥",
            node_type=NodeType.EVENT_ACTIVITY,
            embedding=generate_mock_embedding(),
            timestamp=1700000100
        )
    ]

    # --- Batch 3: 大量行为 (用于测试聚类涌现) ---
    batch_t3 = []
    base_vec = generate_mock_embedding()
    for i in range(5):
        batch_t3.append(MemoryNode(
            node_id=f"e_sport_{i}",
            content=f"Andy 购买了专业的登山装备 item_{i}",
            node_type=NodeType.EVENT_ACTIVITY,
            embedding=base_vec + np.random.normal(0, 0.05, 768),
            timestamp=1800000000 + i * 100
        ))

    # 3. 注入流水线
    print("\n[Step 1] Ingesting Batch 1 (Baseline)...")
    pipeline.ingest_atoms(batch_t1)

    print("\n[Step 2] Ingesting Batch 2 (Evolution Trigger)...")
    pipeline.ingest_atoms(batch_t2)

    print("\n[Step 3] Ingesting Batch 3 (Emergence Trigger)...")
    pipeline.ingest_atoms(batch_t3)

    # 4. 运行评测
    print("\n[Step 4] Running Evaluation for Thesis Metrics...")
    evaluator = GraphEvaluator(pipeline.kernel)
    metrics = evaluator.evaluate_all()

    print(">>> Experiment Complete. Check log file for details.")


if __name__ == "__main__":
    run_thesis_experiment()