import json
import os
import random
import time
import uuid
import numpy as np

# === 配置 ===
OUTPUT_DIR = "../c2/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "memory_graphs_1.jsonl")
SAMPLE_COUNT = 100  # 生成 100 个样本
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的维度

# === 场景模板库 ===
SCENARIOS = {
    "travel": {
        "activities": [
            "Bought flight tickets to {}",
            "Booked a hotel in {}",
            "Visited the {} museum",
            "Tried local street food at {}",
            "Took a train to {}"
        ],
        "thoughts": [
            "Feeling excited about the trip",
            "Worried about the weather",
            "The scenery is breathtaking",
            "I want to stay here forever"
        ],
        "profiles": [
            "{} is an avid traveler",
            "{} prefers solo trips",
            "{} loves cultural heritage"
        ],
        "entities": ["Tokyo", "Paris", "New York", "London", "Kyoto", "Rome"]
    },
    "work": {
        "activities": [
            "Attended a meeting about {}",
            "Finished the {} report",
            "Emailed client regarding {}",
            "Debugged the {} code",
            "Deployed {} to production"
        ],
        "thoughts": [
            "Feeling stressed about the deadline",
            "Satisfied with the team's progress",
            "Need a coffee break",
            "Thinking about a promotion"
        ],
        "profiles": [
            "{} is a workaholic",
            "{} is a software engineer",
            "{} values efficiency"
        ],
        "entities": ["Q3 Project", "API Backend", "UI Design", "Marketing Strategy", "Budget Plan"]
    },
    "diet": {
        "activities": [
            "Ate {} for lunch",
            "Cooked {} for dinner",
            "Bought {} from the supermarket",
            "Ordered {} via delivery",
            "Drank a cup of {}"
        ],
        "thoughts": [
            "It tasted delicious",
            "Feeling guilty about the calories",
            "This is too spicy",
            "I should eat more healthy food"
        ],
        "profiles": [
            "{} loves spicy food",
            "{} is a vegetarian",
            "{} has a sweet tooth"
        ],
        "entities": ["Mapo Tofu", "Salad", "Steak", "Latte", "Sushi", "Pizza"]
    },
    "health": {
        "activities": [
            "Went jogging in the {}",
            "Did a 30-minute {} workout",
            "Took vitamin supplements",
            "Visited the dentist",
            "Slept for {} hours"
        ],
        "thoughts": [
            "Feeling energetic",
            "Muscles are sore",
            "Need to improve sleep quality",
            "Determined to lose weight"
        ],
        "profiles": [
            "{} is fitness conscious",
            "{} is an early bird",
            "{} cares about health"
        ],
        "entities": ["Park", "HIIT", "Yoga", "Gym", "8", "6"]
    }
}


def get_random_embedding():
    # 生成归一化的随机向量，模拟真实 Embedding
    vec = np.random.normal(0, 1, EMBEDDING_DIM)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()


def generate_single_sample(index):
    # 随机选择一个场景
    scenario_key = random.choice(list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_key]
    user_name = random.choice(["Alice", "Bob", "Charlie", "David", "Eve"])

    # 随机生成一些节点
    nodes = []
    base_time = 1700000000.0 + index * 86400  # 每天一个样本

    # 1. 生成 3-5 个 Activity
    act_count = random.randint(3, 5)
    act_ids = []
    for i in range(act_count):
        entity = random.choice(scenario["entities"])
        template = random.choice(scenario["activities"])
        content = template.format(entity)

        node_id = f"node_{index}_act_{i}"
        act_ids.append(node_id)

        nodes.append({
            "id": node_id,
            "content": content,
            "category": "episodic_activity",
            "type": "episodic",
            "timestamp": base_time + i * 3600,
            "embedding": get_random_embedding(),  # 仿真向量
            "meta": {"entities": [entity]}
        })

    # 2. 生成 1-2 个 Thought
    thought_count = random.randint(1, 2)
    thought_ids = []
    for i in range(thought_count):
        content = random.choice(scenario["thoughts"])
        node_id = f"node_{index}_th_{i}"
        thought_ids.append(node_id)

        nodes.append({
            "id": node_id,
            "content": content,
            "category": "episodic_thought",
            "type": "episodic",
            "timestamp": base_time + i * 3600 + 1800,  # 插在活动中间
            "embedding": get_random_embedding(),
            "meta": {}
        })

    # 3. 偶尔生成 Profile (30% 概率)
    profile_id = None
    if random.random() < 0.3:
        template = random.choice(scenario["profiles"])
        content = template.format(user_name)
        profile_id = f"node_{index}_profile"

        nodes.append({
            "id": profile_id,
            "content": content,
            "category": "semantic_profile",
            "type": "conceptual",
            "timestamp": base_time,
            "embedding": get_random_embedding(),
            "meta": {}
        })

    # === 生成边 ===
    links = []

    # TEMPORAL: 串联 Activity
    for i in range(len(act_ids) - 1):
        links.append({
            "source": act_ids[i],
            "target": act_ids[i + 1],
            "type": "TEMPORAL",
            "weight": 1.0
        })

    # SEMANTIC: Thought -> Random Activity
    for th_id in thought_ids:
        target = random.choice(act_ids)
        links.append({
            "source": th_id,
            "target": target,
            "type": "SEMANTIC",
            "weight": 0.8 + random.random() * 0.2
        })

    # ABSTRACT: Profile -> All Activities (模拟涌现)
    if profile_id:
        for act_id in act_ids:
            links.append({
                "source": profile_id,
                "target": act_id,
                "type": "ABSTRACT",
                "weight": 1.0
            })

    # IMPLICIT: 随机挑两个没连的点 (模拟 GNN 发现)
    if len(act_ids) >= 2 and random.random() < 0.5:
        u = act_ids[0]
        v = act_ids[-1]
        links.append({
            "source": u,
            "target": v,
            "type": "IMPLICIT",
            "weight": 0.75
        })

    # VERSION: 极低概率生成冲突 (模拟 Evolution)
    if profile_id and random.random() < 0.1:
        # 生成一个旧版本节点
        old_id = f"node_{index}_profile_old"
        nodes.append({
            "id": old_id,
            "content": f"{user_name} used to dislike this topic",
            "category": "semantic_profile",
            "type": "conceptual",
            "timestamp": base_time - 864000,  # 很久以前
            "embedding": get_random_embedding(),
            "energy_level": 0.5  # 衰减
        })
        links.append({
            "source": profile_id,
            "target": old_id,
            "type": "VERSION",
            "weight": 1.0
        })

    return {
        "source_id": f"sample_{index}",
        "graph_data": {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": nodes,
            "links": links
        },
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(links),
            "scenario": scenario_key
        }
    }


def generate_mock_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 开始生成 {SAMPLE_COUNT} 条高仿真 Mock 数据...")
    print(f"   - 向量维度: {EMBEDDING_DIM}")
    print(f"   - 场景覆盖: {list(SCENARIOS.keys())}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in range(SAMPLE_COUNT):
            sample = generate_single_sample(i)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                print(f"   ✅ 已生成 {i + 1} 条...")

    print(f"🎉 生成完成！文件已保存至: {OUTPUT_FILE}")
    print(f"   文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")


if __name__ == "__main__":
    generate_mock_data()