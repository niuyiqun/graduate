# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：reward.py
@Author  ：niu
@Date    ：2025/12/24 13:09 
@Desc    ：GRPO 多维正交奖励函数定义 (适配双流四维架构)
          包含：格式约束、正交性检测、完备性检测
"""

# c1/reward.py
import json
import re
from typing import List, Dict


def extract_json_content(text: str) -> Dict:
    """辅助：提取 JSON"""
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except:
        return None


# =============================================================================
# 1. 格式奖励 (基础)
# =============================================================================
def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """必须是合法 JSON 且包含 4 个 Key"""
    rewards = []
    required_keys = {"semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"}

    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        keys = set(data.keys())
        if required_keys.issubset(keys):
            rewards.append(1.0)
        else:
            rewards.append(0.5)  # 格式不对扣分
    return rewards


# =============================================================================
# 2. 正交性奖励 (解决"情景被提取成概念"的问题)
# =============================================================================
def orthogonality_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    语义流 (Profile/Knowledge) 和 情景流 (Activity/Thought) 不应有重叠。
    如果一个信息既被放进了 Profile 又被放进了 Activity，说明模型分类不清，要重罚。
    """
    rewards = []
    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        # 提取 Set 用于对比
        s_list = (data.get("semantic_profile", []) or []) + (data.get("semantic_knowledge", []) or [])
        e_list = (data.get("episodic_activity", []) or []) + (data.get("episodic_thought", []) or [])

        # 转换为小写单词集合
        s_text = " ".join([str(x) for x in s_list]).lower()
        e_text = " ".join([str(x) for x in e_list]).lower()

        # 简单分词
        s_tokens = set(s_text.split())
        e_tokens = set(e_text.split())

        # 移除停用词
        stopwords = {'the', 'is', 'a', 'an', 'and', 'to', 'of', 'in', 'user', 'speaker'}
        s_tokens -= stopwords
        e_tokens -= stopwords

        if not s_tokens or not e_tokens:
            rewards.append(1.0)  # 没有内容就没有冲突
            continue

        # 计算重叠 (IoU)
        intersection = s_tokens.intersection(e_tokens)
        union = s_tokens.union(e_tokens)
        iou = len(intersection) / len(union) if union else 0

        # IoU 越高，惩罚越重。我们要的是完全正交 (IoU = 0)
        rewards.append(1.0 - iou)

    return rewards


# =============================================================================
# 3. [新] 原子性与简洁奖励 (解决"冗余/罗嗦"问题)
# =============================================================================
def brevity_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    奖励简短、原子的事实。惩罚长篇大论。
    原则：一个好的原子记忆通常在 5-15 个单词之间。
    """
    rewards = []
    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        all_items = []
        for key in data:
            if isinstance(data[key], list):
                all_items.extend(data[key])

        if not all_items:
            rewards.append(1.0)  # 空提取也是一种简洁（如果没有信息的话）
            continue

        # 评分逻辑
        scores = []
        for item in all_items:
            item_str = str(item)
            word_count = len(item_str.split())

            # 完美区间：3 到 15 个词
            if 3 <= word_count <= 15:
                scores.append(1.0)
            # 太短 (可能信息缺失)
            elif word_count < 3:
                scores.append(0.5)
            # 太长 (罗嗦，可能是原始句子的直接复制)
            else:
                # 长度超过 20 个词开始线性扣分
                penalty = max(0.0, 1.0 - (word_count - 15) * 0.05)
                scores.append(penalty)

        # 取所有原子的平均分
        rewards.append(sum(scores) / len(scores))

    return rewards


# =============================================================================
# 4. [新] 内部去重奖励 (解决"同一件事说两遍"问题)
# =============================================================================
def deduplication_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    同一个 JSON 内，不同字段不应该包含极度相似的内容。
    比如 Profile 里有了 "Likes apples"，Thought 里就别再来 "User likes apples"
    """
    rewards = []
    from difflib import SequenceMatcher

    def is_similar(a, b):
        return SequenceMatcher(None, a, b).ratio() > 0.6  # 相似度阈值

    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        all_items = []
        for key in data:
            if isinstance(data[key], list):
                all_items.extend([str(x).lower() for x in data[key]])

        if len(all_items) < 2:
            rewards.append(1.0)
            continue

        # 两两对比寻找重复
        conflict_count = 0
        total_pairs = 0
        for i in range(len(all_items)):
            for j in range(i + 1, len(all_items)):
                total_pairs += 1
                if is_similar(all_items[i], all_items[j]):
                    conflict_count += 1

        # 极其严格：发现一对重复就扣分
        if total_pairs > 0:
            rewards.append(max(0.0, 1.0 - conflict_count * 0.5))
        else:
            rewards.append(1.0)

    return rewards

