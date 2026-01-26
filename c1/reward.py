# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：reward.py
@Desc    ：GRPO 奖励函数 (最终适配版：支持 List/Dict 拆包)
"""

import json
import re
from typing import List, Dict, Any


# =============================================================================
# 0. 核心修复：智能拆包提取函数
# =============================================================================
def extract_json_content(text: Any) -> Dict:
    """
    智能提取：自动处理 String, List, Dict 等各种格式
    """
    content_str = ""

    # --- [Step 1] 拆包逻辑 ---
    if isinstance(text, str):
        content_str = text
    elif isinstance(text, list) and len(text) > 0:
        # 如果是 [{'role': 'assistant', 'content': '...'}]
        first_item = text[0]
        if isinstance(first_item, dict) and 'content' in first_item:
            content_str = first_item['content']
        # 也有可能是纯 list [123, 456] (Token IDs)，这种情况通常无法处理，暂且跳过
    elif isinstance(text, dict) and 'content' in text:
        content_str = text['content']

    # 如果拆包后还是空的，或者不是字符串，说明真没法提取
    if not isinstance(content_str, str) or not content_str:
        return None
    # -----------------------

    # --- [Step 2] 提取逻辑 ---
    try:
        # 尝试 Markdown
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_str, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # 尝试正则 {}
        match_raw = re.search(r'(\{.*\})', content_str, re.DOTALL)
        if match_raw:
            return json.loads(match_raw.group(1))

        # 尝试直接解析
        return json.loads(content_str)
    except:
        return None


# =============================================================================
# 1. 格式奖励
# =============================================================================
def format_reward_func(completions: List[Any], **kwargs) -> List[float]:
    rewards = []
    required_keys = {"semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"}

    # --- DEBUG: 打印一次确认拆包成功 ---
    if completions:
        print(f"\n{'=' * 20} [Checking First Sample] {'=' * 20}")
        # 这里模拟调用一下提取函数
        preview_data = extract_json_content(completions[0])
        print(f"Extraction Result: {type(preview_data)}")
        if preview_data:
            print("SUCCESS: JSON extracted successfully!")
        else:
            print("WARNING: Extraction returned None")
        print(f"{'=' * 60}\n")
    # --------------------------------

    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        keys = set(data.keys())
        if required_keys.issubset(keys):
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards


# =============================================================================
# 2. 正交性奖励
# =============================================================================
def orthogonality_reward_func(completions: List[Any], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        s_list = (data.get("semantic_profile", []) or []) + (data.get("semantic_knowledge", []) or [])
        e_list = (data.get("episodic_activity", []) or []) + (data.get("episodic_thought", []) or [])

        s_text = " ".join([str(x) for x in s_list]).lower()
        e_text = " ".join([str(x) for x in e_list]).lower()

        if not s_text and not e_text:
            rewards.append(1.0)
            continue

        s_tokens = set(s_text.split())
        e_tokens = set(e_text.split())
        stopwords = {'the', 'is', 'a', 'an', 'and', 'to', 'of', 'in', 'user', 'speaker', 'i', 'you'}
        s_tokens -= stopwords
        e_tokens -= stopwords

        if not s_tokens or not e_tokens:
            rewards.append(1.0)
            continue

        intersection = s_tokens.intersection(e_tokens)
        union = s_tokens.union(e_tokens)
        iou = len(intersection) / len(union) if union else 0

        rewards.append(1.0 - iou)

    return rewards


# =============================================================================
# 3. 简洁性奖励
# =============================================================================
def brevity_reward_func(completions: List[Any], **kwargs) -> List[float]:
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
            rewards.append(1.0)
            continue

        scores = []
        for item in all_items:
            item_str = str(item)
            word_count = len(item_str.split())

            if 3 <= word_count <= 15:
                scores.append(1.0)
            elif word_count < 3:
                scores.append(0.5)
            else:
                penalty = max(0.0, 1.0 - (word_count - 15) * 0.05)
                scores.append(penalty)

        if scores:
            rewards.append(sum(scores) / len(scores))
        else:
            rewards.append(1.0)

    return rewards


# =============================================================================
# 4. 去重奖励
# =============================================================================
def deduplication_reward_func(completions: List[Any], **kwargs) -> List[float]:
    rewards = []
    from difflib import SequenceMatcher

    def is_similar(a, b):
        return SequenceMatcher(None, a, b).ratio() > 0.6

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

        conflict_count = 0
        total_pairs = 0
        limit = min(len(all_items), 20)
        for i in range(limit):
            for j in range(i + 1, limit):
                total_pairs += 1
                if is_similar(all_items[i], all_items[j]):
                    conflict_count += 1

        if total_pairs > 0:
            rewards.append(max(0.0, 1.0 - conflict_count * 0.5))
        else:
            rewards.append(1.0)

    return rewards