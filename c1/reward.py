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

import json
import re
from typing import List, Dict


def extract_json_content(text: str) -> Dict:
    """辅助函数：从模型输出中提取 JSON"""
    try:
        # 尝试寻找 markdown json 代码块
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # 尝试直接解析
        return json.loads(text)
    except:
        return None


def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    【维度1：格式规范性】
    奖励目标：输出必须是合法的 JSON，且包含双流四维的 Key。
    """
    rewards = []
    # 必须严格匹配 c1/prompts.py 中的定义
    required_keys = {"semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"}

    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)  # 格式错误直接 0 分
            continue

        # 检查 Key 是否齐全
        keys = set(data.keys())
        if required_keys.issubset(keys):
            rewards.append(1.0)  # 完美格式
        else:
            # 缺失 Key 扣分
            missing = len(required_keys - keys)
            rewards.append(max(0.0, 1.0 - missing * 0.2))

    return rewards


def orthogonality_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    【维度2：双流正交性】(核心创新点)
    奖励目标：Semantic Stream 和 Episodic Stream 的内容重叠度要低。
    逻辑：
       - Semantic Stream = Profile + Knowledge (静态/抽象)
       - Episodic Stream = Activity + Thought (动态/具体)
       两者不应包含相同的长文本片段。
    """
    rewards = []
    for completion in completions:
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        # 1. 聚合两个流的文本
        # 注意：这里需要处理 list 为空的情况
        s_profile = data.get("semantic_profile", []) or []
        s_knowledge = data.get("semantic_knowledge", []) or []
        e_activity = data.get("episodic_activity", []) or []
        e_thought = data.get("episodic_thought", []) or []

        semantic_text = " ".join([str(x) for x in s_profile + s_knowledge])
        episodic_text = " ".join([str(x) for x in e_activity + e_thought])

        if not semantic_text or not episodic_text:
            rewards.append(0.5)  # 其中一个为空，无法判断正交，给个中间分
            continue

        # 2. 计算 Token 级别的 Jaccard 相似度
        s_tokens = set(semantic_text.lower().split())
        e_tokens = set(episodic_text.lower().split())

        # 移除停用词 (防止 the, is, a 导致的高重叠)
        stopwords = {'the', 'is', 'a', 'an', 'and', 'to', 'of', 'in', 'i', 'you', 'he', 'she', 'it', 'this', 'that'}
        s_tokens -= stopwords
        e_tokens -= stopwords

        if not s_tokens or not e_tokens:
            rewards.append(1.0)  # 无有效词重叠，视为正交
            continue

        intersection = s_tokens.intersection(e_tokens)
        union = s_tokens.union(e_tokens)

        iou = len(intersection) / len(union) if union else 0

        # 3. 奖励函数：IoU 越低越好
        # 这是一个软约束，完全正交(IoU=0)得 1.0 分，重叠一半得 0.5 分
        rewards.append(1.0 - iou)

    return rewards


def completeness_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    【维度3：信息完备性】
    奖励目标：Prompt (原始对话) 中的关键实体应该出现在 Output (任意流) 中。
    解释：只要能在 Output 的任意位置找到该实体，就算成功。
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        data = extract_json_content(completion)
        if data is None:
            rewards.append(0.0)
            continue

        # 获取所有提取出的内容 (Flatten)
        all_values = []
        for key in ["semantic_profile", "semantic_knowledge", "episodic_activity", "episodic_thought"]:
            val_list = data.get(key, [])
            if isinstance(val_list, list):
                all_values.extend([str(v) for v in val_list])

        all_extracted_text = " ".join(all_values).lower()

        # 简单的关键词覆盖率检查
        # 策略：检查 Prompt 中长度 > 4 的词 (简单筛选实体/动词) 是否被包含
        # 注意：这里假设 prompts 输入的是原始对话文本
        # 如果 prompts 包含 system prompt，需要截取 user input 部分

        # 简单清洗 Prompt
        prompt_text = prompt.lower()
        if "dialogue stream:" in prompt_text:
            prompt_text = prompt_text.split("dialogue stream:")[-1]

        prompt_words = [w for w in prompt_text.split() if len(w) > 4]

        if not prompt_words:
            rewards.append(1.0)
            continue

        hit_count = sum(1 for w in prompt_words if w in all_extracted_text)
        recall = hit_count / len(prompt_words)

        rewards.append(recall)

    return rewards

