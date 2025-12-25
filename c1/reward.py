# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：reward.py
@Author  ：niu
@Date    ：2025/12/24 13:09 
@Desc    ：
"""

# c1/reward.py
import json
import re


class RewardScorer:
    """
    【研究内容一核心】自动奖励打分器
    用于 RFT (拒绝采样微调) 阶段的数据筛选。
    不依赖人工标注，基于规则自动评价分解质量。
    """

    def compute_score(self, raw_input: str, model_output_json: dict) -> float:
        """
        计算总分
        :param raw_input: 原始输入文本
        :param model_output_json: 模型解析后的字典
        :return: 分数 (越高越好)
        """
        score = 0.0

        # 1. 【格式完整性奖励】 (基础分)
        # 只要能被解析且包含四大金刚 Key，就及格
        required_keys = ["event", "entity", "knowledge", "rule"]
        if all(key in model_output_json for key in required_keys):
            score += 2.0
        else:
            return -10.0  # 格式残缺直接淘汰

        # 2. 【正交性奖励】 (关键创新点)
        # Event (动态) 和 Rule (静态偏好) 不应重叠
        # 如果 "用户喜欢吃苹果" 既在 event 又在 rule，说明分解不清
        events = " ".join(model_output_json["event"])
        rules = " ".join(model_output_json["rule"])

        if self._check_overlap(events, rules):
            score -= 2.0  # 惩罚混淆
        else:
            score += 1.0

        # 3. 【去噪/压缩奖励】
        # 输出的总字数应该少于输入字数 (去除废话)
        # 但也不能太少 (信息丢失)
        total_output_len = sum(len(str(v)) for v in model_output_json.values())
        input_len = len(raw_input)

        compression_ratio = total_output_len / (input_len + 1e-5)

        if 0.3 <= compression_ratio <= 0.8:
            score += 2.0  # 完美的压缩区间
        elif compression_ratio > 1.0:
            score -= 1.0  # 啰嗦，可能有幻觉
        elif compression_ratio < 0.2:
            score -= 1.0  # 删减过度

        # 4. 【规则敏感度奖励】
        # 如果输入里有 "不要"、"必须"、"喜欢"，但 Rule 为空，重罚
        # 这就是强迫模型对齐用户的指令
        strong_keywords = ["不要", "必须", "喜欢", "讨厌", "总是"]
        if any(k in raw_input for k in strong_keywords):
            if not model_output_json["rule"]:
                score -= 3.0  # 漏掉了关键规则！
            else:
                score += 2.0  # 捕获到了规则

        return score

    def _check_overlap(self, text_a, text_b):
        """简单的重叠检测 (可以用 Jaccard 或 关键词匹配)"""
        set_a = set(text_a.replace("用户", "").split())  # 去除通用主语
        set_b = set(text_b.replace("用户", "").split())
        # 如果交集过大，认为重叠
        intersection = set_a & set_b
        return len(intersection) > 2  # 容忍2个词的重叠

