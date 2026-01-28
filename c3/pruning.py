# c3/pruning.py

import numpy as np
from scipy.stats import entropy


class AdaptivePruner:
    """
    Step 4: 基于熵减的自适应剪枝
    """

    def __init__(self):
        self.base_threshold = 0.7

    def prune(self, candidates, scores, system_load=0.5):
        """
        candidates: 候选节点列表
        scores: 对应的相似度分数 (logits)
        system_load: 当前系统负载 (0~1)
        """
        # 归一化为概率分布
        probs = np.exp(scores) / np.sum(np.exp(scores))

        # 计算信息熵
        ent = entropy(probs)
        max_ent = np.log(len(probs))
        normalized_ent = ent / max_ent  # 0~1, 越高越迷茫

        # 动态调整阈值
        # 如果熵很低(自信)，阈值降低，允许Top-1/2通过
        # 如果熵很高(迷茫)，且负载高，大幅提高阈值，保守剪枝
        dynamic_threshold = self.base_threshold + 0.2 * normalized_ent + 0.1 * system_load

        final_nodes = []
        for node, score, prob in zip(candidates, scores, probs):
            if prob > dynamic_threshold:  # 这里的逻辑可根据具体score分布调整
                final_nodes.append(node)

        # 保底机制: 至少保留 Top-1，除非完全无关
        if not final_nodes and len(candidates) > 0:
            final_nodes.append(candidates[0])

        return final_nodes