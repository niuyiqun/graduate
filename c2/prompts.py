# -*- coding: UTF-8 -*-
# c2/prompts.py

# 1. 实体提取提示词 (用于 SemanticBuilder)
ENTITY_EXTRACTION_PROMPT = """
任务：从下面的文本中提取关键实体（人名、地名、组织、核心概念）。
文本："{text}"

请严格按照以下 JSON 格式返回结果：
{{
    "entities": ["实体1", "实体2"]
}}
注意：不要输出任何多余的解释，必须是合法的 JSON。
"""

# 2. 冲突检测提示词 (用于 EvolutionBuilder - 严格版)
# 使用 Few-Shot 示例来强制模型遵循格式
CONFLICT_DETECTION_PROMPT = """
任务：判断两句关于用户的描述是否存在【事实上的逻辑冲突】。

示例 1:
A: 我喜欢吃辣。
B: 我一点辣都不能吃。
回答: YES

示例 2:
A: 我喜欢吃辣。
B: 我也喜欢吃甜食。
回答: NO

当前输入:
A: {text_a}
B: {text_b}

请仅回答 YES 或 NO。不要输出任何解释，不要输出标点符号。
"""

# 3. 逻辑验证提示词 (用于 StructuralBuilder - 严格版)
LOGIC_VERIFICATION_PROMPT = """
任务：判断两句话是否存在【潜在逻辑关联】（如因果、条件、顺序、互补）。
A: {text_a}
B: {text_b}

如果存在明显关联，请回答 YES。
如果是完全无关的话题，请回答 NO。

请仅回答 YES 或 NO。
"""

# 4. 概念抽象提示词 (用于 EmergenceBuilder)
CONCEPT_ABSTRACTION_PROMPT = """
以下是一组用户的具体行为记忆片段，它们在图结构上紧密关联：
{context_str}

任务：请分析这些行为背后的共同模式、用户性格特质或高层抽象概念。
输出：生成一条简短的“用户画像（Profile）”或“一般性知识（Knowledge）”。
要求：仅输出结论，不要解释。不要包含"根据..."等字样。
"""