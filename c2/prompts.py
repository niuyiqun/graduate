# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：prompts.py
@Author  ：niu
@Date    ：2026/1/8 14:58 
@Desc    ：
"""

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

# 2. 冲突检测提示词 (用于 EvolutionBuilder)
CONFLICT_DETECTION_PROMPT = """
任务：判断两段记忆是否存在【事实冲突】。
旧记忆：{old_text}
新记忆：{new_text}

规则：
1. 如果两者描述的事实完全相反（例如：旧的说在北京，新的说在上海；旧的说喜欢，新的说讨厌），请认为存在冲突。
2. 如果只是补充信息或无关信息，请认为无冲突。

请严格按照以下 JSON 格式返回结果：
{{
    "is_conflict": "YES"
}}
或者
{{
    "is_conflict": "NO"
}}
"""

# 3. 逻辑验证提示词 (用于 StructuralBuilder)
LOGIC_VERIFICATION_PROMPT = """
任务：判断两句话是否存在【潜在逻辑关联】（因果、条件、顺序、互补）。
A: {text_a}
B: {text_b}

规则：
1. 如果有合理的逻辑联系，状态为 PASS。
2. 如果完全风马牛不相及，状态为 REJECT。

请严格按照以下 JSON 格式返回结果：
{{
    "status": "PASS"
}}
或者
{{
    "status": "REJECT"
}}
"""
