# c1/prompts.py


# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：prompts.py
@Desc    ：双流语义解耦的 Prompt 模板定义
          (专为 滑动窗口 / 轮次级提取 优化)
"""


class DecouplerPrompt:
    """
    GRPO 训练专用的 Prompt 模板类。
    任务：利用历史记录作为上下文，仅从【当前轮次】中提取原子记忆。
    """

    # SYSTEM Prompt 必须保持英文，以强迫模型输出英文结果
    SYSTEM = """You are an advanced memory processing module.
Your task is to analyze the **CURRENT TURN** of a dialogue and decouple it into orthogonal semantic atoms.

### 🚨 Critical Instruction: Scope of Extraction
1.  **Focus ONLY on the 'Current Turn'**: You must extract information *only* if it appears or is implied in the "Current Turn" section.
2.  **Use History for Context**: Read the "Dialogue History" *only* to resolve pronouns (e.g., know who "he" is) or understand the topic. **DO NOT** extract facts that appear *only* in the history.
3.  **Atomic & Concise**: Extract short, atomic facts (Subject + Verb + Object). Avoid redundancy.

---

### A. Semantic Stream (Static / Abstract)
*Intrinsic attributes and objective world knowledge.*

**1. [semantic_profile] User Model**
* **Definition**: Long-term attributes, personality, habits, identity, relationships.
* *Example*: "Alex is a vegetarian.", "Bella owns a cat named Oreo."
* *Constraint*: Do NOT record temporary states (e.g., "Alex is hungry" -> NO).

**2. [semantic_knowledge] World Model**
* **Definition**: Objective facts, common sense, definitions independent of the speakers.
* *Example*: "Python is a programming language.", "The Alps are in Europe."

### B. Episodic Stream (Dynamic / Concrete)
*Specific events and internal thoughts tied to a timeline.*

**3. [episodic_activity] Outer Activity**
* **Definition**: Specific actions, events, past experiences, or behaviors happening NOW or in the PAST.
* *Example*: "Charlie went to the gym yesterday.", "Diana is cooking pasta."

**4. [episodic_thought] Inner Thought**
* **Definition**: Specific intentions, opinions, motivations, or feelings about a specific event.
* *Example*: "Ethan wants to lose weight.", "Fiona found the movie boring."

---

### Output Format (JSON)
Return an empty list `[]` if no NEW information is present in the current turn.
```json
{
    "semantic_profile": [],
    "semantic_knowledge": ["Extract FACTS from the current turn"],
    "episodic_activity": ["Extract EVENTS from the current turn"],
    "episodic_thought": []
}
```"""

    @staticmethod
    def build_user_input(history_text: str, current_turn_text: str) -> str:
        """
        构造用户输入：清晰地物理隔离【上下文】和【提取目标】。
        让模型一眼就能看出它该从哪段文字里提取信息。
        """
        return f"""### Dialogue History (Context ONLY - Do NOT Extract):
{history_text}

### Current Turn (TARGET - Extract Here):
{current_turn_text}"""


class VerifierPrompt:
    """
    研究内容一(2)：自监督反事实一致性校验
    (适配双人对话，重点检查张冠李戴)
    """
    SYSTEM = """你是一个严谨的认知记忆系统校验员。你的任务是基于【原始对话历史】检测提取的记忆是否包含'幻觉'或'归因错误'。"""

    @staticmethod
    def build_input(raw_dialogue: str, atoms_list_str: str) -> str:
        return f"""以下是【原始对话历史】和系统从中提取的【待验证记忆原子】。

### 原始对话历史 (Ground Truth):
"{raw_dialogue}"

### 待校验的记忆原子 (Claims):
{atoms_list_str}

### 校验任务:
1.  **事实一致性**: 提取的内容是否真实存在于对话中？
2.  **归因准确性 (Attribution Check)**: **这是重点**。
    * 系统是否把 Speaker A 的经历安到了 Speaker B 头上？
    * 例如：原文是 Caroline 想做咨询师，如果提取为 "Melanie 想做咨询师"，必须判定为 **False**。

### 输出格式 (JSON):
```json
{{
    "verification_results": [
        {{
            "index": 1,
            "is_consistent": true,
            "reasoning": "原文明确提到 Caroline 说..."
        }},
        {{
            "index": 2,
            "is_consistent": false,
            "reasoning": "归因错误：原文中提到去研讨会的是 Caroline，而不是 Melanie。"
        }}
    ]
}}
```"""


class DeduplicatorPrompt:
    """
    研究内容一(3)：基于逻辑博弈与预测偏差的双层压缩
    【修正版】：修正了“惊奇度”定义，防止过度删除合乎逻辑的新事件。
    """

    # === Layer 1: 批次内博弈 ===
    LAYER1_SYSTEM = """你是一个基于“信息增益”的记忆博弈仲裁器。
你的任务是分析一批提取出的信息原子，根据【四视图博弈矩阵】决定哪些需要保留，哪些是冗余的。

### 核心判据：信息增益 (Information Gain)
不要简单地让 Rule 覆盖 Event。必须判断 Event 是否提供了 Rule 之外的**新细节**。
* **Redundant (冗余 - 丢弃)**: 
    - 完全重复的内容。
    - 只是同义改写 (e.g. "他很开心" vs "他非常高兴")。
    - Rule 已经完全包含了 Event 的所有信息 (e.g. Rule="不吃辣", Event="没吃辣")。
* **Informative (有益 - 保留)**: 
    - Rule="喜辣", Event="今天尝试了特辣火锅并拉肚子"。 (包含具体实例/后果 -> 保留)
    - Event A="User is busy", Event B="User is working on a project". (Event B 提供了具体细节 -> 保留)
    - **不同主体**的动作永远互不冗余 (e.g. "A is sad" 不能覆盖 "B is happy")。
"""

    @staticmethod
    def build_layer1_input(atoms_text: str) -> str:
        return f"""### 待仲裁原子 (The Players):
{atoms_text}

### 仲裁指令:
请返回一个 JSON，包含需要**保留 (Keep)** 的 ID 列表。
对于没被选中的原子，视为冗余被淘汰。

### 输出格式:
```json
{{
    "keep_ids": [0, 2],
    "reasoning": "ID 1 (喜欢咖啡) 被 ID 0 (每天喝咖啡的习惯) 逻辑包含且无新细节，故淘汰。"
}}
```"""

    # === Layer 2A: Episodic Stream (预测偏差) ===
    # 🔥 核心修改：重新定义了“惊奇度”，防止误杀新事件
    LAYER2_EPISODIC_SYSTEM = """你是一个“情节新颖性检测器 (Episodic Novelty Detector)”。
你的任务是判断【新事件 (New Event)】相对于【现有上下文 (Context)】是否包含**事实性新信息**。

### ⚠️ 重要原则:
- **不要**仅仅因为新事件“合乎逻辑”或“符合人设”就将其视为冗余。
- **不要**因为 Context 里有一个宽泛的状态（如“User is busy”）就丢弃具体的后续动作（如“User finished the report”）。

### 判定标准:
1. **Low Surprise (冗余/完全重复)** -> **Drop**:
   - 只有当 Context **已经显式包含**了该 Event 的核心信息（时间、地点、人物、动作）时。
   - 或者 Event 只是 Context 的简单摘要/复述。
   - 例子: Context="Bob ate an apple at 2pm.", Event="Bob had fruit this afternoon." -> Low Surprise.

2. **High Surprise (新信息/细节补充)** -> **Keep**:
   - 这是一个新发生的时间片段，Context 里没记录过。
   - 即使它符合 Character 的性格（预测之内），只要是**具体的、未记录过的实例**，就是 High Surprise。
   - 例子: Context="Bob likes apples.", Event="Bob ate an apple at 2pm." -> **High Surprise** (具体的实例化行为).
   - 例子: Context="Melanie is busy.", Event="Caroline went to a party." -> **High Surprise** (不同主体的独立事件).
"""

    @staticmethod
    def build_episodic_predict_input(old_mems_text: str, new_atom_content: str) -> str:
        return f"""### 现有上下文 (Context):
{old_mems_text}

### 新发生事件 (New Event):
"{new_atom_content}"

### 任务:
1. 检查 Context 中是否**已经存在**这个具体事件？
2. 如果 Context 只是暗示了可能性，但没记录具体发生，则必须视为 **High Surprise**。

返回 json: {{"surprise_level": "low" | "high", "reasoning": "..."}}
"""

    # === Layer 2B: Semantic Stream (逻辑蕴含) ===
    LAYER2_SEMANTIC_SYSTEM = """你是一个“知识整合器”。
你的任务是判断【新规则】是否被【旧规则】逻辑蕴含 (Entailment)。

### 判定标准:
1. **Drop (被蕴含/重复)**:
   - 旧规则是上位概念，完全覆盖新规则 (e.g., Old="擅长所有球类运动", New="会打篮球")。
   - 语义完全重复。

2. **Add (新知识/特例)**:
   - 新规则包含旧规则未提及的属性。
   - 新规则修正了旧规则。
   - 只要有任何事实性细节差异，就选择 Add。
"""

    @staticmethod
    def build_semantic_entailment_input(old_mems_text: str, new_atom_content: str) -> str:
        return f"""### 现有知识库 (Knowledge Base):
{old_mems_text}

### 待入库新知 (New Knowledge):
"{new_atom_content}"

### 任务:
决策动作: "add" 或 "drop"。
返回 json: {{"action": "add" | "drop", "reasoning": "..."}}
"""
