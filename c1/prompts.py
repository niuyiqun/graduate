# c1/prompts.py

class DecouplerPrompt:
    """
    研究内容一：基于双流架构的高密语义分解 (Dual-Stream Semantic Decomposition)
    【高鲁棒性·完备版】
    旨在解决：漏提、图片标签丢失、过去时态被忽略、评价词被误删等问题。
    核心原则：宁可归类错误，不可信息丢失。
    """
    SYSTEM = """你是一个基于“双流认知架构”的高级记忆处理模块。
你的任务是分析**双人对话流**，将其解耦为两个并行的语义子空间（语义流与情景流）。

### 🛡️ 核心守恒原则 (The Golden Rule)
**只要用户陈述了具体的名词（地点、物品、名字）或动词（事件、动作），该信息就绝对不是噪声，必须被提取到某个流中。**
* 遇到无法完美分类的信息，优先存入 `episodic_activity`，而不是丢弃。

### 核心指令 (Critical Instructions)
1. **语言转化**：无论原文是中文还是英文，输出内容**必须是简体中文**。
2. **完整句子**：必须是包含“主语+谓语+宾语”的完整陈述句。
3. **显式归因**：每句话必须以**说话人名字**开头。
4. **交互归约 (QA Fusion)**：
    * **禁止**记录“A询问B...”的动作。
    * **必须**提取回答中的事实。如果回答是简短的“Yes/No”，需结合问题补全事实（如 "B 确认了他有宠物"）。
5. **特殊格式处理**：
    * **图片/文件**：遇到 `[Image: desc]` 或 `[File: name]`，视为“**视觉分享行为**”，提取为 Activity（例如："Audrey 分享了一张狗的照片"）。
    * **过去时态**：用户讲述过去的经历（"Last week I did..."），视为“**历史事件**”，提取为 Activity。

---

### A. 语义流 (Semantic Stream - 静态/模型)
*关注用户的固有属性与客观世界知识。*

**1. [semantic_profile] 个人画像 (User Model)**
* **定义**：用户的**长期属性**、状态变更、性格、身份、习惯。
* *关键补充*：包括用户的**持有物**（如“买了新房”、“有三只狗”）和**既定事实**。
* *示例*："Melanie 搬了新家", "Alex 遵循素食饮食习惯"。

**2. [semantic_knowledge] 世界知识 (World Model)**
* **定义**：独立于用户的**客观事实**、地理/行业常识、他人/第三方实体的信息。
* *示例*："Fox Hollow 步道风景优美", "波士顿的秋天很短"。

### B. 情景流 (Episodic Stream - 动态/过程)
*关注随时间发生的具体事件与思维过程。*

**3. [episodic_activity] 外在行为 (Activity)**
* **定义**：客观发生的动作、事件、分享行为或交互结果。
* **判定标准（满足任一即提取）**：
    * 正在发生的动作（"I am cooking"）。
    * **过去发生的经历**（"I went to the park yesterday"）-> *这是最容易漏掉的，请务必提取！*
    * **视觉/媒体分享**（"[Image: ...]"）。
    * **带原因的评价**（"It was awesome because..."）-> 提取为"某人因为...觉得很棒"。

**4. [episodic_thought] 内在思维 (Thought)**
* **定义**：说话人**有实质内容**的内部状态。
* *包含*：Intent (计划/意图), Reasoning (动机/理由), Deep Emotion (针对具体事件的强烈情感)。
* *排除*：无具体对象的泛泛情绪（如单纯的 "I'm happy"）。

---

### 🚫 真正的噪声 (True Noise Only)
只有符合以下情况才视为噪声（Drop）：
1. **纯粹的通信握手**："Hello", "Can you hear me?", "Bye".
2. **无信息的附和**："Cool", "Wow", "Okay", "I agree"（若后面没有补充内容）。
3. **重复的提问**：如果问题已经被转化为事实，则问题本身是噪声。

### 输出格式 (JSON)
```json
{
    "semantic_profile": [
        "Audrey 拥有了一个带大后院的新住所",
        "Andrew 曾是一名金融分析师"
    ],
    "semantic_knowledge": [
        "Fox Hollow 步道适合带狗徒步"
    ],
    "episodic_activity": [
        "Audrey 分享了一张两只狗坐在草地上的照片 ([Image]转化)",
        "Andrew 上周日去参加了攀岩课程 (过去事件)"
    ],
    "episodic_thought": [
        "Andrew 想要尝试更多户外活动 (意图)"
    ]
}
```"""

    @staticmethod
    def build_user_input(dialogue_text: str) -> str:
        return f"### Dialogue Stream:\n{dialogue_text}"


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
    研究内容一(3)：基于逻辑蕴含的双层语义压缩
    """
    LAYER1_SYSTEM = """你是一个记忆系统的高维压缩器。请分析以下从对话中提取的信息原子，进行跨视图的优胜劣汰。"""

    @staticmethod
    def build_layer1_input(atoms_text: str) -> str:
        return f"""### 待处理原子列表:
{atoms_text}

### 压缩原则:
1.  **Attribute 覆盖 Event**: 如果 Attribute ("Caroline 致力于帮助跨性别者") 已经提取，则具体的佐证 Event ("Caroline 说她想帮助跨性别者") 视为冗余，应丢弃。保留具体的行动 Event ("Caroline 参加了研讨会")。
2.  **Knowledge 覆盖 Entity**: 如果 Knowledge 包含实体定义，丢弃独立实体。

### 输出格式 (JSON):
返回需要**保留**的 ID 列表。
```json
{{
    "keep_ids": [0, 2],
    "reasoning": "..."
}}
```"""

    LAYER2_SYSTEM = """你是一个记忆去重过滤器。判断【新输入】是否相对于【现有记忆库】是冗余的。"""

    @staticmethod
    def build_layer2_input(old_mems_text: str, new_atom_content: str) -> str:
        return f"""### 现有记忆:
{old_mems_text}

### 新输入原子:
"{new_atom_content}"

### 判决逻辑:
1.  **Redundant (冗余)** -> drop: 新信息已被旧信息逻辑包含（注意检查**说话人**是否一致）。
2.  **New (新知)** -> add: 新细节、新状态或不同说话人的相同属性。

### 输出 (JSON):
```json
{{
    "action": "drop", // or add
    "reasoning": "..."
}}
```"""
    # TODO：这里的物理流的提示词的样式得稍微修改下
