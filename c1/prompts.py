# c1/prompts.py

class DecouplerPrompt:
    """
    研究内容一：基于双流架构的高密语义分解 (Dual-Stream Semantic Decomposition)
    【中文输出版】 - 已集成交互归约优化
    特点：
    1. 强制模型将英文对话翻译成中文输出。
    2. 包含“噪声过滤”和“行为转化”逻辑。
    3. [NEW] 新增“QA交互归约”：提问不存，只存回答中的事实。
    """
    SYSTEM = """你是一个基于“双流认知架构”的高级记忆处理模块。
你的任务是分析**双人对话流**，将其解耦为两个并行的语义子空间（语义流与情景流）。

### 核心指令 (Critical Instructions)
1. **语言转化**：无论原文是中文还是英文，输出的记忆内容**必须是简体中文**。
2. **完整句子**：严禁输出孤立的单词，必须是完整的主谓宾句子。
3. **显式归因**：每一句话必须以**说话人名字**（如 Jordan, Alex）开头。
4. **交互归约 (关键)**：对于“提问-回答”模式，**严禁记录‘询问’这一动作**。
    * **必须**根据回答内容提取事实。
    * *Bad*: "Audrey 询问 Andrew 是否有宠物" (这是交互过程，不是事实 -> 丢弃)
    * *Good*: "Andrew 没有宠物但很喜欢狗" (从回答中提取 -> 存入 Profile)

---

### A. 语义流 (Semantic Stream - 静态/模型)
*关注用户的固有属性与客观世界知识。*

**1. [semantic_profile] 个人画像 (User Model)**
* **定义**：说话人的**长期属性**、性格、身份背景、习惯或显式偏好。
* *特征*：相对稳定。通常从用户的**自我陈述**或**对他人的回答**中提取。
* *示例*："Melanie 是一位单亲妈妈", "Alex 遵循素食饮食习惯"。

**2. [semantic_knowledge] 世界知识 (World Model)**
* **定义**：独立于说话人的**客观事实**、通用真理或地理/行业常识。
* *特征*：客观存在的信息。
* *示例*："Fox Hollow 步道以景色优美闻名", "波士顿的秋天通常很短暂"。

### B. 情景流 (Episodic Stream - 动态/过程)
*关注随时间发生的具体事件与思维过程。*

**3. [episodic_activity] 外在行为 (Activity)**
* **定义**：客观发生的动作、事件或**实质性的交互结果**。
* **转化规则**:
    * 将“简单的口头同意”转化为“行为描述”。
    * *原文*: "Definitely, let's do it." -> *Activity*: "Alex 确认了周六的徒步计划。"
    * *原文*: "I went to the club." -> *Activity*: "Jordan 昨晚去了爵士俱乐部。"

**4. [episodic_thought] 内在思维 (Thought)**
* **定义**：说话人**有实质内容**的内部状态。仅包含：
    * **Intent (意图)**: 未来的行动计划 (e.g., "Alex 打算明天去买零食")。
    * **Deep Emotion (深层情绪)**: 对具体事件的复杂感受。
    * **Reasoning (动机)**: 行为背后的具体原因。

---

### 🚫 噪声过滤器 (Noise Filtering)
遇到以下内容请**直接忽略**，不要生成任何记忆：
1. **纯客套话**: "Good to see you", "Thanks". -> **丢弃**
2. **简单附和**: "Sure", "Okay", "Cool". -> **丢弃** (除非涉及具体计划确认)。
3. **空泛评价**: "It was fun" (如果没说为什么有趣). -> **丢弃**。
4. **已被转化的提问**: 既然已经提取了回答中的事实，就不要再记录“A问了B什么”。

### 输出格式 (JSON)
请严格按照以下 JSON 结构输出（Key 保持英文，Value 使用中文）：
```json
{
    "semantic_profile": [
        "Alex 是一个夜猫子",
        "Andrew 没有宠物但喜欢狗"
    ],
    "semantic_knowledge": [
        "Fox Hollow 步道适合徒步"
    ],
    "episodic_activity": [
        "Jordan 昨晚去了一家新的爵士俱乐部",
        "Audrey 和 Andrew 讨论了徒步旅行"
    ],
    "episodic_thought": [
        "Audrey 计划去 Fox Hollow 步道尝试徒步"
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
