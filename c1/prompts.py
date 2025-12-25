# c1/prompts.py

class DecouplerPrompt:
    """
    研究内容一(1)：多粒度语义正交分解
    """
    # 使用圆括号拼接字符串，避免使用三引号，防止 Markdown 渲染错误
    SYSTEM = """你是一个基于“双流认知架构”的先进记忆系统预处理模块。你的任务是将用户的长程对话流进行**多粒度语义正交分解**。
请将输入的对话内容映射到以下四个**互斥**的特征子空间：

### 1. [Event] 情景流 (Episodic Stream)
* **定义**：具有时间戳的动态交互过程，记录“谁在什么时候做了什么”。
* **格式**：必须包含动作的主体和具体内容，尽量保留因果关系。

### 2. [Entity] 实体 (Entity)
* **定义**：对话中涉及的关键名词、人名、地名、专有名词。

### 3. [Knowledge] 语义流-世界知识 (Semantic Knowledge)
* **定义**：客观的、独立于当前对话语境存在的真理或常识（如“苹果属于蔷薇科”）。

### 4. [Rule] 语义流-用户规则 (User Rules/Preferences)
* **定义**：用户的显式指令、偏好、习惯或约束条件（如“以后不要给我推荐恐怖片”）。
* **重要性**：这是系统必须遵守的长期约束。

---
### 约束条件 (Constraints)
1. **去除噪声**：忽略口语助词和无实质内容的重复。
2. **正交性**：确保一条信息只属于一个类别。
3. **JSON输出**：必须严格输出 Markdown 格式的 JSON 代码块。

### 输出格式示例
```json
{
    "event": ["用户询问了杭州的天气"],
    "entity": ["杭州", "西湖"],
    "knowledge": ["西湖是杭州的著名景点"],
    "rule": ["用户偏好：查询天气时需要包含湿度信息"]
}
```"""

    @staticmethod
    def build_user_input(timestamp, text) -> str:
        return f"Time: {timestamp}\nText: {text}"


class VerifierPrompt:
    """
    研究内容一(2)：自监督反事实一致性校验 (语义剪枝)
    """
    SYSTEM = """你是一个严谨的认知记忆系统校验员。你的任务是基于原始文本检测提取的记忆是否包含'幻觉'或'过度推断'。"""

    @staticmethod
    def build_input(raw_text: str, atoms_list_str: str) -> str:
        return f"""以下是用户的【原始输入】和系统提取出的【待验证记忆原子】。
请利用**自监督反事实校验机制 (Self-Supervised Counterfactual Check)** 对每一条记忆进行核查。

### 原始输入 (Ground Truth):
"{raw_text}"

### 待校验的记忆原子 (Claims):
{atoms_list_str}

### 校验任务:
对于列表中的每一条原子，请执行：
1. **生成探针 (Probe)**: 根据记忆内容，反向生成一个疑问句。
2. **回溯验证 (Recall)**: **完全基于【原始输入】**来回答这个问题。
3. **一致性判断 (Check)**: 比较你的回答与记忆原子是否事实一致。如果不一致或原文未提及，则判定为“幻觉”。

### 输出格式 (JSON):
```json
{{
    "verification_results": [
        {{
            "index": 1,
            "probe_question": "...",
            "ground_truth": "...",
            "is_consistent": true,
            "reasoning": "..."
        }}
    ]
}}
```"""


class DeduplicatorPrompt:
    """
    研究内容一(3)：基于逻辑蕴含的双层语义压缩
    """

    # Layer 1: 跨视图消解 (Cross-View)
    # 核心更新：引入四视图动态博弈逻辑，而非简单的 Rule > Event
    LAYER1_SYSTEM = """你是一个记忆系统的高维压缩器。"""

    @staticmethod
    def build_layer1_input(atoms_text: str) -> str:
        return f"""用户的一段话被分解为了多个视图（Event, Entity, Knowledge, Rule）。
这些视图之间可能存在语义重叠、包含或互补关系。请分析原子列表，进行**跨视图的优胜劣汰**。

### 待处理原子列表:
{atoms_text}

### 压缩与去重原则 (Cross-View Competition):
请基于 **'信息增益 (Information Gain)'** 进行判断：

1. **Generalization (归纳覆盖)**: 
   - 如果高维原子（如 `Rule`, `Knowledge`）完全概括了低维原子（如 `Event`, `Entity`），且后者无额外细节。
   - **决策**: 保留高维原子，丢弃低维原子。
   - *例子*: Rule='习惯晨跑' 覆盖 Event='今天晨跑了' -> 丢弃 Event。

2. **Specific Exception (特例保留)**: 
   - 如果低维原子包含了高维原子未提及的**异常、特例或具体数值**。
   - **决策**: **两者都保留**。
   - *例子*: Rule='习惯晨跑' vs Event='今天晨跑扭伤了脚' -> 全部保留。

3. **Contextualization (语境包含)**: 
   - 如果 `Knowledge` 完整包含了 `Entity` 的语义关系。
   - **决策**: 保留 Knowledge，丢弃独立的 Entity。
   - *例子*: Knowledge='西湖是杭州的景点' 覆盖 Entity='西湖', '杭州'。

4. **Subjective Priority (主观优先)**: 
   - 如果用户的 `Rule` (主观认知) 与 `Knowledge` (客观事实) 冲突，在个性化记忆中，**用户的主观偏好优先**。

### 输出格式 (JSON):
返回需要**保留**的 ID 列表。reasoning 字段需解释谁覆盖了谁。
```json
{{
    "keep_ids": [0, 2], 
    "reasoning": "ID[1](Event) 是 ID[0](Rule) 的重复例证；ID[3](Entity) 被 ID[2](Knowledge) 包含。"
}}
```"""

    # Layer 2: 全局去重 (Global)
    LAYER2_SYSTEM = """你是一个记忆去重过滤器。"""

    @staticmethod
    def build_layer2_input(old_mems_text: str, new_atom_content: str) -> str:
        return f"""判断【新输入】是否相对于【现有记忆】是冗余的。

### 现有记忆:
{old_mems_text}

### 新输入原子:
"{new_atom_content}"

### 判决逻辑:
1. **Entailment (冗余)** -> **drop**: 新信息是旧信息的子集、重复表达、或已被旧规则覆盖。
2. **Novel/Conflict (新知)** -> **add**: 新信息提供了新细节，或者是状态变更（哪怕冲突也要存，留给第二章处理）。

### 输出 (JSON):
```json
{{
    "action": "drop", // or add
    "reasoning": "..."
}}
```
"""
