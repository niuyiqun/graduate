# c1/deduplicator.py
import json
from typing import List
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# 导入通用结构
from general.decoupled_memory import DecoupledMemoryAtom
from general.model import BaseModel
from general.base_memory import AgenticMemorySystem
# 导入 Prompt 库
from c1.prompts import DeduplicatorPrompt


@dataclass
class ResolutionAction:
    """LLM 返回的消解动作"""
    action_type: str  # 'add' (新增/保留) 或 'drop' (丢弃/压缩)
    reasoning: str = ""


class SemanticRedundancyFilter:
    """
    【研究内容一(3)：基于逻辑蕴含与向量门控的双层语义压缩器】

    核心科研逻辑：
    该模块负责将高维的 Atom 流压缩进有限的存储空间，同时为第二章的图谱演化预留接口。

    架构设计 (Dual-Layer Architecture):
    -------------------------------------------------------
    Layer 1: 批次内跨视图消解 (Intra-Batch Cross-View Resolution)
    - 目标：解决多粒度分解带来的“视图冗余”。
    - 机制：基于“四视图博弈矩阵” (4-View Game Matrix) 进行动态优胜劣汰。
      不强制 Rule 吃掉 Event，而是基于“信息增益 (Information Gain)”判断。
      如果 Event 提供了 Rule 未涵盖的细节（特例/异常），则共存；否则压缩。

    Layer 2: 全局增量去重 (Global Incremental Deduplication)
    - 目标：解决新信息与存量记忆的冗余。
    - 机制：
      Stage 1 (Vector Gating): 自适应向量门控。利用相似度快速筛选，大幅降低 LLM 调用成本。
      Stage 2 (Logic Judge): 逻辑蕴含判决。仅在相似度模糊区间介入。
    -------------------------------------------------------
    """

    def __init__(self, memory_system: AgenticMemorySystem, llm_model: BaseModel):
        self.memory_sys = memory_system
        self.llm = llm_model

        # --- 超参数：自适应向量门控阈值 ---
        # 对应 PPT 中的“有限算力约束”优化
        self.SIM_THRESHOLD_LOW = 0.5  # 低于此值：语义无关 -> Fast Pass (直接存)
        self.SIM_THRESHOLD_HIGH = 0.95  # 高于此值：物理重复 -> Fast Drop (直接丢)

    def filter_and_add_batch(self, new_atoms: List[DecoupledMemoryAtom]):
        """
        [主入口] 执行双层过滤流程。
        """
        print(f"--- [Filter] 收到 {len(new_atoms)} 条原子，启动双层自适应压缩 ---")

        # === Layer 1: 批次内跨视图消解 ===
        # 仅当输入包含多条原子时才需要做内部PK
        if len(new_atoms) > 1:
            clean_atoms = self._intra_batch_cross_view_compression(new_atoms)
        else:
            clean_atoms = new_atoms

        print(f"--- [Layer 1] 跨视图压缩完成，剩余 {len(clean_atoms)} 条，进入全局处理 ---")

        # === Layer 2: 全局去重 (带向量门控) ===
        for atom in clean_atoms:
            self._process_single_atom_global(atom)

    # =========================================================================
    # Layer 1: 批次内跨视图消解 (Intra-Batch Cross-View)
    # =========================================================================
    def _intra_batch_cross_view_compression(self, atoms: List[DecoupledMemoryAtom]) -> List[DecoupledMemoryAtom]:
        """
        执行 Layer 1 压缩逻辑：基于信息增益的跨视图优胜劣汰。
        """
        # 如果类型单一（全是 Event 或全是 Rule），则不存在跨视图冗余，跳过
        if len(set(a.atom_type for a in atoms)) == 1:
            return atoms

        # 构造 Prompt 输入 (引用 Prompts 库)
        atoms_text = "\n".join([f"ID[{i}] type={atom.atom_type}: {atom.content}" for i, atom in enumerate(atoms)])
        user_content = DeduplicatorPrompt.build_layer1_input(atoms_text)

        messages = [{"role": "user", "content": user_content}]

        try:
            # 调用 LLM
            res_data = self.llm.chat(messages)

            # 容错解析
            if isinstance(res_data, str):
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res_data, re.DOTALL)
                res_data = json.loads(match.group(1)) if match else {}

            # 获取保留列表
            keep_ids = res_data.get("keep_ids", [])

            # 如果 LLM 返回空或解析失败，默认保留所有（防止丢数据）
            if not keep_ids and "keep_ids" not in res_data:
                return atoms

            kept_atoms = []
            for i, atom in enumerate(atoms):
                if i in keep_ids:
                    kept_atoms.append(atom)
                else:
                    # 记录被压缩掉的低维原子
                    print(f"  [Layer 1 Drop] 视图融合丢弃: [{atom.atom_type}] {atom.content[:15]}...")

            return kept_atoms

        except Exception as e:
            print(f"[Layer 1 Error] {e}，跳过压缩")
            return atoms

    # =========================================================================
    # Layer 2: 全局去重 (Global Deduplication)
    # =========================================================================
    def _process_single_atom_global(self, new_atom: DecoupledMemoryAtom):
        """
        执行 Layer 2 逻辑：包含自适应向量门控。
        """
        # 1. 检索：查找语义最相关的旧记忆 (Top-3)
        related_memories = self.memory_sys.find_related_memories(new_atom.content, k=3)

        # 【核心修复】：过滤掉 None 对象
        # 防止数据库与索引不同步导致返回空对象，引发 AttributeError
        related_memories = [m for m in related_memories if m is not None]

        # 如果记忆库为空，或者是全新的概念 -> 直接新增
        if not related_memories:
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # --- Stage 1: 自适应向量门控 (Adaptive Vector Gating) ---
        # 计算新原子与最相似旧记忆的余弦相似度
        emb_model = self.memory_sys.retriever.model
        new_emb = emb_model.encode([new_atom.content])

        # 取第一条（最相关的）有效记忆进行比对
        old_emb = emb_model.encode([related_memories[0].content])
        similarity = cosine_similarity(new_emb, old_emb)[0][0]

        # 1. Fast Pass (快路径-放行)
        if similarity < self.SIM_THRESHOLD_LOW:
            print(f"  [Gate Fast-Pass] Sim={similarity:.2f} (无关信息) -> 直接存储 (无需LLM)")
            self._execute_action(ResolutionAction('add'), new_atom)
            return

        # 2. Fast Drop (快路径-丢弃)
        if similarity > self.SIM_THRESHOLD_HIGH:
            print(f"  [Gate Fast-Drop] Sim={similarity:.2f} (物理重复) -> 直接丢弃 (无需LLM)")
            return

        # 3. Slow Check (慢路径-LLM判决)
        # 处于 [0.5, 0.95] 模糊区间，语义相似但逻辑关系不明（可能是蕴含，也可能是反转）
        print(f"  [Gate Slow-Check] Sim={similarity:.2f} (模糊区间) -> 启用 LLM 深度判决...")
        action = self._judge_redundancy(new_atom, related_memories)
        self._execute_action(action, new_atom)

    def _judge_redundancy(self, new_atom: DecoupledMemoryAtom, old_memories: List) -> ResolutionAction:
        """
        Stage 2: LLM 逻辑判决
        """
        old_mems_text = "\n".join([f"ID[{m.id}]: {m.content}" for m in old_memories])
        user_content = DeduplicatorPrompt.build_layer2_input(old_mems_text, new_atom.content)

        messages = [
            {"role": "system", "content": DeduplicatorPrompt.LAYER2_SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            res = self.llm.chat(messages)
            if isinstance(res, str):
                import re
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res, re.DOTALL)
                res = json.loads(match.group(1)) if match else {"action": "add"}

            return ResolutionAction(res.get('action', 'add'), res.get('reasoning', ''))
        except:
            return ResolutionAction('add')

    def _execute_action(self, action: ResolutionAction, new_atom: DecoupledMemoryAtom):
        """
        执行最终动作：Add 或 Drop
        注意：此处不处理 Update，冲突信息 (Conflict) 也会被 Add，留给第二章做演化。
        """
        if action.action_type == 'drop':
            print(f"  [LLM Drop] 逻辑冗余: {new_atom.content[:15]}... ({action.reasoning})")
        else:
            self.memory_sys.memory_manager.add_memory(new_atom)
            self.memory_sys.retriever.add_documents([new_atom.content])
            print(f"  [+ Save] 存入记忆: {new_atom.content[:15]}...")