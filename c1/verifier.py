# c1/verifier.py
import json
from typing import List

# 导入通用数据结构和模型基类
from general.decoupled_memory import DecoupledMemoryAtom
from general.model import BaseModel
# 导入隔离后的提示词库
from c1.prompts import VerifierPrompt


class ConsistencyVerifier:
    """
    【研究内容一(2)：自监督反事实一致性校验器】
    (Self-Supervised Counterfactual Consistency Verifier)

    核心科研逻辑：
    1. 痛点解决：解决大模型记忆提取中的“事实幻觉 (Hallucination)”和“过度推断”。
    2. 算法机制：构建“生成式验证探针 (Generative Probing)”。
       - Step 1: 把提取出的原子 (Atom) 反向转化为问题 (Probe)。
       - Step 2: 仅利用原始上下文 (Raw Text) 重新回答。
       - Step 3: 比对“提取值”与“回溯值”的一致性。
    3. 价值：实现“基于真值过滤的语义剪枝”，保证存储的高信噪比。
    """

    def __init__(self, llm_model: BaseModel):
        self.llm = llm_model

    def verify_batch(self, atoms: List[DecoupledMemoryAtom], raw_text: str) -> List[DecoupledMemoryAtom]:
        """
        批量执行校验流程。

        Args:
            atoms: 初步分解得到的“脏”原子列表 (Dirty Atoms)
            raw_text: 原始用户输入，作为唯一的真值来源 (Ground Truth)

        Returns:
            经过反事实校验后的“净”原子列表 (Clean Atoms)
        """
        if not atoms:
            return []

        print(f"--- [Verifier] 启动反事实校验，当前候选原子数: {len(atoms)} ---")

        # 1. 构造 Prompt 输入
        # 将原子列表序列化为带序号的文本，方便 LLM 逐条检查
        atoms_text = "\n".join([f"{i + 1}. [{atom.atom_type}] {atom.content}" for i, atom in enumerate(atoms)])

        # 使用 Prompts 类构建完整的 Prompt (包含 System Prompt 和 User Input)
        user_content = VerifierPrompt.build_input(raw_text, atoms_text)

        messages = [
            {"role": "system", "content": VerifierPrompt.SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            # 2. 调用大模型 (LLM) 执行校验
            # 注意：这里的 chat 方法内部应包含正则解析，返回 dict
            res_data = self.llm.chat(messages)

            # 2.1 容错处理：如果返回的是字符串（未被 model.py 自动解析），手动正则提取 JSON
            if isinstance(res_data, str):
                import re
                # 使用非贪婪匹配提取 ```json ... ``` 内容
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res_data, re.DOTALL)
                res_data = json.loads(match.group(1)) if match else {}

            # 获取校验结果列表
            results = res_data.get("verification_results", [])
            # 建立 index -> result 的映射，方便查找
            res_map = {r["index"]: r for r in results}

            verified_atoms = []

            # 3. 遍历原有的原子，根据校验结果决定“留”还是“删”
            for i, atom in enumerate(atoms):
                # 提示词里的序号是从 1 开始的
                res = res_map.get(i + 1)

                # --- 日志记录 (Meta Data Logging) ---
                # 将 LLM 的推理过程写入元数据，这对于论文写 Case Study 非常重要
                if res:
                    if not hasattr(atom, "meta_data") or atom.meta_data is None:
                        atom.meta_data = {}

                    atom.meta_data['verification_log'] = {
                        "is_consistent": res.get("is_consistent"),  # 校验结论
                        "probe_question": res.get("probe_question"),  # 生成的探针问题
                        "reasoning": res.get("reasoning")  # 判定理由
                    }

                # --- 核心剪枝逻辑 (Pruning Logic) ---
                if res and res.get("is_consistent"):
                    # Case A: 验证通过
                    # 说明该信息在原文中有确凿证据 -> 保留并提升置信度
                    atom.confidence = min(1.0, atom.confidence + 0.1)
                    verified_atoms.append(atom)
                    print(f"  [√ Pass] {atom.content}")

                elif not res:
                    # Case B: LLM 漏检 (未返回结果)
                    # 策略：保守起见，默认放行，但可以考虑稍微降权
                    verified_atoms.append(atom)
                    # print(f"  [? Skip] {atom.content} (未被检测)")

                else:
                    # Case C: 验证失败 (幻觉/过度推断)
                    # 策略：直接剪枝 (Drop)。这是实现“语义压缩”的关键一步。
                    print(f"  [x Fail] {atom.content}")
                    print(f"     └-> 理由: {res.get('reasoning')}")

            return verified_atoms

        except Exception as e:
            print(f"[Verifier Error] 校验过程发生异常: {e}")
            # 系统健壮性原则：如果校验器挂了，不要让整个系统崩溃，返回原始列表
            return atoms