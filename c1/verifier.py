# c1/verifier.py
import json
import re
from typing import List

from general.decoupled_memory import DecoupledMemoryAtom
from general.model import BaseModel
from c1.prompts import VerifierPrompt


class ConsistencyVerifier:
    """
    【研究内容一(2)：自监督反事实一致性校验器】
    (Self-Supervised Counterfactual Consistency Verifier)

    核心科研逻辑：
    1. 痛点解决：解决大模型记忆提取中的“事实幻觉 (Hallucination)”。
    2. 算法机制：构建“生成式验证探针 (Generative Probing)”。
       - Step 1: 把提取出的原子转化为问题。
       - Step 2: 仅利用原始上下文重新回答。
       - Step 3: 比对一致性。
    """

    def __init__(self, llm_model: BaseModel):
        self.llm = llm_model
        self.json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def verify_batch(self, atoms: List[DecoupledMemoryAtom], raw_text: str) -> List[DecoupledMemoryAtom]:
        """
        批量执行校验流程。
        Args:
            atoms: 脏原子列表
            raw_text: 原始 Ground Truth (History + Current Turn)
        Returns:
            Clean Atoms
        """
        if not atoms: return []

        # print(f"--- [Verifier] Checking {len(atoms)} atoms ---")

        # 1. 构造 Prompt (序列化原子列表)
        atoms_text = "\n".join([f"{i + 1}. [{atom.atom_type}] {atom.content}" for i, atom in enumerate(atoms)])
        user_content = VerifierPrompt.build_input(raw_text, atoms_text)

        messages = [
            {"role": "system", "content": VerifierPrompt.SYSTEM},
            {"role": "user", "content": user_content}
        ]

        try:
            # 2. 调用 LLM
            res_data = self.llm.chat(messages)

            # 解析兼容
            if isinstance(res_data, str):
                match = self.json_pattern.search(res_data)
                res_data = json.loads(match.group(1)) if match else {}

            results = res_data.get("verification_results", [])
            res_map = {r["index"]: r for r in results}
            verified_atoms = []

            # 3. 遍历并过滤
            for i, atom in enumerate(atoms):
                res = res_map.get(i + 1)

                # --- 元数据记录 (Meta Logging) ---
                if res:
                    if not hasattr(atom, "meta_data") or atom.meta_data is None:
                        atom.meta_data = {}
                    atom.meta_data['verification_log'] = {
                        "is_consistent": res.get("is_consistent"),
                        "reasoning": res.get("reasoning")
                    }

                # --- 剪枝逻辑 ---
                if not res or res.get("is_consistent"):
                    # 验证通过或漏检(Pass/Skip) -> 保留
                    if res: atom.confidence = min(1.0, atom.confidence + 0.1)
                    verified_atoms.append(atom)
                    # print(f"  [√ Pass] {atom.content}")
                else:
                    # 验证失败 -> 丢弃
                    # print(f"  [x Fail] {atom.content} (Reason: {res.get('reasoning')})")
                    pass

            return verified_atoms

        except Exception as e:
            # print(f"[Verifier Error] {e}")
            return atoms