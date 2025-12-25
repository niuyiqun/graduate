# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：pipeline.py
@Author  ：niu
@Date    ：2025/12/25 14:20 
@Desc    ：
"""

# c1/pipeline.py
import sys
import os
import time
from typing import List

# --- 路径适配 (确保能导入根目录模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# --- 模块导入 ---
from general.model import ZhipuChat
from general.decoupled_memory import DecoupledMemoryAtom
from general.base_memory import AgenticMemorySystem

# 导入第一章的三大核心组件
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter


class MemoryPipeline:
    """
    【第一章总控流水线】
    串联：正交分解 -> 反事实验证 -> 双层语义压缩 -> 存储
    """

    def __init__(self, config_path: str):
        print(">>> [Pipeline] 正在初始化各个组件...")

        # 1. 基础组件
        self.llm = ZhipuChat(config_path)
        self.memory_sys = AgenticMemorySystem()  # 内存管理器 + 检索器

        # 2. 研究内容一的核心模块
        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

        print(">>> [Pipeline] 初始化完成！\n")

    def process_session_stream(self, session_text: str, session_time: str = None):
        """
        处理一段对话流 (Stream Processing)
        """
        print("=" * 60)
        print(f"【Input Stream】:\n{session_text.strip()[:100]}...")
        print("=" * 60)

        # --- Step 1: 多粒度语义正交分解 (Decoupling) ---
        print("\n=== Step 1: 正交分解 (Decoupler) ===")
        raw_obj = RawInputObj(text=session_text, timestamp=time.time())
        dirty_atoms = self.decoupler.decouple(raw_obj)

        if not dirty_atoms:
            print("  [Warn] 未提取到任何有效信息。")
            return

        print(f"  -> 初步提取 {len(dirty_atoms)} 条原子:")
        for a in dirty_atoms:
            print(f"    - [{a.atom_type}] {a.content}")

        # --- Step 2: 自监督反事实校验 (Verification) ---
        print("\n=== Step 2: 反事实校验 (Verifier) ===")
        # 这一步是为了去幻觉 (Hallucination Removal)
        clean_atoms = self.verifier.verify_batch(dirty_atoms, session_text)

        if not clean_atoms:
            print("  [Info] 所有原子均未通过校验（被视为幻觉/噪声）。")
            return

        print(f"  -> 校验通过 {len(clean_atoms)} 条有效原子。")

        # --- Step 3: 双层语义压缩与存储 (Deduplication & Storage) ---
        print("\n=== Step 3: 双层压缩与存储 (Filter & Store) ===")
        # Layer 1 (Cross-View) + Layer 2 (Global Vector Gating)
        self.deduplicator.filter_and_add_batch(clean_atoms)

    def print_final_memory_state(self):
        """打印当前记忆库状态，验证压缩效果"""
        print("\n" + "#" * 30 + " 最终记忆库状态 " + "#" * 30)
        all_mems = self.memory_sys.memory_manager.get_all_memories()

        if not all_mems:
            print("(记忆库为空)")
            return

        # 按类型分组打印
        sorted_mems = sorted(all_mems, key=lambda x: x.atom_type)
        current_type = ""
        for mem in sorted_mems:
            if mem.atom_type != current_type:
                print(f"\n--- {mem.atom_type.upper()} ---")
                current_type = mem.atom_type
            print(f"ID[{mem.id}]: {mem.content}")
        print("#" * 76 + "\n")


# =============================================================================
# 运行示例 (Main)
# =============================================================================
if __name__ == "__main__":
    # 1. 配置文件路径
    config_path = os.path.join(root_dir, "config", "llm_config.yaml")

    # 2. 实例化流水线
    pipeline = MemoryPipeline(config_path)

    # 3. 构造模拟对话数据 (LoCoMo 风格，3-5 轮)
    # 包含：Rule与Event冗余、自我修正、新知识
    dialogue_turn_1 = """
[User]: 我最近刚搬到杭州，准备开始新的生活。
[Assistant]: 欢迎来到杭州！
[User]: 谢谢。我有个习惯，每天早上必须喝一杯拿铁，不然没精神。
[User]: 哦对了，昨天我去找房子，结果那个中介太坑了，带我看了好几个破房子。
    """

    dialogue_turn_2 = """
[User]: 今天早上我又喝了一杯拿铁，感觉好多了。
[User]: 不过我发现杭州的菜稍微有点甜，我其实不太能吃甜的，我喜欢吃辣。
    """

    # 4. 执行流程

    # --- 第 1 轮处理 ---
    # 预期：提取"搬到杭州"(Event/Know), "每天喝拿铁"(Rule), 昨天看房(Event)
    pipeline.process_session_stream(dialogue_turn_1)

    # --- 第 2 轮处理 ---
    # 预期：
    # 1. "今天早上喝拿铁" -> 应该被 "每天喝拿铁"(Rule) 在跨视图层吃掉 (Layer 1)。
    # 2. "不吃甜/喜欢辣" -> 新 Rule，存入。
    pipeline.process_session_stream(dialogue_turn_2)

    # 5. 打印最终结果，验收 "高密度表征"
    pipeline.print_final_memory_state()
