# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：pipeline.py
@Desc    ：第一章核心流水线 (Sliding Window Mode)
"""

import sys
import os
import time
from datetime import datetime
from typing import List

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# --- 模块导入 ---
from general.model import ZhipuChat, QwenChat
from general.base_memory import AgenticMemorySystem

# 导入第一章组件
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter

# --- 导入数据加载器 ---
from env.load_locomo import load_locomo_dataset, LoCoMoSample, Turn


class MemoryPipeline:
    def __init__(self, config_path: str, use_local: bool = True):
        print(">>> [Pipeline] 初始化 LoCoMo 记忆处理流水线 (Sliding Window Mode)...")

        # 自动选择模型后端
        if use_local:
            print(f"--> Mode: Local Inference (Qwen2.5 + vLLM)")
            print(f"    Target: http://localhost:8001/v1 (确保 vLLM 已启动)")
            self.llm = QwenChat(config_path)
        else:
            print(f"--> Mode: Cloud API (Zhipu GLM)")
            self.llm = ZhipuChat(config_path)

        self.memory_sys = AgenticMemorySystem()

        # 初始化三大核心组件
        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

    def process_locomo_sample(self, sample: LoCoMoSample):
        """
        处理单个 LoCoMo 样本 (滑动窗口模式)
        """
        print(f"\n{'=' * 80}")
        print(f"开始处理 Sample ID: {sample.sample_id}")
        print(f"参与者: {sample.conversation.speaker_a} & {sample.conversation.speaker_b}")
        print(f"{'=' * 80}")

        sorted_session_ids = sorted(sample.conversation.sessions.keys())

        for s_id in sorted_session_ids:
            session = sample.conversation.sessions[s_id]
            date_str = session.date_time
            timestamp_float = self._parse_locomo_time(date_str)

            print(f"\n>>> 处理 Session {s_id} (Turns: {len(session.turns)})")

            # === 【核心逻辑：滑动窗口】 ===
            # 定义窗口大小 (只看最近 6 轮作为 Context)
            WINDOW_SIZE = 6
            history_buffer = []

            for i, turn in enumerate(session.turns):
                # 1. 构建当前句 (Target)
                current_text = f"[{turn.speaker}]: {turn.text}"

                # 2. 构建上下文 (Context)
                # 取 history_buffer 的最后 WINDOW_SIZE 条
                context_turns = history_buffer[-WINDOW_SIZE:]
                context_text = "\n".join(context_turns) if context_turns else "(No prior history)"

                print(f"\n--- Turn {i + 1}/{len(session.turns)} [{turn.speaker}] ---")

                # 3. 执行单步 Pipeline
                self._run_pipeline_step(
                    current_text=current_text,
                    context_text=context_text,
                    timestamp=timestamp_float
                )

                # 4. 更新历史 buffer
                history_buffer.append(current_text)

        print(f"\n{'#' * 30} Sample {sample.sample_id} 处理完毕 {'#' * 30}")
        self.print_final_memory_state()

    def _parse_locomo_time(self, time_str: str) -> float:
        try:
            dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
            return dt.timestamp()
        except:
            return time.time()

    def _run_pipeline_step(self, current_text: str, context_text: str, timestamp: float):
        """标准的三步处理流程 (针对单句)"""

        # Step 1: Decouple (传入 Target 和 Context)
        raw_obj = RawInputObj(
            text=current_text,  # 重点：当前句
            context=context_text,  # 重点：上下文
            timestamp=timestamp
        )

        dirty_atoms = self.decoupler.decouple(raw_obj)

        if not dirty_atoms:
            # print("  [Info] No memories extracted.")
            return

        # Step 2: Verify (校验)
        # 校验时，我们将 Context + Current 拼起来作为完整的证据链 (Ground Truth)
        full_evidence = f"{context_text}\n{current_text}"

        # 注意：这里传给 verify_batch 的是 full_evidence
        clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)

        if not clean_atoms:
            return

        # Step 3: Deduplicate & Store
        self.deduplicator.filter_and_add_batch(clean_atoms)
        # print(f"  [Success] Stored {len(clean_atoms)} atoms.")

    def print_final_memory_state(self):
        """打印记忆库快照"""
        print("\n=== 最终记忆库状态 ===")
        all_mems = self.memory_sys.memory_manager.get_all_memories()

        if not all_mems:
            print("  (记忆库为空)")
            return

        # 简单的类型统计打印
        mems_by_type = {}
        for m in all_mems:
            m_type = getattr(m, 'atom_type', 'unknown')
            mems_by_type.setdefault(m_type, []).append(m)

        for m_type, mems in mems_by_type.items():
            print(f"\n--- {m_type.upper()} ({len(mems)}) ---")
            for m in mems:
                content = m.content.replace('\n', ' ')
                print(f"  - {content[:100]}...")


if __name__ == "__main__":
    # 1. 路径设置
    config_file = os.path.join(root_dir, "config", "llm_config.yaml")
    dataset_file = os.path.join(root_dir, "dataset", "locomo10.json")

    # 2. 初始化 (开启本地模式)
    try:
        pipeline = MemoryPipeline(config_file, use_local=True)
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    # 3. 加载数据
    try:
        print(f"正在加载数据集: {dataset_file}")
        samples = load_locomo_dataset(dataset_file)

        if samples:
            target_id = "conv-44"  # 您的测试样本ID
            target_sample = next((s for s in samples if s.sample_id == target_id), None)

            if target_sample:
                pipeline.process_locomo_sample(target_sample)
            else:
                print(f"[Error] Sample {target_id} not found.")

    except Exception as e:
        print(f"Error: {e}")