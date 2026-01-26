# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：pipeline.py
@Desc    ：第一章核心流水线 (Sliding Window Mode) - GRPO 适配版
"""

import sys
import os
import time
from datetime import datetime
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# --- 导入 QwenGRPOChat ---
from general.model import ZhipuChat, QwenChat, QwenGRPOChat
from general.base_memory import AgenticMemorySystem

from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter
from env.load_locomo import load_locomo_dataset, LoCoMoSample, Turn


class MemoryPipeline:
    def __init__(self, config_path: str, use_local: bool = True):
        print(">>> [Pipeline] 初始化 LoCoMo 记忆处理流水线 (GRPO Mode)...")

        if use_local:
            print(f"--> Mode: Local Inference (Qwen2.5-GRPO via vLLM)")
            # 关键：使用微调后的 GRPO 模型
            self.llm = QwenGRPOChat(config_path)
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
        print(f"\n{'=' * 40}")
        print(f"Processing Sample ID: {sample.sample_id}")
        print(f"{'=' * 40}")

        sorted_session_ids = sorted(sample.conversation.sessions.keys())

        for s_id in sorted_session_ids:
            session = sample.conversation.sessions[s_id]
            timestamp_float = self._parse_locomo_time(session.date_time)

            print(f"Processing Session {s_id} ({len(session.turns)} turns)...")

            # === 【核心逻辑：滑动窗口】 ===
            WINDOW_SIZE = 6
            history_buffer = []

            for i, turn in enumerate(session.turns):
                # 1. 构建当前句 (Target)
                current_text = f"[{turn.speaker}]: {turn.text}"

                # 2. 构建上下文 (Context)
                context_turns = history_buffer[-WINDOW_SIZE:]
                context_text = "\n".join(context_turns) if context_turns else "(No prior history)"

                # 3. 执行单步 Pipeline
                self._run_pipeline_step(
                    current_text=current_text,
                    context_text=context_text,
                    timestamp=timestamp_float
                )

                # 4. 更新历史 buffer
                history_buffer.append(current_text)

        print(f"Sample {sample.sample_id} Done. Final Memories: {len(self.memory_sys.memory_manager.get_all_memories())}")
        # self.print_final_memory_state() # 想看结果可以解开这个注释

    def _parse_locomo_time(self, time_str: str) -> float:
        try:
            dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
            return dt.timestamp()
        except:
            return time.time()

    def _run_pipeline_step(self, current_text: str, context_text: str, timestamp: float):
        """标准的三步处理流程 (针对单句)"""

        # Step 1: Decouple (提取)
        raw_obj = RawInputObj(text=current_text, context=context_text, timestamp=timestamp)
        dirty_atoms = self.decoupler.decouple(raw_obj)

        if not dirty_atoms: return

        # Step 2: Verify (校验)
        full_evidence = f"{context_text}\n{current_text}"
        clean_atoms = self.verifier.verify_batch(dirty_atoms, full_evidence)

        if not clean_atoms: return

        # Step 3: Deduplicate & Store (去重并存储)
        self.deduplicator.filter_and_add_batch(clean_atoms)

    def print_final_memory_state(self):
        """打印记忆库快照"""
        print("\n=== 最终记忆库状态 ===")
        all_mems = self.memory_sys.memory_manager.get_all_memories()
        # ... (后续代码省略，仅用于打印)


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