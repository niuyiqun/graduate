# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：pipeline.py
@Author  ：niu
@Date    ：2025/12/25 14:20 
@Desc    ：
"""

# -*- coding: UTF-8 -*-
# c1/pipeline.py

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
from general.model import ZhipuChat
from general.base_memory import AgenticMemorySystem

# 导入第一章组件
from c1.decoupler import SemanticDecoupler, RawInputObj
from c1.verifier import ConsistencyVerifier
from c1.deduplicator import SemanticRedundancyFilter

# --- 导入您的数据加载器 ---
from env.load_locomo import load_locomo_dataset, LoCoMoSample, Turn


class MemoryPipeline:
    def __init__(self, config_path: str):
        print(">>> [Pipeline] 初始化 LoCoMo 记忆处理流水线...")
        self.llm = ZhipuChat(config_path)
        self.memory_sys = AgenticMemorySystem()

        self.decoupler = SemanticDecoupler(self.llm)
        self.verifier = ConsistencyVerifier(self.llm)
        self.deduplicator = SemanticRedundancyFilter(self.memory_sys, self.llm)

    def process_locomo_sample(self, sample: LoCoMoSample):
        """
        处理单个 LoCoMo 样本（包含多个 Session）
        """
        print(f"\n{'=' * 80}")
        print(f"开始处理 Sample ID: {sample.sample_id}")
        print(f"参与者: {sample.conversation.speaker_a} & {sample.conversation.speaker_b}")
        print(f"{'=' * 80}")

        sorted_session_ids = sorted(sample.conversation.sessions.keys())

        for s_id in sorted_session_ids:
            session = sample.conversation.sessions[s_id]
            date_str = session.date_time

            # 【修改点】先解析时间为 float，再传入
            timestamp_float = self._parse_locomo_time(date_str)

            print(f"\n>>> 处理 Session {s_id} (Time: {date_str})")

            dialogue_text = self._format_turns(session.turns)
            print(f"--- 对话片段 ({len(session.turns)} turns) ---")
            print(dialogue_text[:200] + "..." if len(dialogue_text) > 200 else dialogue_text)

            # 【修改点】传入 float 类型的时间戳
            self._run_pipeline_step(dialogue_text, timestamp=timestamp_float)

        print(f"\n{'#' * 30} Sample {sample.sample_id} 处理完毕 {'#' * 30}")
        self.print_final_memory_state()

        # 注意：实际实验中，处理下一个 Sample 前应该清空记忆库
        # self.memory_sys.clear() # 如果 BaseMemory 有这个方法的话

        # ---【新增】辅助函数：解析 LoCoMo 时间格式 ---

    def _parse_locomo_time(self, time_str: str) -> float:
        """
        将字符串 "1:14 pm on 25 May, 2023" 转换为 float timestamp
        """
        try:
            # LoCoMo 时间格式匹配
            dt = datetime.strptime(time_str, "%I:%M %p on %d %B, %Y")
            return dt.timestamp()
        except ValueError:
            print(f"  [Warn] 时间格式解析失败: '{time_str}'，使用当前系统时间。")
            return time.time()
        except Exception as e:
            print(f"  [Warn] 时间解析未知错误: {e}")
            return time.time()

    def _format_turns(self, turns: List[Turn]) -> str:
        """将 Turn 对象列表转为 LLM 易读的字符串"""
        lines = []
        for turn in turns:
            # 格式: [Caroline]: I want to help people...
            lines.append(f"[{turn.speaker}]: {turn.text}")
        return "\n".join(lines)

    def _run_pipeline_step(self, text: str, timestamp: str):
        """标准的三步处理流程"""
        # Step 1: Decouple
        raw_obj = RawInputObj(text=text, timestamp=timestamp)
        dirty_atoms = self.decoupler.decouple(raw_obj)
        if not dirty_atoms:
            print("  [Warn] 无有效信息提取。")
            return

        # Step 2: Verify (传入原始对话作为 Ground Truth)
        clean_atoms = self.verifier.verify_batch(dirty_atoms, text)
        if not clean_atoms:
            return

        # Step 3: Deduplicate & Store
        self.deduplicator.filter_and_add_batch(clean_atoms)
        print(f"  [Success] 入库完成。")

    def print_final_memory_state(self):
        """打印记忆库快照"""
        print("\n=== 最终记忆库状态 ===")
        all_mems = self.memory_sys.memory_manager.get_all_memories()

        # 按 Attribute (Rule) 和 Event 分组打印，方便检查归因
        mems_by_type = {}
        for m in all_mems:
            mems_by_type.setdefault(m.atom_type, []).append(m)

        for m_type, mems in mems_by_type.items():
            print(f"\n--- {m_type.upper()} ({len(mems)}) ---")
            for m in mems:
                print(f"  - {m.content}")


if __name__ == "__main__":
    # 1. 路径设置
    config_file = os.path.join(root_dir, "config", "llm_config.yaml")
    # 假设 locomo10.json 在项目根目录的 dataset 文件夹下
    dataset_file = os.path.join(root_dir, "dataset", "locomo10.json")

    # 2. 初始化
    pipeline = MemoryPipeline(config_file)

    # 3. 加载数据
    try:
        print(f"正在加载数据集: {dataset_file}")
        samples = load_locomo_dataset(dataset_file)

        if samples:
            # 【修改点】指定一个安全的 Sample ID (conv-44: 聊狗和工作; conv-50: 聊车和旅游)
            # 之前的 Caroline 是 conv-26，包含敏感词会导致报错
            target_id = "conv-44"

            # 精确查找该 ID 的样本
            target_sample = next((s for s in samples if s.sample_id == target_id), None)

            if target_sample:
                print(f"\n>>> 选中测试样本: {target_sample.sample_id}")
                pipeline.process_locomo_sample(target_sample)
            else:
                print(f"[Error] 未找到 ID 为 {target_id} 的样本，请检查 ID 是否正确。")
                # 打印所有可用 ID 供调试
                all_ids = [s.sample_id for s in samples]
                print(f"可用 ID 列表: {all_ids}")

    except FileNotFoundError:
        print(f"错误：找不到数据集文件 {dataset_file}")
    except Exception as e:
        print(f"运行时错误: {e}")
