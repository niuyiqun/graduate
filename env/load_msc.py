# -*- coding: UTF-8 -*-
"""
@Project ：graduate 
@File    ：load_msc.py
@Author  ：niu
@Date    ：2025/12/3 15:11
@Desc    ：
"""
import json
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    # category 字段的不同类别代表不同类型的问题：
    # 1: 单跳推理（Single-hop Reasoning） - 问题的答案可以直接从单一对话回合中提取。
    # 2: 多跳推理（Multi-hop Reasoning） - 问题的答案需要跨多个对话回合或会话推理。
    # 3: 时间推理（Temporal Reasoning） - 问题涉及时间的推理，需要根据时间顺序推理事件。
    # 4: 事件推理（Event Reasoning） - 问题需要根据对话中的事件进行推理。
    # 5: 对抗性问题（Adversarial Reasoning） - 问题设计得具有挑战性，用于测试模型的鲁棒性和推理能力。

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events


@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]


@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session

    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []

    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data (保持原有逻辑不变)
            qa_list = []
            for qa in sample.get("qa", []):
                # ... (中间的 QA 解析逻辑保持不变) ...
                qa_obj = QA(
                    question=qa["question"],
                    answer=qa.get("answer"),
                    evidence=qa.get("evidence", []),
                    category=qa.get("category"),
                    adversarial_answer=qa.get("adversarial_answer")
                )
                qa_list.append(qa_obj)

            # Parse conversation
            conversation = parse_conversation(sample["conversation"])

            # Parse event summary
            event_summary = EventSummary(events=sample.get("event_summary", {}))

            # Parse observation
            observation = Observation(observations=sample.get("observation", {}))

            # Get session summary
            session_summary = sample.get("session_summary", {})

            # ---【核心修改点】---
            # 读取真实的 sample_id，如果读不到才用索引兜底
            real_id = sample.get("sample_id", str(sample_idx))

            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=real_id,  # 使用真实 ID
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)

            # Print statistics for this sample
            # print(f"Loaded Sample ID: {real_id}")

        except Exception as e:
            print(f"Error processing sample index {sample_idx}: {str(e)}")
            continue  # 跳过错误样本，继续加载下一个

    print(f"\n[Success] 成功加载 {len(samples)} 个样本。")
    # 打印前几个 ID 供检查
    ids = [s.sample_id for s in samples[:5]]
    print(f"前5个样本 ID: {ids}")

    return samples


def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.

    Args:
        samples: List of LoCoMoSample objects

    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats


if __name__ == "__main__":
    # Example usage
    dataset_path = Path(__file__).parent.parent / "dataset" / "locomo10.json"
    try:
        print(f"Loading dataset from: {dataset_path}")
        samples = load_locomo_dataset(dataset_path)
        for sample_idx, sample in enumerate(samples):
            print(f"\nSample {sample_idx}:")
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    print(turn)
                    break
                    # stats = get_dataset_statistics(samples)
        # print("\nDataset Statistics (Text-only content):")
        # for key, value in stats.items():
        #     print(f"{key}: {value}")
        # print(len(samples))
        # for sample in samples:
        #     print(sample)
        #     break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise