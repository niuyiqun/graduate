# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：base_memory.py
@Author  ：niu
@Date    ：2025/12/4 15:43
@Desc    ：
"""
import pickle
import uuid
import os
from abc import abstractmethod, ABC
from typing import Optional, List
from datetime import datetime
from rank_bm25 import BM25Okapi

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- 1. 动态获取项目根目录 ---
# 获取 current file dir: .../graduate/general
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project root: .../graduate
project_root = os.path.dirname(current_dir)
# 拼接本地模型默认绝对路径
DEFAULT_MODEL_PATH = os.path.join(project_root, "model", "all-MiniLM-L6-v2")


class MemoryNote(ABC):
    """基本记忆块类，可扩展"""

    def __init__(self,
                 content: str,  # 压缩后内容
                 id: Optional[str] = None,  # 块标识
                 valid: Optional[bool] = True,  # 块是否有效
                 importance_score: Optional[float] = None,  # 优势值
                 retrieval_count: Optional[int] = 0,  # 该记忆块被检索的次数
                 timestamp: Optional[str] = None,  # 记忆块的时间戳
                 last_accessed: Optional[str] = None,  # 记录最后一次访问该记忆块的时间
                 # 🔥【核心新增】支持存储记忆类型 (Profile/Knowledge/Activity/Thought)
                 atom_type: str = "general"
                 ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.valid = valid
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_accessed = last_accessed or self.timestamp
        # 保存类型
        self.atom_type = atom_type

    def update_validity(self, valid: bool):
        self.valid = valid

    def update_last_accessed(self):
        self.last_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def increment_retrieval_count(self):
        self.retrieval_count += 1

    def set_importance_score(self, importance_score: float):
        self.importance_score = importance_score

    def get_summary(self) -> str:
        return f"ID: {self.id}, Type: {self.atom_type}, Valid: {self.valid}"

    def __str__(self) -> str:
        # 🔥 打印时带上类型，方便调试
        return f"[{self.atom_type.upper()}] MemoryNote({self.id}): {self.content}\nValid: {self.valid}"

    def __repr__(self) -> str:
        return f"MemoryNote(Type={self.atom_type}, ID={self.id})"

    @abstractmethod
    def extra_info(self) -> str:
        pass


class ContentBasedMemoryNote(MemoryNote):
    """研究内容1中专注于内容的记忆块"""

    def __init__(self, content: str, *args, **kwargs):
        super().__init__(content, *args, **kwargs)

    def extra_info(self) -> str:
        return f"Content-based additional info: {self.content[:20]}..."


class MemoryManager:
    """Memory management system to store and retrieve MemoryNote objects"""

    def __init__(self):
        self.memory_store = {}

    def add_memory(self, memory: MemoryNote) -> str:
        self.memory_store[memory.id] = memory
        # 简化打印，防止刷屏太严重
        # print(f"    [Manager] Added {memory.atom_type}: {memory.content[:30]}...")
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[MemoryNote]:
        return self.memory_store.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self.memory_store:
            del self.memory_store[memory_id]
            return True
        return False

    def get_all_memories(self) -> List[MemoryNote]:
        return list(self.memory_store.values())

    def update_memory(self, memory_id: str, new_content: Optional[str] = None) -> bool:
        memory = self.memory_store.get(memory_id)
        if memory:
            if new_content:
                memory.content = new_content
            return True
        return False

    def clear(self):
        self.memory_store = {}


class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""

    def __init__(self, model_name: Optional[str] = None, alpha: float = 0.5):
        target_model = model_name or DEFAULT_MODEL_PATH
        if os.path.exists(target_model):
            print(f"[HybridRetriever] Loading local model from: {target_model}")
            self.model = SentenceTransformer(target_model)
        else:
            print(f"[Warn] Local path not found: {target_model}")
            # 自动下载模型作为 fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.embeddings = None
        self.document_ids = {}

    def add_documents(self, documents: List[str], ids: List[str] = None) -> bool:
        if not documents: return False

        # BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Vector
        self.embeddings = self.model.encode(documents)
        self.corpus = documents

        if ids and len(ids) == len(documents):
            self.doc_ids = ids
        else:
            self.doc_ids = []

        return True

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus: return []

        tokenized_query = query.lower().split()
        if self.bm25:
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        else:
            bm25_scores = np.zeros(len(self.corpus))

        # Normalize BM25
        if len(bm25_scores) > 0 and (bm25_scores.max() - bm25_scores.min() > 0):
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        # Semantic
        query_embedding = self.model.encode([query])[0]
        if self.embeddings is not None and len(self.embeddings) > 0:
            semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        else:
            semantic_scores = np.zeros(len(self.corpus))

        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

    def clear(self):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.embeddings = None


class AgenticMemorySystem:
    """Memory system with management and retrieval capabilities."""

    def __init__(self, model_name: Optional[str] = None, alpha: float = 0.5):
        self.memory_manager = MemoryManager()
        self.retriever = HybridRetriever(model_name=model_name, alpha=alpha)

    def add_note(self, content: str, **kwargs):
        """
        Add a new note.
        **kwargs allows passing 'atom_type' down to MemoryNote
        """
        # 这里会把 atom_type='...' 传给 ContentBasedMemoryNote -> MemoryNote
        note = ContentBasedMemoryNote(content=content, **kwargs)
        self.memory_manager.add_memory(note)
        self.consolidate_memories()
        return note.id

    def find_related_memories(self, query: str, k: int = 5):
        indices = self.retriever.retrieve(query, k)
        if self.retriever.doc_ids:
            related_memories = []
            for i in indices:
                mem_id = self.retriever.doc_ids[i]
                mem = self.memory_manager.get_memory(mem_id)
                if mem: related_memories.append(mem)
            return related_memories
        return []

    def consolidate_memories(self):
        all_memories = self.memory_manager.get_all_memories()
        if not all_memories:
            self.retriever.clear()
            return
        all_contents = [memory.content for memory in all_memories]
        all_ids = [memory.id for memory in all_memories]
        self.retriever.add_documents(all_contents, ids=all_ids)

    def clear(self):
        self.memory_manager.clear()
        self.retriever.clear()


if __name__ == '__main__':
    print("Testing Memory System...")
    memory_system = AgenticMemorySystem()
    note_id = memory_system.add_note("This is a general about neural networks.")

    print("\n--- Testing Retrieval ---")
    query = "What are neural networks?"
    related_memories = memory_system.find_related_memories(query)
    for memory in related_memories:
        print(f"Found: {memory.content}")

    print("\n--- Testing Clear ---")
    memory_system.clear()
    related_memories = memory_system.find_related_memories(query)
    if not related_memories:
        print("Clear successful! No memories found.")