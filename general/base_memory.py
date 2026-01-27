# -*- coding: UTF-8 -*-
"""
@Project ：graduate
@File    ：base_memory.py
@Desc    ：【Debug 版】底层存储 + 探针
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DEFAULT_MODEL_PATH = os.path.join(project_root, "model", "all-MiniLM-L6-v2")


class MemoryNote(ABC):
    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 valid: Optional[bool] = True,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = 0,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 atom_type: str = "general"
                 ):
        # ⚠️ 注意：每次 new 的时候 uuid.uuid4() 保证 ID 唯一
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.valid = valid
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_accessed = last_accessed or self.timestamp
        self.atom_type = atom_type

    def update_validity(self, valid: bool): self.valid = valid

    def update_last_accessed(self): self.last_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def increment_retrieval_count(self): self.retrieval_count += 1

    def set_importance_score(self, importance_score: float): self.importance_score = importance_score

    def get_summary(self) -> str: return f"ID: {self.id}, Type: {self.atom_type}, Valid: {self.valid}"

    def __str__(self) -> str: return f"[{self.atom_type.upper()}] MemoryNote({self.id}): {self.content}"

    def __repr__(self) -> str: return f"MemoryNote(Type={self.atom_type}, ID={self.id})"

    @abstractmethod
    def extra_info(self) -> str: pass


class ContentBasedMemoryNote(MemoryNote):
    def __init__(self, content: str, *args, **kwargs):
        super().__init__(content, *args, **kwargs)

    def extra_info(self) -> str: return f"Content-based additional info: {self.content[:20]}..."


class MemoryManager:
    def __init__(self):
        self.memory_store = {}

    def add_memory(self, memory: MemoryNote) -> str:
        # 存入字典
        self.memory_store[memory.id] = memory

        # 🔥 Debug 探针：查看当前到底存了多少个
        # print(f"      [BaseDB] Stored ID={memory.id} | Store Size={len(self.memory_store)}")
        # print(f"      [BaseDB Keys] {list(self.memory_store.keys())}")

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
            if new_content: memory.content = new_content
            return True
        return False

    def clear(self):
        self.memory_store = {}


class HybridRetriever:
    def __init__(self, model_name: Optional[str] = None, alpha: float = 0.5):
        target_model = model_name or DEFAULT_MODEL_PATH
        if os.path.exists(target_model):
            # print(f"[HybridRetriever] Loading local model from: {target_model}")
            self.model = SentenceTransformer(target_model)
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.embeddings = None

    def add_documents(self, documents: List[str], ids: List[str] = None) -> bool:
        if not documents: return False
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        self.doc_ids = ids if ids and len(ids) == len(documents) else []
        return True

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus: return []
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query)) if self.bm25 else np.zeros(len(self.corpus))
        if len(bm25_scores) > 0 and (bm25_scores.max() - bm25_scores.min() > 0):
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[
            0] if self.embeddings is not None else np.zeros(len(self.corpus))

        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        k = min(k, len(self.corpus))
        return np.argsort(hybrid_scores)[-k:][::-1].tolist()

    def clear(self):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.embeddings = None


class AgenticMemorySystem:
    def __init__(self, model_name: Optional[str] = None, alpha: float = 0.5):
        self.memory_manager = MemoryManager()
        self.retriever = HybridRetriever(model_name=model_name, alpha=alpha)

    def add_note(self, content: str, **kwargs):
        note = ContentBasedMemoryNote(content=content, **kwargs)
        self.memory_manager.add_memory(note)
        # 每次添加都刷新索引，确保能被检索到
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