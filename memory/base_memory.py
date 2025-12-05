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
from abc import abstractmethod, ABC
from typing import Optional, List
from datetime import datetime
from rank_bm25 import BM25Okapi

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class MemoryNote(ABC):
    """基本记忆块类，可扩展"""

    def __init__(self,
                 content: str,  # 压缩后内容
                 id: Optional[str] = None,  # 块标识
                 valid: Optional[bool] = True,  # 块是否有效
                 importance_score: Optional[float] = None,  # 优势值
                 retrieval_count: Optional[int] = 0,  # 该记忆块被检索的次数，反映其被访问的频率
                 timestamp: Optional[str] = None,  # 记忆块的时间戳
                 last_accessed: Optional[str] = None,  # 记录最后一次访问该记忆块的时间
                 ):
        # 初始化记忆块的字段
        self.id = id or str(uuid.uuid4())  # 如果没有提供 ID，则自动生成一个 UUID
        self.content = content
        self.valid = valid
        self.importance_score = importance_score or 1.0  # 如果没有提供，默认重要性为 1.0
        self.retrieval_count = retrieval_count  # 初始化检索次数
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 默认时间戳为当前时间
        self.last_accessed = last_accessed or self.timestamp  # 如果没有提供最后访问时间，默认为时间戳

    def update_validity(self, valid: bool):
        """更新记忆块的有效性"""
        self.valid = valid

    def update_last_accessed(self):
        """更新记忆块的最后访问时间为当前时间"""
        self.last_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def increment_retrieval_count(self):
        """增加记忆块的检索次数"""
        self.retrieval_count += 1

    def set_importance_score(self, importance_score: float):
        """设置记忆块的优势值"""
        self.importance_score = importance_score

    def get_summary(self) -> str:
        """获取记忆块的简要信息"""
        return f"ID: {self.id}, Valid: {self.valid}, Importance: {self.importance_score}, Retrieval Count: {self.retrieval_count}, Last Accessed: {self.last_accessed}"

    def __str__(self) -> str:
        """返回记忆块的详细信息"""
        return f"MemoryNote({self.id}): {self.content}\nValid: {self.valid}\nImportance Score: {self.importance_score}\nRetrieval Count: {self.retrieval_count}\nTimestamp: {self.timestamp}\nLast Accessed: {self.last_accessed}"

    def __repr__(self) -> str:
        """返回记忆块的简洁表示，用于调试"""
        return f"MemoryNote({self.id}, Importance: {self.importance_score})"

    @abstractmethod
    def extra_info(self) -> str:
        """子类可扩展的额外信息"""
        pass


class ContentBasedMemoryNote(MemoryNote):
    """研究内容1中专注于内容的记忆块"""

    def __init__(self, content: str, *args, **kwargs):
        super().__init__(content, *args, **kwargs)

    def extra_info(self) -> str:
        """返回额外的内容信息"""
        return f"Content-based additional info: {self.content[:20]}..."


class MemoryManager:
    """Memory management system to store and retrieve MemoryNote objects"""

    def __init__(self):
        # 存储所有记忆块的字典，key 为 memory.id，value 为 MemoryNote 对象
        self.memory_store = {}

    def add_memory(self, memory: MemoryNote) -> str:
        """Add a new memory to the manager."""
        self.memory_store[memory.id] = memory
        return memory.id  # 返回新记忆块的 ID

    def get_memory(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory by its ID."""
        return self.memory_store.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        if memory_id in self.memory_store:
            del self.memory_store[memory_id]
            return True
        return False

    def get_all_memories(self) -> List[MemoryNote]:
        """Retrieve all stored memories."""
        return list(self.memory_store.values())

    def update_memory(self, memory_id: str, new_content: Optional[str] = None) -> bool:
        """Update an existing memory's content or metadata."""
        memory = self.memory_store.get(memory_id)
        if memory:
            if new_content:
                memory.content = new_content
            return True
        return False

    def get_summary(self) -> str:
        """Get a summary of all stored memories."""
        summary = "\n".join([f"{memory.id}: {memory.get_summary()}" for memory in self.memory_store.values()])
        return summary

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        print("Consolidating memories...")


class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""

    def __init__(self, model_name: str = '../model/all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize the hybrid retriever.

        Args:
            model_name: Name of the SentenceTransformer model to use
            alpha: Weight for combining BM25 and semantic scores (0 = only BM25, 1 = only semantic)
        """
        self.model = SentenceTransformer(model_name)  # 使用预训练的 SentenceTransformer 模型
        self.alpha = alpha  # alpha 用于调整 BM25 和语义检索的加权比例
        self.bm25 = None  # 用于存储 BM25 模型
        self.corpus = []  # 存储所有的文档（记忆块）
        self.embeddings = None  # 存储文档的语义嵌入向量
        self.document_ids = {}  # 存储文档的 ID 和索引对应关系

    def add_documents(self, documents: List[str]) -> bool:
        """Add documents to both BM25 and semantic index"""
        if not documents:
            return False

        # 对文档进行 BM25 分词处理
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)  # 初始化 BM25

        # 为每个文档创建语义嵌入向量
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        for idx, document in enumerate(documents):
            self.document_ids[document] = idx

        return True

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """Retrieve documents using hybrid scoring"""
        if not self.corpus:
            return []

        # 获取 BM25 分数
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 标准化 BM25 分数
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        # 获取语义相似度分数
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        # 结合 BM25 和语义相似度分数
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores

        # 获取排名前 k 的索引
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]

        return top_k_indices.tolist()

class AgenticMemorySystem:
    """Memory system with management and retrieval capabilities."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        self.memory_manager = MemoryManager()  # 记忆管理系统
        self.retriever = HybridRetriever(model_name=model_name, alpha=alpha)  # 混合检索器
        self.evo_threshold = 10  # 演化阈值
        self.evo_count = 0  # 演化计数

    def add_note(self, content: str, **kwargs):
        """Add a new memory note and update retriever."""
        note = ContentBasedMemoryNote(content=content, **kwargs)
        self.memory_manager.add_memory(note)  # 添加记忆块
        self.retriever.add_documents([note.content])  # 更新检索器
        return note.id

    def find_related_memories(self, query: str, k: int = 5):
        """Find related memories using hybrid retrieval."""
        indices = self.retriever.retrieve(query, k)
        related_memories = [self.memory_manager.get_memory(self.retriever.corpus[i]) for i in indices]
        return related_memories

    def consolidate_memories(self):
        """Consolidate all memories in the system."""
        self.retriever = HybridRetriever()  # 重置检索器
        all_memories = self.memory_manager.get_all_memories()
        all_contents = [memory.content for memory in all_memories]
        self.retriever.add_documents(all_contents)  # 重新添加所有记忆块到检索器
        print("Memories consolidated.")


if __name__ == '__main__':
    print(1)
    memory_system = AgenticMemorySystem()
    note_id = memory_system.add_note("This is a memory about neural networks.")
    query = "What are neural networks?"
    related_memories = memory_system.find_related_memories(query)
    for memory in related_memories:
        print(memory)

