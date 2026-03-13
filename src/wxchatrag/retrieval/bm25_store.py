"""
BM25 索引存储与管理

提供 BM25 索引的构建、持久化和加载功能。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Store:
    """BM25 索引存储类。"""

    def __init__(self, index_dir: Path):
        """
        初始化 BM25 存储。

        Args:
            index_dir: BM25 索引存储目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "bm25_index.pkl"
        self.documents_path = self.index_dir / "documents.pkl"

        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []

    def build_index(self, documents: Iterable[Document]) -> None:
        """
        构建 BM25 索引。

        Args:
            documents: 文档列表
        """
        self.documents = list(documents)

        # 对每个文档进行分词（简单的中文分词：按字符分割）
        # 对于中文，可以使用 jieba 等分词工具，这里先用简单方法
        tokenized_docs = []
        for doc in self.documents:
            # 简单的中文分词：按字符分割，过滤空白字符
            tokens = [char for char in doc.page_content if char.strip()]
            tokenized_docs.append(tokens)

        if not tokenized_docs:
            raise ValueError("无法构建 BM25 索引：文档列表为空")

        self.bm25 = BM25Okapi(tokenized_docs)

    def save(self) -> None:
        """保存 BM25 索引到磁盘。"""
        if self.bm25 is None:
            raise ValueError("BM25 索引未构建，无法保存")

        # 保存 BM25 索引
        with self.index_path.open("wb") as f:
            pickle.dump(self.bm25, f)

        # 保存文档列表
        with self.documents_path.open("wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> None:
        """从磁盘加载 BM25 索引。"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 索引文件不存在: {self.index_path}")

        with self.index_path.open("rb") as f:
            self.bm25 = pickle.load(f)

        if self.documents_path.exists():
            with self.documents_path.open("rb") as f:
                self.documents = pickle.load(f)
        else:
            # 兼容旧版本：如果没有文档列表，创建空列表
            self.documents = []

    def add_documents(self, documents: Iterable[Document]) -> None:
        """
        向现有索引添加文档（增量更新）。

        Args:
            documents: 新文档列表
        """
        new_docs = list(documents)
        if not new_docs:
            return

        # 如果索引不存在，直接构建
        if self.bm25 is None:
            self.build_index(new_docs)
            return

        # 合并文档列表
        self.documents.extend(new_docs)

        # 重新构建索引（BM25Okapi 不支持增量更新，需要重建）
        tokenized_docs = []
        for doc in self.documents:
            tokens = [char for char in doc.page_content if char.strip()]
            tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        使用 BM25 检索。

        Args:
            query: 查询文本
            top_k: 返回 top-k 结果

        Returns:
            [(Document, score), ...] 列表，按分数降序排列
        """
        if self.bm25 is None:
            raise ValueError("BM25 索引未加载或未构建")

        # 对查询进行分词
        query_tokens = [char for char in query if char.strip()]

        if not query_tokens:
            return []

        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 获取 top-k 索引
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # 构建结果列表
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(scores[idx])))

        return results

    def exists(self) -> bool:
        """检查索引是否存在。"""
        return self.index_path.exists()

