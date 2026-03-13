"""
层次化切分策略（Hierarchical Chunking）

先按段落切分，再按句子切分，形成多层次的文档结构。
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from .sentence_splitter import SentenceSplitter


class HierarchicalSplitter:
    """层次化切分器。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        paragraph_separator: str = "\n\n",
    ):
        """
        初始化层次化切分器。

        Args:
            chunk_size: 单个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            paragraph_separator: 段落分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        切分文档列表。

        策略：
        1. 先按段落分割
        2. 如果段落超过 chunk_size，再按句子分割
        3. 保留层次结构信息

        Args:
            documents: 待切分的文档列表

        Returns:
            切分后的文档块列表
        """
        chunks = []

        for doc in documents:
            text = doc.page_content
            paragraphs = text.split(self.paragraph_separator)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            for para_idx, paragraph in enumerate(paragraphs):
                # 如果段落较小，直接作为一个块
                if len(paragraph) <= self.chunk_size:
                    chunk = Document(
                        page_content=paragraph,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "hierarchical",
                            "chunk_level": "paragraph",
                            "paragraph_index": para_idx,
                            "chunk_size": len(paragraph),
                            "chunk_overlap": 0,
                        },
                    )
                    chunks.append(chunk)
                else:
                    # 段落较大，按句子切分
                    para_doc = Document(
                        page_content=paragraph,
                        metadata={
                            **doc.metadata,
                            "paragraph_index": para_idx,
                        },
                    )
                    sentence_chunks = self.sentence_splitter.split_documents([para_doc])
                    # 更新元数据，标记为句子级别
                    for chunk in sentence_chunks:
                        chunk.metadata["chunk_strategy"] = "hierarchical"
                        chunk.metadata["chunk_level"] = "sentence"
                        chunk.metadata["paragraph_index"] = para_idx
                    chunks.extend(sentence_chunks)

        return chunks

