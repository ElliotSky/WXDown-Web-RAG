"""
基于句子的切分策略（Sentence-based Chunking）

按句子边界切分文档，保留句子完整性。
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.documents import Document


class SentenceSplitter:
    """基于句子的切分器。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        sentence_separators: list[str] | None = None,
    ):
        """
        初始化句子切分器。

        Args:
            chunk_size: 单个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数（按句子数计算）
            sentence_separators: 句子分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if sentence_separators is None:
            # 中文和英文句子分隔符
            sentence_separators = [
                r"。\s*",
                r"！\s*",
                r"？\s*",
                r"\.\s+",
                r"!\s+",
                r"\?\s+",
                r"\n\n+",
            ]
        self.sentence_separators = sentence_separators

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        将文本分割成句子列表。

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 使用正则表达式分割句子
        pattern = "|".join(self.sentence_separators)
        sentences = re.split(pattern, text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        切分文档列表。

        Args:
            documents: 待切分的文档列表

        Returns:
            切分后的文档块列表
        """
        chunks = []

        for doc in documents:
            text = doc.page_content
            sentences = self._split_into_sentences(text)

            if not sentences:
                continue

            current_chunk: list[str] = []
            current_size = 0
            overlap_sentences: list[str] = []

            for sentence in sentences:
                sentence_size = len(sentence)

                # 如果单个句子超过 chunk_size，需要特殊处理
                if sentence_size > self.chunk_size:
                    # 先保存当前块
                    if current_chunk:
                        chunk_text = "".join(current_chunk)
                        chunk = Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_strategy": "sentence_based",
                                "chunk_size": len(chunk_text),
                                "chunk_overlap": 0,
                            },
                        )
                        chunks.append(chunk)
                        current_chunk = []
                        current_size = 0

                    # 将超长句子按字符切分
                    for i in range(0, sentence_size, self.chunk_size - self.chunk_overlap):
                        sub_sentence = sentence[i : i + self.chunk_size]
                        chunk = Document(
                            page_content=sub_sentence,
                            metadata={
                                **doc.metadata,
                                "chunk_strategy": "sentence_based",
                                "chunk_size": len(sub_sentence),
                                "chunk_overlap": 0,
                            },
                        )
                        chunks.append(chunk)
                    continue

                # 检查添加当前句子是否会超过 chunk_size
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # 保存当前块
                    chunk_text = "".join(current_chunk)
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "sentence_based",
                            "chunk_size": len(chunk_text),
                            "chunk_overlap": len("".join(overlap_sentences)),
                        },
                    )
                    chunks.append(chunk)

                    # 设置重叠：保留最后几个句子作为重叠
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break

                    # 开始新块，包含重叠句子
                    current_chunk = overlap_sentences.copy()
                    current_size = overlap_size

                current_chunk.append(sentence)
                current_size += sentence_size

            # 保存最后一个块
            if current_chunk:
                chunk_text = "".join(current_chunk)
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "sentence_based",
                        "chunk_size": len(chunk_text),
                        "chunk_overlap": 0,
                    },
                )
                chunks.append(chunk)

        return chunks

