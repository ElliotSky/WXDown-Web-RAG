"""
滑动窗口切分策略（Sliding Window Chunking）

固定大小的窗口，通过滑动产生重叠块。
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document


class SlidingWindowSplitter:
    """滑动窗口切分器。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        window_step: int | None = None,
    ):
        """
        初始化滑动窗口切分器。

        Args:
            chunk_size: 窗口大小（字符数）
            chunk_overlap: 重叠大小（字符数）
            window_step: 窗口步长（字符数），默认为 chunk_size - chunk_overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.window_step = window_step or (chunk_size - chunk_overlap)

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
            text_length = len(text)

            if text_length <= self.chunk_size:
                # 文档小于窗口大小，直接作为一个块
                chunk = Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "sliding_window",
                        "chunk_size": text_length,
                        "chunk_overlap": 0,
                        "window_start": 0,
                        "window_end": text_length,
                    },
                )
                chunks.append(chunk)
                continue

            # 滑动窗口切分
            start = 0
            chunk_index = 0

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]

                # 计算实际重叠大小
                actual_overlap = (
                    self.chunk_overlap if chunk_index > 0 else 0
                )

                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "sliding_window",
                        "chunk_size": len(chunk_text),
                        "chunk_overlap": actual_overlap,
                        "window_start": start,
                        "window_end": end,
                        "chunk_index": chunk_index,
                    },
                )
                chunks.append(chunk)

                # 移动到下一个窗口
                start += self.window_step
                chunk_index += 1

        return chunks

