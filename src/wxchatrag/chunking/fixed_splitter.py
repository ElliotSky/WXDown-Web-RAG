"""
固定大小切分策略（Fixed Size Chunking）

使用 RecursiveCharacterTextSplitter 按固定大小切分文档。
这是最基础也是最常用的切分策略。
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FixedSizeSplitter:
    """固定大小切分器。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """
        初始化固定大小切分器。

        Args:
            chunk_size: 单个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            separators: 分隔符列表，按优先级排序
        """
        if separators is None:
            # 默认分隔符：段落、句子、单词、字符
            separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            add_start_index=True,
        )

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        切分文档列表。

        Args:
            documents: 待切分的文档列表

        Returns:
            切分后的文档块列表
        """
        chunks = self.splitter.split_documents(documents)
        # 添加切分策略元数据
        for chunk in chunks:
            chunk.metadata["chunk_strategy"] = "fixed_size"
            chunk.metadata["chunk_size"] = self.splitter._chunk_size
            chunk.metadata["chunk_overlap"] = self.splitter._chunk_overlap
        return chunks

