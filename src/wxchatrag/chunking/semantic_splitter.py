"""
语义切分策略（Semantic Chunking）

基于嵌入相似度的语义切分，自动识别语义边界。

"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


SEMANTIC_SENTENCE_SPLIT_REGEX_PRESETS = {
    "english": r"(?<=[.?!])\s+",
    "chinese": r"(?<=[。！？；])|\n+",
    "mixed": r"(?<=[。！？；])|(?<=[.?!])\s+|\n+",
}


class SemanticSplitter:
    """语义切分器。
    
    语义切分的工作原理：
    1. 将文档分割成句子或小段
    2. 调用嵌入模型 API，将每个句子转换为向量
    3. 计算相邻句子的语义相似度（余弦相似度）
    4. 在相似度低的地方（语义边界）进行切分
    
    """

    def __init__(
        self,
        embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.5,
        breakpoint_threshold_type: str = "percentile",
        sentence_split_mode: str = "mixed",
        sentence_split_regex: str | None = None,
    ):
        """
        初始化语义切分器。

        Args:
            embeddings: 嵌入模型实例（用于计算语义相似度）
            chunk_size: 单个块的目标字符数（作为参考）
            chunk_overlap: 相邻块之间的重叠字符数
            similarity_threshold: 相似度阈值，低于此值则切分
            breakpoint_threshold_type: 阈值类型（"percentile" 或 "standard_deviation"）
            sentence_split_mode: 分句模式（"mixed"、"chinese"、"english"、"custom"）
            sentence_split_regex: 自定义分句正则；如果提供，则优先于 sentence_split_mode
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.sentence_split_mode = sentence_split_mode
        self.sentence_split_regex = self._resolve_sentence_split_regex(
            sentence_split_mode=sentence_split_mode,
            sentence_split_regex=sentence_split_regex,
        )

        # 创建 SemanticChunker
        # 注意：SemanticChunker使用min_chunk_size而不是chunk_size
        # min_chunk_size控制最小块大小，语义切分会根据相似度自动切分，但会确保块不小于min_chunk_size
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            add_start_index=True,
            min_chunk_size=chunk_size,  # 使用min_chunk_size参数
            sentence_split_regex=self.sentence_split_regex,
        )

    @staticmethod
    def _resolve_sentence_split_regex(
        *,
        sentence_split_mode: str,
        sentence_split_regex: str | None,
    ) -> str:
        if sentence_split_regex:
            return sentence_split_regex

        normalized_mode = sentence_split_mode.strip().lower()
        if normalized_mode == "custom":
            raise ValueError(
                "sentence_split_mode=custom 时，必须提供 sentence_split_regex"
            )

        if normalized_mode not in SEMANTIC_SENTENCE_SPLIT_REGEX_PRESETS:
            supported_modes = ", ".join(
                ["mixed", "chinese", "english", "custom"]
            )
            raise ValueError(
                f"不支持的 sentence_split_mode: {sentence_split_mode}，"
                f"支持的模式: {supported_modes}"
            )

        return SEMANTIC_SENTENCE_SPLIT_REGEX_PRESETS[normalized_mode]

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
            chunk.metadata["chunk_strategy"] = "semantic"
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunk_overlap"] = 0  # 语义切分通常不重叠
            chunk.metadata["similarity_threshold"] = self.similarity_threshold
            chunk.metadata["sentence_split_mode"] = self.sentence_split_mode
            chunk.metadata["sentence_split_regex"] = self.sentence_split_regex

        return chunks

