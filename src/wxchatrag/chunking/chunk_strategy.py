"""
切分策略工厂

统一管理所有切分策略，提供统一的接口。
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..embeddings.bge_embeddings import BGEEmbeddings
from ..settings import get_embedding_api_config, get_settings
from .fixed_splitter import FixedSizeSplitter
from .hierarchical_splitter import HierarchicalSplitter
from .semantic_splitter import SemanticSplitter
from .sentence_splitter import SentenceSplitter
from .sliding_window_splitter import SlidingWindowSplitter


class ChunkStrategy:
    """切分策略工厂类。"""

    @staticmethod
    def create_splitter(
        strategy: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: OpenAIEmbeddings | None = None,
        **kwargs,
    ):
        """
        创建切分器实例。

        Args:
            strategy: 切分策略名称
                - "fixed": 固定大小切分
                - "sentence": 基于句子的切分
                - "semantic": 语义切分
                - "hierarchical": 层次化切分
                - "sliding_window": 滑动窗口切分
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            embeddings: 嵌入模型（仅语义切分需要）
            **kwargs: 其他策略特定参数

        Returns:
            切分器实例

        Raises:
            ValueError: 未知的策略名称
        """
        if strategy == "fixed":
            return FixedSizeSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=kwargs.get("separators"),
            )

        elif strategy == "sentence":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                sentence_separators=kwargs.get("sentence_separators"),
            )

        elif strategy == "semantic":
            settings = get_settings()
            if embeddings is None:
                # 从配置或参数获取语义切分embedding设置
                mode = kwargs.get("semantic_embedding_mode", settings.semantic_embedding_mode)
                embedding_model_name = kwargs.get(
                    "embedding_model_name", settings.semantic_embedding_model_name
                )
                
                if mode == "local":
                    # 使用本地部署的 BGE 模型（北京智源研究院）
                    embeddings = BGEEmbeddings(
                        model_name=embedding_model_name,
                    )
                elif mode == "api":
                    # 使用标准API接口（该方法不支持GLM）
                    api_key, base_url = get_embedding_api_config()
                    if not api_key:
                        raise ValueError(
                            "语义切分使用API模式时，必须配置 EMBEDDING_API_KEY 环境变量"
                        )
                    embeddings = OpenAIEmbeddings(
                        model=embedding_model_name,
                        chunk_size=64,
                        api_key=api_key,
                        base_url=base_url,
                    )
                else:
                    raise ValueError(
                        f"不支持的语义切分embedding模式: {mode}，支持的模式: local, api"
                    )

            sentence_split_mode = kwargs.get(
                "sentence_split_mode",
                kwargs.get(
                    "semantic_sentence_split_mode",
                    settings.semantic_sentence_split_mode,
                ),
            )
            sentence_split_regex = kwargs.get(
                "sentence_split_regex",
                kwargs.get(
                    "semantic_sentence_split_regex",
                    settings.semantic_sentence_split_regex,
                ),
            )
            return SemanticSplitter(
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=kwargs.get("similarity_threshold", 0.5),
                breakpoint_threshold_type=kwargs.get(
                    "breakpoint_threshold_type", "percentile"
                ),
                sentence_split_mode=sentence_split_mode,
                sentence_split_regex=sentence_split_regex or None,
            )

        elif strategy == "hierarchical":
            return HierarchicalSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator=kwargs.get("paragraph_separator", "\n\n"),
            )

        elif strategy == "sliding_window":
            return SlidingWindowSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                window_step=kwargs.get("window_step"),
            )

        else:
            raise ValueError(
                f"未知的切分策略: {strategy}. "
                f"支持的策略: fixed, sentence, semantic, hierarchical, sliding_window"
            )

    @staticmethod
    def split_documents(
        documents: Sequence[Document],
        strategy: str = "fixed",
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: OpenAIEmbeddings | None = None,
        **kwargs,
    ) -> list[Document]:
        """
        使用指定策略切分文档。

        Args:
            documents: 待切分的文档列表
            strategy: 切分策略名称
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            embeddings: 嵌入模型（仅语义切分需要）
            **kwargs: 其他策略特定参数

        Returns:
            切分后的文档块列表
        """
        splitter = ChunkStrategy.create_splitter(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embeddings=embeddings,
            **kwargs,
        )
        return splitter.split_documents(documents)

