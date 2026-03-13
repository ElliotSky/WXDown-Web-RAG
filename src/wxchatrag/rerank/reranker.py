"""
重排序器封装

提供统一的重排序接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from .cross_encoder_rerank import CrossEncoderReranker


class Reranker:
    """重排序器封装类。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        cache_dir: Path | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        """
        初始化重排序器。

        Args:
            model_name: 模型名称
            cache_dir: 模型缓存目录
            device: 设备类型
            batch_size: 批量推理大小
        """
        self.reranker = CrossEncoderReranker(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        对文档列表进行重排序。

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回 top-k 结果

        Returns:
            [(Document, score), ...] 列表，按分数降序排列
        """
        return self.reranker.rerank(query, documents, top_k=top_k)

