"""
Cross-Encoder 重排序实现

使用 Cross-Encoder 模型对检索结果进行重排序。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


class CrossEncoderReranker:
    """基于 Cross-Encoder 的重排序器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        cache_dir: Path | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        """
        初始化 Cross-Encoder 重排序器。

        Args:
            model_name: 模型名称或路径
            cache_dir: 模型缓存目录
            device: 设备类型（"cpu" 或 "cuda"）
            batch_size: 批量推理大小
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self) -> None:
        """延迟加载模型。"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers: pip install sentence-transformers>=2.2.0"
            )

        cache_dir_str = str(self.cache_dir) if self.cache_dir else None
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            cache_folder=cache_dir_str,
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
            top_k: 返回 top-k 结果，None 表示返回全部

        Returns:
            [(Document, score), ...] 列表，按分数降序排列
        """
        if not documents:
            return []

        self._load_model()

        # 构建 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in documents]

        # 批量推理
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # 组合结果并排序
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回 top-k
        if top_k is not None:
            results = results[:top_k]

        return results

