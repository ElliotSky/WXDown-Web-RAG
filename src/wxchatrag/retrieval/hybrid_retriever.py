"""
混合检索器（Hybrid Retriever）

结合 BM25 和向量检索的优势，使用 RRF 融合结果。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .bm25_store import BM25Store
from .rrf_fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from pathlib import Path


class HybridRetriever:
    """混合检索器：BM25 + 向量检索。"""

    def __init__(
        self,
        vector_store: FAISS,
        bm25_store: BM25Store,
        *,
        hybrid_alpha: float = 0.7,
        bm25_k: int = 20,
        vector_k: int = 20,
    ):
        """
        初始化混合检索器。

        Args:
            vector_store: FAISS 向量库
            bm25_store: BM25 索引存储
            hybrid_alpha: 向量检索权重（0-1），BM25 权重 = 1 - alpha
            bm25_k: BM25 检索数量
            vector_k: 向量检索数量
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.hybrid_alpha = hybrid_alpha
        self.bm25_k = bm25_k
        self.vector_k = vector_k

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        执行混合检索。

        Args:
            query: 查询文本
            top_k: 最终返回的 top-k 结果

        Returns:
            [(Document, score), ...] 列表，按融合分数降序排列
        """
        # 1. BM25 检索
        bm25_results = self.bm25_store.search(query, top_k=self.bm25_k)

        # 2. 向量检索
        vector_results_with_score = self.vector_store.similarity_search_with_score(
            query, k=self.vector_k
        )
        # 转换为 (Document, score) 格式，注意 FAISS 返回的是 distance，需要转换为相似度
        vector_results = [
            (doc, 1.0 / (1.0 + score)) for doc, score in vector_results_with_score
        ]

        # 3. 使用 RRF 融合结果
        fused = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            k=60,
        )

        # 4. 取 top-k
        top_results = fused[:top_k]

        # 5. 恢复 Document 对象（RRF 返回的是 item，即 Document）
        final_results = []
        for doc, rrf_score in top_results:
            if isinstance(doc, Document):
                final_results.append((doc, rrf_score))

        return final_results

    @classmethod
    def from_storage(
        cls,
        vector_store_dir: Path,
        bm25_index_dir: Path,
        embedding_model_name: str,
        *,
        hybrid_alpha: float = 0.7,
        bm25_k: int = 20,
        vector_k: int = 20,
    ) -> "HybridRetriever":
        """
        从存储目录加载混合检索器。

        Args:
            vector_store_dir: 向量库目录
            bm25_index_dir: BM25 索引目录
            embedding_model_name: 嵌入模型名称
            hybrid_alpha: 向量检索权重
            bm25_k: BM25 检索数量
            vector_k: 向量检索数量

        Returns:
            HybridRetriever 实例
        """
        from langchain_openai import OpenAIEmbeddings
        from ..settings import get_embedding_api_config

        # 加载向量库
        api_key, base_url = get_embedding_api_config()
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name, 
            chunk_size=64,
            api_key=api_key,
            base_url=base_url,
        )
        vector_store = FAISS.load_local(
            str(vector_store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # 加载 BM25 索引
        bm25_store = BM25Store(bm25_index_dir)
        if bm25_store.exists():
            bm25_store.load()
        else:
            raise FileNotFoundError(
                f"BM25 索引不存在，请先执行 ingest: {bm25_index_dir}"
            )

        return cls(
            vector_store=vector_store,
            bm25_store=bm25_store,
            hybrid_alpha=hybrid_alpha,
            bm25_k=bm25_k,
            vector_k=vector_k,
        )

