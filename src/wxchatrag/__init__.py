"""
wxchatrag

面向微信公众号 PDF 语料的轻量级 RAG 库。

对外暴露两类核心能力：
- ingest：构建 / 更新向量索引
- query_rag：基于本地向量库进行检索增强问答
"""

from .ingest import build_vector_store, persist_vector_store, split_docs
from .rag_query import RagResponse, RagSource, query_rag
from .services import IngestResult, RagIngestService, RagQueryService
from .settings import get_settings

__all__ = [
    "build_vector_store",
    "persist_vector_store",
    "split_docs",
    "query_rag",
    "RagResponse",
    "RagSource",
    "RagIngestService",
    "RagQueryService",
    "IngestResult",
    "get_settings",
]

