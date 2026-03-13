from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chunking.chunk_strategy import ChunkStrategy
from .embeddings.bge_embeddings import BGEEmbeddings
from .retrieval.bm25_store import BM25Store
from .settings import get_embedding_api_config, get_settings


def split_docs(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str = "fixed",
    embedding_model_name: str | None = None,
    semantic_threshold: float = 0.5,
    semantic_embedding_mode: str | None = None,
    semantic_embedding_model_name: str | None = None,
    semantic_sentence_split_mode: str | None = None,
    semantic_sentence_split_regex: str | None = None,
) -> list[Document]:
    """
    将原始文档切分为带重叠的语义块，用于后续向量化。

    Args:
        documents: 待切分的文档列表
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        chunk_strategy: 切分策略（"fixed", "sentence", "semantic", "hierarchical", "sliding_window"）
        semantic_threshold: 语义切分阈值（仅语义切分需要）
        semantic_embedding_mode: 语义切分embedding模式，"local"（本地BGE）或 "api"（API接口），默认从配置读取
        semantic_embedding_model_name: 语义切分embedding模型名称，默认从配置读取
        semantic_sentence_split_mode: 语义分句模式，默认从配置读取
        semantic_sentence_split_regex: 语义分句正则，默认从配置读取

    Returns:
        切分后的文档块列表
    """
    # 如果是语义切分，需要嵌入模型
    embeddings = None
    if chunk_strategy == "semantic":
        # 从配置或参数获取语义切分embedding设置
        settings = get_settings()
        mode = semantic_embedding_mode or settings.semantic_embedding_mode
        model_name = semantic_embedding_model_name or embedding_model_name or settings.semantic_embedding_model_name
        
        if mode == "local":
            # 使用本地部署的 BGE 模型（北京智源研究院）
            embeddings = BGEEmbeddings(
                model_name=model_name,
            )
        elif mode == "api":
            # 使用标准API接口（支持GLM、OpenAI等）
            api_key, base_url = get_embedding_api_config()
            if not api_key:
                raise ValueError(
                    "语义切分使用API模式时，必须配置 EMBEDDING_API_KEY 环境变量"
                )
            embeddings = OpenAIEmbeddings(
                model=model_name,
                chunk_size=64,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            raise ValueError(
                f"不支持的语义切分embedding模式: {mode}，支持的模式: local, api"
            )

    return ChunkStrategy.split_documents(
        documents=documents,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings=embeddings,
        embedding_model_name=semantic_embedding_model_name or embedding_model_name,
        semantic_embedding_mode=semantic_embedding_mode,
        semantic_sentence_split_mode=semantic_sentence_split_mode,
        semantic_sentence_split_regex=semantic_sentence_split_regex,
        similarity_threshold=semantic_threshold,
    )


def build_vector_store(
    chunks: Iterable[Document],
    *,
    embedding_model_name: str,
) -> FAISS:
    """根据文档块构建新的向量库。"""
    # GLM Embedding 接口限制单次请求 input 数组长度 <= 64，
    # 这里通过 chunk_size 显式控制批大小，避免 400 code=1214 错误。
    api_key, base_url = get_embedding_api_config()
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name, 
        chunk_size=64,
        api_key=api_key,
        base_url=base_url,
    )
    return FAISS.from_documents(chunks, embeddings)


def persist_vector_store(vs: FAISS, save_dir: str) -> None:
    """将向量库持久化到本地目录（如不存在会自动创建）。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(save_dir)


def build_bm25_index(
    chunks: Iterable[Document],
    *,
    bm25_index_dir: Path,
) -> BM25Store:
    """根据文档块构建 BM25 索引。"""
    bm25_store = BM25Store(bm25_index_dir)
    bm25_store.build_index(chunks)
    return bm25_store


def persist_bm25_index(bm25_store: BM25Store) -> None:
    """将 BM25 索引持久化到磁盘。"""
    bm25_store.save()


def _load_or_create_store(
    *,
    vector_store_dir: Path,
    embedding_model_name: str,
    chunks: Iterable[Document],
    mode: str,
) -> FAISS:
    """按模式加载或新建向量库。

    - mode="rebuild"：删除旧库并重新构建
    - mode="update"：在原有库基础上追加文档
    """
    import shutil

    if mode == "rebuild" and vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)

    if vector_store_dir.exists():
        api_key, base_url = get_embedding_api_config()
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name, 
            chunk_size=64,
            api_key=api_key,
            base_url=base_url,
        )
        vs = FAISS.load_local(
            str(vector_store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vs.add_documents(list(chunks))
        return vs

    return build_vector_store(chunks, embedding_model_name=embedding_model_name)


def _load_or_create_bm25_store(
    *,
    bm25_index_dir: Path,
    chunks: Iterable[Document],
    mode: str,
) -> BM25Store:
    """按模式加载或新建 BM25 索引。

    - mode="rebuild"：删除旧索引并重新构建
    - mode="update"：在原有索引基础上追加文档
    """
    import shutil

    if mode == "rebuild" and bm25_index_dir.exists():
        shutil.rmtree(bm25_index_dir)

    bm25_store = BM25Store(bm25_index_dir)
    if bm25_store.exists():
        bm25_store.load()
        bm25_store.add_documents(chunks)
    else:
        bm25_store.build_index(chunks)

    return bm25_store

