from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """将原始文档切分为带重叠的语义块，用于后续向量化。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vector_store(
    chunks: Iterable[Document],
    *,
    embedding_model_name: str,
) -> FAISS:
    """根据文档块构建新的向量库。"""
    # GLM Embedding 接口限制单次请求 input 数组长度 <= 64，
    # 这里通过 chunk_size 显式控制批大小，避免 400 code=1214 错误。
    embeddings = OpenAIEmbeddings(model=embedding_model_name, chunk_size=64)
    return FAISS.from_documents(chunks, embeddings)


def persist_vector_store(vs: FAISS, save_dir: str) -> None:
    """将向量库持久化到本地目录（如不存在会自动创建）。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(save_dir)


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
        embeddings = OpenAIEmbeddings(model=embedding_model_name, chunk_size=64)
        vs = FAISS.load_local(
            str(vector_store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vs.add_documents(list(chunks))
        return vs

    return build_vector_store(chunks, embedding_model_name=embedding_model_name)

