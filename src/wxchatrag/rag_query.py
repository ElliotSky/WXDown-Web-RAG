from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from wxchatrag.exceptions import VectorStoreNotFoundError
from wxchatrag.settings import get_settings


@dataclass(frozen=True)
class RagSource:
    """单条命中文档来源的元信息。"""

    index: int
    title: str
    channel: str | None
    date: str | None
    page: int | None
    url: str | None
    path: str | None


@dataclass(frozen=True)
class RagResponse:
    """标准化的 RAG 响应结果。"""

    answer: str
    sources: list[RagSource]


def load_vector_store(*, vector_store_dir: str, embedding_model_name: str) -> FAISS:
    """从本地目录加载向量库。

    如目录不存在，则抛出 VectorStoreNotFoundError，
    调用方可据此提示用户先执行 ingest。
    """
    if not Path(vector_store_dir).is_dir():
        raise VectorStoreNotFoundError(f"未找到向量库目录，请先执行 ingest: {vector_store_dir}")
    # 与 ingest 阶段保持一致，控制 GLM Embedding 的批大小，避免 input 数组超过 64 条。
    embeddings = OpenAIEmbeddings(model=embedding_model_name, chunk_size=64)
    return FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)


def _build_prompt() -> ChatPromptTemplate:
    """构造中文问答提示词模板。"""
    return ChatPromptTemplate.from_template(
        "你是一个基于微信公众号语料的中文问答助手，请严格按照下列要求回答：\n"
        "1) 优先使用给定的参考内容，不要凭空编造事实；\n"
        "2) 无法从参考内容中得到答案时，请直接说明“根据当前资料无法回答”；\n"
        "3) 如引用具体片段，请用 [序号] 标注对应的来源。\n\n"
        "【参考内容】\n{context}\n\n"
        "【用户问题】{question}\n\n"
        "【请给出你的回答】"
    )


def _source_from_doc(doc, index: int) -> RagSource:
    """从检索到的文档对象构造标准化的 RagSource。"""
    meta = doc.metadata or {}
    title = meta.get("title") or Path(meta.get("source", "")).stem
    return RagSource(
        index=index,
        title=title,
        channel=meta.get("channel"),
        date=meta.get("date"),
        page=meta.get("page"),
        url=meta.get("url") or meta.get("msg_link"),
        path=meta.get("source"),
    )


def _format_docs(docs: Iterable) -> tuple[str, list[RagSource]]:
    """将检索到的文档格式化为大模型上下文字符串与来源列表。"""
    sources: list[RagSource] = []
    blocks: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = _source_from_doc(doc, i)
        sources.append(source)
        header_parts = [f"[{i}]", source.title]
        if source.channel:
            header_parts.append(f"公众号:{source.channel}")
        if source.date:
            header_parts.append(f"日期:{source.date}")
        if source.page is not None:
            header_parts.append(f"页码:{source.page}")
        header = " | ".join(header_parts)
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks), sources


def _preview_text(text: str, *, limit: int) -> str:
    s = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    # 简单压缩多余空行，便于在终端快速扫视
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    if limit <= 0:
        return ""
    return s[:limit] + ("…" if len(s) > limit else "")


def _print_debug_hits(
    scored_docs: list[tuple[object, float]],
    *,
    k: int,
    question: str,
    preview_chars: int,
) -> None:
    """
    输出检索中间状态（写入 stderr，避免污染 --json 输出）。

    scored_docs: [(Document, distance)]
    """
    print("\n=== 检索中间状态 ===", file=sys.stderr)
    print(f"- question: {question}", file=sys.stderr)
    print(f"- top_k: {k}", file=sys.stderr)
    print("", file=sys.stderr)

    if not scored_docs:
        print("(无命中文档)", file=sys.stderr)
        print("", file=sys.stderr)
        return

    for i, (doc, distance) in enumerate(scored_docs, 1):
        meta = getattr(doc, "metadata", None) or {}
        title = meta.get("title") or Path(meta.get("source", "")).stem
        channel = meta.get("channel")
        date = meta.get("date")
        page = meta.get("page")
        start_index = meta.get("start_index")
        source = meta.get("source")

        header_parts = [f"[{i}]", f"distance={distance:.6f}", title]
        if channel:
            header_parts.append(f"公众号:{channel}")
        if date:
            header_parts.append(f"日期:{date}")
        if page is not None:
            header_parts.append(f"页码:{page}")
        if start_index is not None:
            header_parts.append(f"片段起始:{start_index}")
        if source:
            header_parts.append(f"PDF:{source}")

        print(" | ".join(header_parts), file=sys.stderr)
        content = getattr(doc, "page_content", "") or ""
        if preview_chars > 0:
            print(f"片段预览: {_preview_text(content, limit=preview_chars)}", file=sys.stderr)
        print("", file=sys.stderr)


def query_rag(
    question: str,
    *,
    retriever_k: Optional[int] = None,
    debug_retrieval: bool = False,
    preview_chars: int = 200,
) -> RagResponse:
    """执行一次标准化 RAG 查询，并返回答案与来源列表。"""
    s = get_settings()
    k = retriever_k if retriever_k is not None else s.retriever_k
    vs = load_vector_store(
        vector_store_dir=str(s.vector_store_dir),
        embedding_model_name=s.embedding_model_name,
    )
    # 使用带分数的检索，便于输出“检索到了什么”的中间状态。
    scored = vs.similarity_search_with_score(question, k=k)
    if debug_retrieval:
        _print_debug_hits(scored, k=k, question=question, preview_chars=preview_chars)

    docs = [d for d, _ in scored]
    if not docs:
        return RagResponse(answer="当前向量库中没有找到与问题足够相关的内容，暂时无法回答。", sources=[])

    context, sources = _format_docs(docs)
    prompt = _build_prompt()

    # 向量检索已使用 GLM / 其它 OpenAI 兼容服务提供的 Embedding（由 OPENAI_* 决定），
    # 这里显式使用 DeepSeek 提供的 Chat 接口，避免与 Embedding 复用同一组环境变量。
    chat_api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    chat_base_url = (os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()

    llm = ChatOpenAI(
        model=s.chat_model_name,
        temperature=s.temperature,
        api_key=chat_api_key or None,
        base_url=chat_base_url or None,
    )
    messages = prompt.format_messages(context=context, question=question)
    resp = llm.invoke(messages)
    return RagResponse(answer=resp.content, sources=sources)


def main() -> None:
    """简单的交互式命令行入口（调试用）。"""
    q = input("请输入问题：").strip()
    if not q:
        raise ValueError("问题不能为空")
    resp = query_rag(q)
    print("\n=== 回答 ===\n")
    print(resp.answer)
    if resp.sources:
        print("\n=== 命中文档 ===\n")
        for s in resp.sources:
            parts = [f"[{s.index}]", s.title]
            if s.channel:
                parts.append(f"公众号:{s.channel}")
            if s.date:
                parts.append(f"日期:{s.date}")
            if s.page is not None:
                parts.append(f"页码:{s.page}")
            if s.url:
                parts.append(f"URL:{s.url}")
            if s.path:
                parts.append(f"路径:{s.path}")
            print(" | ".join(parts))


if __name__ == "__main__":
    main()
