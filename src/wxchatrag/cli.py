from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from wxchatrag.exceptions import (
    DataSourceNotFoundError,
    EmptyQuestionError,
    VectorStoreNotFoundError,
    WxchatragError,
)
from wxchatrag.rag_query import load_vector_store
from wxchatrag.services import RagIngestService, RagQueryService
from wxchatrag.settings import get_settings


def _configure_logging(level: str | None) -> None:
    """根据命令行参数初始化日志配置。"""
    numeric = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    """处理 `ingest` 子命令：构建或更新本地向量库。"""
    service = RagIngestService()
    result = service.run(mode=args.mode, limit=args.limit)
    if result is None:
        print("没有发现需要更新的 PDF，向量库保持不变。")
        return

    print(f"已处理 PDF 数量: {result.processed_pdf_count}")
    print(f"切分得到文本块数量: {result.chunk_count}")
    print(f"向量库已保存到: {result.vector_store_dir}")


def _read_question(args: argparse.Namespace) -> str:
    """从参数或标准输入读取问题文本，并做空值校验。"""
    if args.question:
        q = args.question.strip()
    else:
        q = input("请输入问题：").strip()
    if not q:
        raise EmptyQuestionError("问题不能为空")
    return q


def _serialize_response(resp) -> dict[str, Any]:
    """将 RAG 响应对象序列化为可 JSON 化的字典。"""
    return {
        "answer": resp.answer,
        "sources": [s.__dict__ for s in resp.sources],
    }


def _cmd_query(args: argparse.Namespace) -> None:
    """处理 `query` 子命令：基于本地向量库执行问答。"""
    service = RagQueryService()
    q = _read_question(args)
    resp = service.answer(
        q,
        retriever_k=args.k,
        debug_retrieval=args.debug_retrieval,
        preview_chars=args.preview_chars,
    )

    if args.json:
        print(json.dumps(_serialize_response(resp), ensure_ascii=False, indent=2))
        return

    print("\n=== 回答 ===\n")
    print(resp.answer)

    if args.with_sources and resp.sources:
        print("\n=== 命中文档 ===\n")
        for s in resp.sources:
            parts = [f"[{s.index}]", s.title]
            if s.channel:
                parts.append(f"公众号: {s.channel}")
            if s.date:
                parts.append(f"日期: {s.date}")
            if s.page is not None:
                parts.append(f"页码: {s.page}")
            if s.url:
                parts.append(f"URL: {s.url}")
            if s.path:
                parts.append(f"路径: {s.path}")
            print(" | ".join(parts))


def _cmd_list_embedded_pdfs(args: argparse.Namespace) -> None:
    """列出当前向量库中已向量化的 PDF 文件列表。"""
    settings = get_settings()
    vs = load_vector_store(
        vector_store_dir=str(settings.vector_store_dir),
        embedding_model_name=settings.embedding_model_name,
    )

    # langchain_community.vectorstores.FAISS 的 docstore 是一个 dict-like 对象
    docs = list(getattr(vs, "docstore", {}).__dict__.get("_dict", {}).values())

    sources: set[str] = set()
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        src = meta.get("source")
        if src:
            sources.add(str(src))

    if not sources:
        print("当前向量库中还没有任何已向量化的 PDF。")
        return

    print("=== 已向量化的 PDF 文件 ===\n")
    for src in sorted(sources):
        print(src)

    print(f"\n总计 {len(sources)} 个 PDF 已向量化。")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    p = argparse.ArgumentParser(prog="wxchatrag", description="基于本地 WXhub 语料的 RAG 问答工具")
    p.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，例如 DEBUG/INFO/WARNING/ERROR（默认：INFO）",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    ingest_parser = sub.add_parser("ingest", help="从 WXhub PDF 构建 / 更新向量库")
    ingest_parser.add_argument(
        "--mode",
        choices=["rebuild", "update"],
        default="update",
        help="rebuild: 全量重建; update: 增量更新（默认）",
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个 PDF（调试用）",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    query = sub.add_parser("query", help="基于本地向量库进行问答")
    query.add_argument("--question", "-q", default="", help="问题文本，如不提供则进入交互模式")
    query.add_argument("--k", type=int, default=None, help="top-k 检索数量，覆盖默认配置")
    query.add_argument("--with-sources", action="store_true", help="是否打印命中文档来源")
    query.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    query.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="在回答前输出检索中间状态（命中 PDF/页码/相似度/片段预览），便于调试",
    )
    query.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="--debug-retrieval 时，每个命中文档打印的片段预览字符数（默认 200，0 表示不打印预览）",
    )
    query.set_defaults(func=_cmd_query)

    embedded = sub.add_parser("embedded-pdfs", help="列出当前向量库中已向量化的 PDF 列表")
    embedded.set_defaults(func=_cmd_list_embedded_pdfs)

    return p


def main(argv: list[str] | None = None) -> int:
    """wxchatrag CLI 入口函数。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(getattr(args, "log_level", "INFO"))

    try:
        args.func(args)
        return 0
    except (DataSourceNotFoundError, VectorStoreNotFoundError, EmptyQuestionError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except WxchatragError as e:
        print(f"wxchatrag 内部错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

