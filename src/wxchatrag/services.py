from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .exceptions import DataSourceNotFoundError
from .ingest import _load_or_create_store, persist_vector_store, split_docs
from .manifest import build_file_state, load_manifest, save_manifest, select_changed
from .rag_query import RagResponse, query_rag
from .settings import Settings, get_settings
from .wxhub_loader import build_channel_indexes, iter_pdf_paths, load_pdf_documents


@dataclass
class IngestResult:
    """一次入库执行的结果摘要。

    作为入库服务 `RagIngestService.run` 的返回值，方便 CLI / 上层服务统一展示统计信息。
    """

    processed_pdf_count: int
    chunk_count: int
    vector_store_dir: Path


class RagIngestService:
    """负责从 WXhub 数据源构建 / 更新向量库的服务。

    对外屏蔽具体的 PDF 加载、切分、向量化与 manifest 维护等细节，
    仅暴露一次性执行 ingest 流程的 `run` 方法。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        """返回当前使用的配置对象。"""
        return self._settings

    def _iter_target_pdfs(
        self,
        *,
        mode: str,
        limit: int | None,
    ) -> list[Path]:
        """根据模式与限制条件，返回本次需要处理的 PDF 列表。"""
        s = self._settings
        if not s.wxhub_root.is_dir():
            raise DataSourceNotFoundError(f"WXhub 数据根目录不存在: {s.wxhub_root}")

        pdf_paths = iter_pdf_paths(s.wxhub_root, s.pdf_glob_pattern)
        if limit is not None:
            pdf_paths = pdf_paths[:limit]

        manifest_path = s.vector_store_dir / "manifest.json"
        manifest = load_manifest(manifest_path)
        return pdf_paths if mode == "rebuild" else select_changed(pdf_paths, manifest)

    def _load_documents(self, pdf_paths: Iterable[Path]) -> list[Document]:
        """从指定 PDF 路径列表中加载文档，并补充频道等元数据。"""
        s = self._settings
        channel_indexes = build_channel_indexes(s.wxhub_root)
        return load_pdf_documents(pdf_paths=pdf_paths, channel_indexes=channel_indexes)

    def run(
        self,
        *,
        mode: str = "update",
        limit: int | None = None,
    ) -> IngestResult | None:
        """执行一次 ingest 流程。

        - mode="update"：仅处理变更的 PDF
        - mode="rebuild"：全量重建向量库
        """
        s = self._settings
        to_process = self._iter_target_pdfs(mode=mode, limit=limit)
        if not to_process:
            # 无需更新时返回 None，调用方可据此决定是否提示
            return None

        docs = self._load_documents(to_process)
        chunks = split_docs(
            docs,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
        )

        vs = _load_or_create_store(
            vector_store_dir=s.vector_store_dir,
            embedding_model_name=s.embedding_model_name,
            chunks=chunks,
            mode=mode,
        )
        persist_vector_store(vs, str(s.vector_store_dir))

        # 更新 manifest：表示“哪些 PDF 已经完成向量化”
        manifest_path = s.vector_store_dir / "manifest.json"
        old_manifest = load_manifest(manifest_path)
        # 先保留旧的已处理文件，再用本次实际处理的文件状态覆盖 / 补充
        merged_states: dict[str, FileState] = {k: v for k, v in old_manifest.items()}
        for p in to_process:
            state = build_file_state(p)
            merged_states[state.path] = state
        save_manifest(manifest_path, merged_states.values())

        return IngestResult(
            processed_pdf_count=len(to_process),
            chunk_count=len(chunks),
            vector_store_dir=s.vector_store_dir,
        )


class RagQueryService:
    """负责基于本地向量库执行 RAG 问答的服务。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def answer(
        self,
        question: str,
        *,
        retriever_k: int | None = None,
        debug_retrieval: bool = False,
        preview_chars: int = 200,
    ) -> RagResponse:
        """对外暴露的统一问答接口。"""
        return query_rag(
            question,
            retriever_k=retriever_k,
            debug_retrieval=debug_retrieval,
            preview_chars=preview_chars,
        )



