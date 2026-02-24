from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


_TITLE_CLEAN_RE = re.compile(r"[\s\-_—–·\.，,。！？!?:：;；'\"“”‘’()（）\[\]【】<>《》/\\]+")
_DATE_TITLE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<title>.+)$")


@dataclass(frozen=True)
class WxhubMeta:
    """单篇微信公众号文章在本项目中的元数据表示。"""

    channel: str
    title: str
    date: str | None
    url: str | None
    msg_link: str | None
    create_time: str | None


def _normalize_title(title: str) -> str:
    """对标题做简单规整，便于与 JSONL 中的标题进行模糊匹配。"""
    return _TITLE_CLEAN_RE.sub("", title).lower()


def _parse_title_from_filename(path: Path) -> tuple[str | None, str]:
    """从文件名中解析日期与标题（如存在）。"""
    stem = path.stem
    m = _DATE_TITLE_RE.match(stem)
    if not m:
        return None, stem
    return m.group("date"), m.group("title")


def _load_channel_index(channel_dir: Path) -> dict[str, dict]:
    """加载单个公众号目录下的 db.jsonl 索引。"""
    db_path = channel_dir / "db" / "db.jsonl"
    if not db_path.exists():
        return {}
    index: dict[str, dict] = {}
    for line in db_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = (row.get("msg_title") or row.get("activity_Name") or "").strip()
        if not title:
            continue
        index[_normalize_title(title)] = row
    return index


def _build_channel_indexes(wxhub_root: Path) -> dict[str, dict[str, dict]]:
    """为所有公众号目录构建标题索引。"""
    indexes: dict[str, dict[str, dict]] = {}
    for channel_dir in wxhub_root.iterdir():
        if channel_dir.is_dir():
            indexes[channel_dir.name] = _load_channel_index(channel_dir)
    return indexes


def build_channel_indexes(wxhub_root: Path) -> dict[str, dict[str, dict]]:
    """公共导出函数：构建并返回频道到标题索引的映射。"""
    return _build_channel_indexes(wxhub_root)


def iter_pdf_paths(wxhub_root: Path, glob_pattern: str) -> list[Path]:
    """根据给定的 glob 模式列出所有需要处理的 PDF 路径。"""
    return sorted(wxhub_root.glob(glob_pattern))


def build_metadata(
    *,
    pdf_path: Path,
    channel_indexes: dict[str, dict[str, dict]],
) -> WxhubMeta:
    channel = pdf_path.parent.parent.name
    date, title = _parse_title_from_filename(pdf_path)
    index = channel_indexes.get(channel, {})
    row = index.get(_normalize_title(title), {})

    return WxhubMeta(
        channel=channel,
        title=title,
        date=date or (row.get("createTime") or "")[:10] or None,
        url=row.get("url"),
        msg_link=row.get("msg_link"),
        create_time=row.get("createTime"),
    )


def load_pdf_documents(
    *,
    pdf_paths: Iterable[Path],
    channel_indexes: dict[str, dict[str, dict]],
) -> list[Document]:
    docs: list[Document] = []
    for pdf_path in pdf_paths:
        meta = build_metadata(pdf_path=pdf_path, channel_indexes=channel_indexes)
        loader = PyPDFLoader(str(pdf_path))
        for d in loader.load():
            d.metadata.update(
                {
                    "source": str(pdf_path),
                    "channel": meta.channel,
                    "title": meta.title,
                    "date": meta.date,
                    "url": meta.url,
                    "msg_link": meta.msg_link,
                    "create_time": meta.create_time,
                }
            )
            docs.append(d)
    return docs


def load_wxhub_documents(wxhub_root: Path, glob_pattern: str) -> list[Document]:
    """从 WXhub 根目录直接加载所有满足条件的 PDF 文档。"""
    channel_indexes = _build_channel_indexes(wxhub_root)
    pdf_paths = iter_pdf_paths(wxhub_root, glob_pattern)
    return load_pdf_documents(pdf_paths=pdf_paths, channel_indexes=channel_indexes)
