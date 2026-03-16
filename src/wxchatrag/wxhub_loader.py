from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


_TITLE_CLEAN_RE = re.compile(r"[\s\-_—–·\.，,。！？!?:：;；'\"\"''()（）\[\]【】<>《》/\\]+")
_DATE_TITLE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<title>.+)$")


def _clean_pdf_text(text: str) -> str:
    """清理 PDF 提取的文本，合并多余的换行。
    
    PyPDFLoader 提取的文本可能保留 PDF 的原始布局格式，导致每个词或短语单独成行。
    此函数将单个换行符合并为空格，但保留段落边界（空行或句子结束后的换行）。
    同时处理中文字符间不应有空格的情况。
    
    Args:
        text: 原始 PDF 文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 使用一个不太可能在文本中出现的字符串作为段落边界标记
    PARA_MARKER = "___PARAGRAPH_BOUNDARY_MARKER___"
    chinese_char_pattern = r'[\u4e00-\u9fff]'  # 中文字符范围
    chinese_punct_pattern = r'[，。！？：；、]'  # 中文标点
    
    # 步骤1: 标记段落边界
    # 1.1 空行标记为段落边界
    text = re.sub(r'\n\s*\n+', PARA_MARKER, text)
    
    # 1.2 句子结束标点后的换行，如果下一行以大写字母、中文或数字开头，标记为段落边界
    sentence_endings = r'[。！？.!?]'
    text = re.sub(
        rf'({sentence_endings})\s*\n+(?=[A-Z{chinese_char_pattern}0-9])',
        r'\1' + PARA_MARKER,
        text,
        flags=re.MULTILINE
    )
    
    # 步骤2: 智能处理换行
    # 2.1 中文字符间的换行：直接删除（不插入空格）
    # 匹配：中文字符 + 可选空格 + 换行 + 可选空格 + 中文字符
    text = re.sub(
        rf'({chinese_char_pattern})\s*\n+\s*({chinese_char_pattern})',
        r'\1\2',
        text
    )
    
    # 2.2 中文字符 + 换行 + 中文标点：删除换行和空格
    text = re.sub(
        rf'({chinese_char_pattern})\s*\n+\s*({chinese_punct_pattern})',
        r'\1\2',
        text
    )
    
    # 2.3 重复的单个中文字符（如"年\n年"）：合并为一个
    text = re.sub(
        rf'({chinese_char_pattern})\s*\n+\s*\1',
        r'\1',
        text
    )
    
    # 2.4 将剩余的单个换行符合并为空格
    text = re.sub(r'\n+', ' ', text)
    
    # 步骤3: 恢复段落边界（将特殊标记替换为双换行）
    text = text.replace(PARA_MARKER, '\n\n')
    # 合并多个连续的段落标记
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # 步骤4: 后处理 - 清理异常空格
    # 4.1 删除中文字符之间的空格
    text = re.sub(
        rf'({chinese_char_pattern})\s+({chinese_char_pattern})',
        r'\1\2',
        text
    )
    
    # 4.2 删除中文标点前的空格
    text = re.sub(
        rf'\s+({chinese_punct_pattern})',
        r'\1',
        text
    )
    
    # 4.3 合并连续重复的单个中文字符（处理可能遗留的情况）
    text = re.sub(
        rf'({chinese_char_pattern})\s+\1',
        r'\1',
        text
    )
    
    # 步骤5: 清理多余的空格（但保留段落分隔）
    # 将多个连续空格合并为单个空格
    text = re.sub(r' +', ' ', text)
    # 清理段落分隔前后的空格
    text = re.sub(r' *\n\n *', '\n\n', text)
    # 清理行首行尾空格
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    return text.strip()


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
            # 清理 PDF 文本，合并多余的换行
            d.page_content = _clean_pdf_text(d.page_content)
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
