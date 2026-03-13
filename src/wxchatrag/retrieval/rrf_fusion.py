"""
Reciprocal Rank Fusion (RRF) 算法实现

用于融合多个检索结果列表，无需调参，简单有效。
"""

from __future__ import annotations

import hashlib
from typing import Any


def _get_document_key(doc: Any) -> str:
    """
    为文档生成唯一标识符（用于 RRF 融合时的去重）。

    Args:
        doc: Document 对象或任何对象

    Returns:
        唯一标识符字符串
    """
    from langchain_core.documents import Document

    if isinstance(doc, Document):
        # 使用 page_content 和 metadata 生成唯一标识
        content = doc.page_content
        metadata_str = str(sorted(doc.metadata.items())) if doc.metadata else ""
        key_string = f"{content}|{metadata_str}"
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()
    else:
        # 对于其他类型，使用字符串表示
        return str(hash(str(doc)))


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[Any, float]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """
    使用 RRF 算法融合多个排序列表。

    Args:
        *ranked_lists: 多个排序列表，每个列表为 [(item, score), ...]
        k: RRF 常数，通常为 60

    Returns:
        融合后的排序列表，按 RRF 分数降序排列
    """
    # 使用文档的唯一标识符作为键，而不是文档对象本身
    scores: dict[str, float] = {}
    # 存储标识符到原始对象的映射
    key_to_item: dict[str, Any] = {}

    for ranked_list in ranked_lists:
        for rank, (item, _) in enumerate(ranked_list, start=1):
            item_key = _get_document_key(item)
            rrf_score = 1.0 / (k + rank)
            scores[item_key] = scores.get(item_key, 0.0) + rrf_score
            # 保存第一个遇到的文档对象（用于返回）
            if item_key not in key_to_item:
                key_to_item[item_key] = item

    # 按分数降序排序
    fused_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 恢复为原始对象和分数的元组列表
    fused = [(key_to_item[key], score) for key, score in fused_keys]
    return fused

