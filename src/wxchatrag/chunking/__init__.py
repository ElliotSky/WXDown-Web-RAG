"""
文档切分模块（Chunking）

提供多种文档切分策略：语义切分、层次化切分等。
"""

from .chunk_strategy import ChunkStrategy
from .fixed_splitter import FixedSizeSplitter
from .hierarchical_splitter import HierarchicalSplitter
from .semantic_splitter import SemanticSplitter
from .sentence_splitter import SentenceSplitter
from .sliding_window_splitter import SlidingWindowSplitter

__all__ = [
    "ChunkStrategy",
    "FixedSizeSplitter",
    "SentenceSplitter",
    "SemanticSplitter",
    "HierarchicalSplitter",
    "SlidingWindowSplitter",
]

