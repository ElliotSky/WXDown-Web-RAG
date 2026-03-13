"""
BGE Embeddings 本地实现

使用北京智源研究院发布的 BGE (BAAI General Embedding) 系列模型。
通过 sentence-transformers 库加载本地部署的模型。
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class BGEEmbeddings(Embeddings):
    """BGE Embeddings 实现，使用本地部署的 BGE 模型。
    
    支持北京智源研究院发布的 BGE 系列模型，如：
    - BAAI/bge-large-zh-v1.5
    - BAAI/bge-base-zh-v1.5
    - BAAI/bge-small-zh-v1.5
    - BAAI/bge-large-en-v1.5
    - BAAI/bge-base-en-v1.5
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_kwargs: dict | None = None,
        encode_kwargs: dict | None = None,
        cache_folder: str | None = None,
    ):
        """
        初始化 BGE Embeddings。

        Args:
            model_name: 模型名称，默认使用 BAAI/bge-small-zh-v1.5
            model_kwargs: 传递给 SentenceTransformer 的模型参数
            encode_kwargs: 传递给 encode 方法的参数（如 normalize_embeddings）
            cache_folder: 模型缓存目录，如果为 None 则使用默认缓存目录
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        
        # 设置缓存目录
        if cache_folder:
            self.model_kwargs["cache_folder"] = cache_folder
        elif "MODELS_CACHE_DIR" in os.environ:
            self.model_kwargs["cache_folder"] = os.environ["MODELS_CACHE_DIR"]
        
        # 加载模型
        self.client = SentenceTransformer(
            model_name,
            **self.model_kwargs
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表。

        Args:
            texts: 文本字符串列表

        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts 必须是列表，当前类型: {type(texts)}")
        
        if not texts:
            return []
        
        # 使用 sentence-transformers 进行批量编码
        embeddings = self.client.encode(
            texts,
            **self.encode_kwargs
        )
        
        # 转换为列表格式
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本。

        Args:
            text: 查询文本

        Returns:
            向量（浮点数列表）
        """
        if not isinstance(text, str):
            raise TypeError(f"text 必须是字符串，当前类型: {type(text)}")
        
        # 使用 sentence-transformers 进行编码
        embedding = self.client.encode(
            [text],
            **self.encode_kwargs
        )
        
        # 返回第一个（也是唯一的）向量
        return embedding[0].tolist()

