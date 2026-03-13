from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv


def _find_repo_root(start: Path) -> Path:
    """从任意模块文件出发，向上查找项目根目录。

    优先级：
    1. 包含 pyproject.toml 的目录
    2. 包含 .git 的目录
    3. 包含 configs/ 的目录
    找不到时，退回到当前起始目录。
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").is_dir():
            return p
        if (p / "configs").is_dir():
            return p
    return start


def _load_dotenv_if_present(repo_root: Path) -> None:
    """加载环境变量文件。
    
    从 configs/env 加载环境变量（统一管理配置）。
    """
    env_path = repo_root / "configs" / "env"
    if env_path.exists():
        load_dotenv(env_path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_str(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _resolve_wxhub_root(repo_root: Path, wxhub_root: Path) -> Path:
    """解析 WXHUB_ROOT。

    - 绝对路径：直接返回
    - 相对路径：基于项目根目录拼接

    不再内置对历史根目录 WXhub/ 的隐式兼容，
    如需使用旧路径，请显式配置 WXHUB_ROOT。
    """
    if wxhub_root.is_absolute():
        return wxhub_root
    return (repo_root / wxhub_root).resolve()


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    wxhub_root: Path
    pdf_subdir_name: str
    pdf_glob_pattern: str
    vector_store_dir: Path

    embedding_model_name: str
    chat_model_name: str
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str  # "fixed" | "sentence" | "semantic" | "hierarchical" | "sliding_window"
    semantic_threshold: float  # 语义切分阈值
    semantic_embedding_mode: str  # "local" | "api" - 语义切分embedding模式
    semantic_embedding_model_name: str  # 语义切分embedding模型名称
    semantic_sentence_split_mode: str  # "mixed" | "chinese" | "english" | "custom"
    semantic_sentence_split_regex: str  # 语义切分句子分隔正则
    retriever_k: int
    temperature: float

    # Hybrid Retrieval 配置
    retrieval_strategy: str  # "vector" | "bm25" | "hybrid"
    hybrid_alpha: float  # 向量检索权重（0-1）
    bm25_k: int  # BM25 单独检索数量
    vector_k: int  # 向量单独检索数量
    bm25_index_dir: Path  # BM25 索引目录

    # Rerank 配置
    enable_rerank: bool
    rerank_model_name: str
    rerank_top_n: int  # 初始检索数量
    rerank_top_k: int  # 重排序后保留数量
    rerank_batch_size: int
    rerank_device: str  # "cpu" | "cuda"
    models_cache_dir: Path  # 模型缓存目录

    @staticmethod
    def default_config() -> dict[str, Any]:
        """
        代码默认配置（最低优先级）。
        
        注意：
        - 个人化配置（API Key、路径等）应在 configs/env 文件中设置
        - 稳定的默认参数应在 configs/wxchatrag.json 中设置
        - 配置优先级：环境变量 > wxchatrag.json > 本函数返回值
        """
        return {
            # 路径配置（通常需要在 configs/env 中设置，这里提供默认值）
            "wxhub_root": "data/WXhub",
            "pdf_subdir_name": "pdf",
            "pdf_glob_pattern": "",
            "vector_store_dir": "storage/vector_store",
            "bm25_index_dir": "storage/bm25_index",
            "models_cache_dir": "storage/models",
            # 模型配置（通常需要在 configs/env 中设置，这里提供默认值）
            "embedding_model_name": "embedding-3",
            "chat_model_name": "deepseek-chat",
            # 稳定的默认参数（通常使用 wxchatrag.json 中的值）
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunk_strategy": "fixed",
            "semantic_threshold": 0.5,
            "semantic_embedding_mode": "local",  # 默认使用本地BGE模型
            "semantic_embedding_model_name": "BAAI/bge-small-zh-v1.5",  # 默认BGE模型
            "semantic_sentence_split_mode": "mixed",
            "semantic_sentence_split_regex": "",
            "retriever_k": 5,
            "temperature": 0.2,
            # Hybrid Retrieval 默认配置
            "retrieval_strategy": "hybrid",
            "hybrid_alpha": 0.7,
            "bm25_k": 20,
            "vector_k": 20,
            # Rerank 默认配置
            "enable_rerank": True,
            "rerank_model_name": "BAAI/bge-reranker-base",
            "rerank_top_n": 20,
            "rerank_top_k": 5,
            "rerank_batch_size": 16,
            "rerank_device": "cpu",
        }

    @classmethod
    def from_sources(
        cls,
        *,
        repo_root: Path,
        file_cfg: Mapping[str, Any],
        env: Mapping[str, str],
    ) -> "Settings":
        cfg = {**cls.default_config(), **dict(file_cfg)}

        wxhub_root = Path(_as_str(env.get("WXHUB_ROOT"), str(cfg.get("wxhub_root"))))
        vector_store_dir = Path(
            _as_str(env.get("VECTOR_STORE_DIR"), str(cfg.get("vector_store_dir")))
        )

        pdf_subdir_name = _as_str(env.get("PDF_SUBDIR_NAME"), str(cfg.get("pdf_subdir_name")))
        pdf_glob_pattern = _as_str(
            env.get("PDF_GLOB_PATTERN"),
            str(cfg.get("pdf_glob_pattern") or f"*/{pdf_subdir_name}/*.pdf"),
        )

        embedding_model_name = _as_str(
            env.get("EMBEDDING_MODEL_NAME"), str(cfg.get("embedding_model_name"))
        )
        chat_model_name = _as_str(env.get("CHAT_MODEL_NAME"), str(cfg.get("chat_model_name")))

        chunk_size = _as_int(env.get("CHUNK_SIZE", cfg.get("chunk_size")), 1000)
        chunk_overlap = _as_int(env.get("CHUNK_OVERLAP", cfg.get("chunk_overlap")), 200)
        chunk_strategy = _as_str(env.get("CHUNK_STRATEGY", cfg.get("chunk_strategy")), "fixed")
        semantic_threshold = _as_float(
            env.get("SEMANTIC_THRESHOLD", cfg.get("semantic_threshold")), 0.5
        )
        semantic_embedding_mode = _as_str(
            env.get("SEMANTIC_EMBEDDING_MODE", cfg.get("semantic_embedding_mode")), "local"
        )
        semantic_embedding_model_name = _as_str(
            env.get("SEMANTIC_EMBEDDING_MODEL_NAME", cfg.get("semantic_embedding_model_name")),
            "BAAI/bge-small-zh-v1.5",
        )
        semantic_sentence_split_mode = _as_str(
            env.get(
                "SEMANTIC_SENTENCE_SPLIT_MODE",
                cfg.get("semantic_sentence_split_mode"),
            ),
            "mixed",
        )
        semantic_sentence_split_regex = _as_str(
            env.get(
                "SEMANTIC_SENTENCE_SPLIT_REGEX",
                cfg.get("semantic_sentence_split_regex"),
            ),
            "",
        )
        retriever_k = _as_int(env.get("RETRIEVER_K", cfg.get("retriever_k")), 5)
        temperature = _as_float(env.get("TEMPERATURE", cfg.get("temperature")), 0.2)

        # Hybrid Retrieval 配置
        retrieval_strategy = _as_str(
            env.get("RETRIEVAL_STRATEGY", cfg.get("retrieval_strategy")), "hybrid"
        )
        hybrid_alpha = _as_float(env.get("HYBRID_ALPHA", cfg.get("hybrid_alpha")), 0.7)
        bm25_k = _as_int(env.get("BM25_K", cfg.get("bm25_k")), 20)
        vector_k = _as_int(env.get("VECTOR_K", cfg.get("vector_k")), 20)
        bm25_index_dir = Path(
            _as_str(env.get("BM25_INDEX_DIR"), str(cfg.get("bm25_index_dir", "storage/bm25_index")))
        )

        # Rerank 配置
        enable_rerank = env.get("ENABLE_RERANK", str(cfg.get("enable_rerank", True))).lower() in (
            "true",
            "1",
            "yes",
        )
        rerank_model_name = _as_str(
            env.get("RERANK_MODEL_NAME", cfg.get("rerank_model_name")),
            "BAAI/bge-reranker-base",
        )
        rerank_top_n = _as_int(env.get("RERANK_TOP_N", cfg.get("rerank_top_n")), 20)
        rerank_top_k = _as_int(env.get("RERANK_TOP_K", cfg.get("rerank_top_k")), 5)
        rerank_batch_size = _as_int(env.get("RERANK_BATCH_SIZE", cfg.get("rerank_batch_size")), 16)
        rerank_device = _as_str(env.get("RERANK_DEVICE", cfg.get("rerank_device")), "cpu")
        models_cache_dir = Path(
            _as_str(
                env.get("MODELS_CACHE_DIR"),
                str(cfg.get("models_cache_dir", "storage/models")),
            )
        )

        wxhub_root = _resolve_wxhub_root(repo_root, wxhub_root)
        if not vector_store_dir.is_absolute():
            vector_store_dir = (repo_root / vector_store_dir).resolve()
        if not bm25_index_dir.is_absolute():
            bm25_index_dir = (repo_root / bm25_index_dir).resolve()
        if not models_cache_dir.is_absolute():
            models_cache_dir = (repo_root / models_cache_dir).resolve()

        return cls(
            repo_root=repo_root,
            wxhub_root=wxhub_root,
            pdf_subdir_name=pdf_subdir_name,
            pdf_glob_pattern=pdf_glob_pattern,
            vector_store_dir=vector_store_dir,
            embedding_model_name=embedding_model_name,
            chat_model_name=chat_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            semantic_threshold=semantic_threshold,
            semantic_embedding_mode=semantic_embedding_mode,
            semantic_embedding_model_name=semantic_embedding_model_name,
            semantic_sentence_split_mode=semantic_sentence_split_mode,
            semantic_sentence_split_regex=semantic_sentence_split_regex,
            retriever_k=retriever_k,
            temperature=temperature,
            retrieval_strategy=retrieval_strategy,
            hybrid_alpha=hybrid_alpha,
            bm25_k=bm25_k,
            vector_k=vector_k,
            bm25_index_dir=bm25_index_dir,
            enable_rerank=enable_rerank,
            rerank_model_name=rerank_model_name,
            rerank_top_n=rerank_top_n,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
            rerank_device=rerank_device,
            models_cache_dir=models_cache_dir,
        )


def _ensure_openai_env() -> None:
    """
    - 请显式通过 configs/env 配置：
        * EMBEDDING_API_KEY / EMBEDDING_BASE_URL 作为嵌入模型 API
        * CHAT_API_KEY / CHAT_BASE_URL 作为对话模型 API
    """
    return None


def get_embedding_api_config() -> tuple[str | None, str | None]:
    """
    获取嵌入模型 API 配置。
    
    使用环境变量：
    - EMBEDDING_API_KEY / EMBEDDING_BASE_URL
    
    Returns:
        (api_key, base_url) 元组，如果未设置则返回 (None, None)
    """
    api_key = os.getenv("EMBEDDING_API_KEY") or None
    base_url = os.getenv("EMBEDDING_BASE_URL") or None
    return (api_key, base_url)


def get_chat_api_config() -> tuple[str | None, str | None]:
    """
    获取对话模型 API 配置。
    
    使用环境变量：
    - CHAT_API_KEY / CHAT_BASE_URL
    
    Returns:
        (api_key, base_url) 元组，如果未设置则返回 (None, None)
    """
    api_key = os.getenv("CHAT_API_KEY") or None
    base_url = os.getenv("CHAT_BASE_URL") or None
    return (api_key, base_url)


def get_config_path(repo_root: Path) -> Path:
    env_path = (os.getenv("WXCHATRAG_CONFIG_PATH") or "").strip()
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else (repo_root / p)
    return repo_root / "configs" / "wxchatrag.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    here = Path(__file__).resolve()
    repo_root = _find_repo_root(here)
    _load_dotenv_if_present(repo_root)
    _ensure_openai_env()

    cfg_path = get_config_path(repo_root)
    file_cfg: Mapping[str, Any] = {}
    if cfg_path.exists():
        file_cfg = _read_json(cfg_path)

    return Settings.from_sources(repo_root=repo_root, file_cfg=file_cfg, env=os.environ)
