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
    env_path = repo_root / ".env"
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
    retriever_k: int
    temperature: float

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "wxhub_root": "data/WXhub",
            "pdf_subdir_name": "pdf",
            "pdf_glob_pattern": "",
            "vector_store_dir": "storage/vector_store",
            "embedding_model_name": "text-embedding-3-large",
            "chat_model_name": "deepseek-chat",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retriever_k": 5,
            "temperature": 0.2,
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
        retriever_k = _as_int(env.get("RETRIEVER_K", cfg.get("retriever_k")), 5)
        temperature = _as_float(env.get("TEMPERATURE", cfg.get("temperature")), 0.2)

        wxhub_root = _resolve_wxhub_root(repo_root, wxhub_root)
        if not vector_store_dir.is_absolute():
            vector_store_dir = (repo_root / vector_store_dir).resolve()

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
            retriever_k=retriever_k,
            temperature=temperature,
        )


def _ensure_openai_env() -> None:
    """
    - 请显式通过 .env 配置：
        * OPENAI_API_KEY / OPENAI_BASE_URL 作为 Embedding 所用的 OpenAI 兼容接口（例如 GLM）。
        * DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 作为聊天模型 DeepSeek 的接口。
    """
    # 为了避免意外覆盖用户为 Embedding（如 GLM）配置的 OPENAI_*，这里不做任何自动回填。
    return None


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
