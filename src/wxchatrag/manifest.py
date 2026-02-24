from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileState:
    """单个文件在 manifest 中记录的状态信息。"""

    path: str
    size: int
    mtime: float


def build_file_state(path: Path) -> FileState:
    """根据真实文件构造 FileState。"""
    stat = path.stat()
    return FileState(path=str(path), size=int(stat.st_size), mtime=float(stat.st_mtime))


def load_manifest(path: Path) -> dict[str, FileState]:
    """从 JSON 文件加载历史文件状态清单。"""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    data: dict[str, FileState] = {}
    for k, v in raw.items():
        data[k] = FileState(path=k, size=int(v["size"]), mtime=float(v["mtime"]))
    return data


def save_manifest(path: Path, states: Iterable[FileState]) -> None:
    """将当前文件状态列表保存到 manifest JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {s.path: {"size": s.size, "mtime": s.mtime} for s in states}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def select_changed(paths: Iterable[Path], manifest: dict[str, FileState]) -> list[Path]:
    """根据 manifest 选择出有新增或变更的文件路径。"""
    changed: list[Path] = []
    for p in paths:
        key = str(p)
        state = build_file_state(p)
        old = manifest.get(key)
        if old is None or old.size != state.size or old.mtime != state.mtime:
            changed.append(p)
    return changed

