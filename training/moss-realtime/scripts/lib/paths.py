"""Path helpers for the local distillation experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _detect_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "app" / "moss_api.py").is_file() and (parent / "docker-compose.yml").is_file():
            return parent
    raise RuntimeError(f"Could not find speaker repo root from {start}")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    train_dir: Path
    legacy_dir: Path
    finetune_dir: Path
    bench_dir: Path
    configs_dir: Path

    @classmethod
    def load(cls, train_dir: Path | None = None) -> Paths:
        here = Path(__file__).resolve()
        legacy_dir = here.parent.parent / "legacy"
        repo_root = Path(os.environ.get("SPEAKER_ROOT", _detect_repo_root(here)))
        train_dir = Path(
            train_dir
            or os.environ.get("MOSS_RT_TRAIN_DIR")
            or repo_root / "training" / "moss-realtime"
        ).resolve()
        return cls(
            repo_root=repo_root,
            train_dir=train_dir,
            legacy_dir=legacy_dir,
            finetune_dir=legacy_dir / "finetune",
            bench_dir=legacy_dir / "bench",
            configs_dir=train_dir / "configs",
        )

    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "SPEAKER_ROOT": str(self.repo_root),
            "MOSS_RT_TRAIN_DIR": str(self.train_dir),
            "LEGACY_DIR": str(self.legacy_dir),
            "PYTHONPATH": os.pathsep.join(
                p for p in (str(self.legacy_dir), os.environ.get("PYTHONPATH", "")) if p
            ),
        }
