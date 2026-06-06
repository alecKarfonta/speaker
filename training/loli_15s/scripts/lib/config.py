"""Load experiment.yaml for the local distillation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    voice_ref: str = "data/voices/loli/loli_15s.wav"
    teacher: str = "v15"
    corpus: str = "corpus/texts.jsonl"
    train_raw: str = "train_raw.jsonl"
    parallel: dict[str, Any] = field(default_factory=dict)
    sft: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    qc_v2: dict[str, Any] = field(default_factory=dict)
    sft_v2: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        if not path.is_file():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            voice_ref=data.get("voice_ref", cls.voice_ref),
            teacher=data.get("teacher", cls.teacher),
            corpus=data.get("corpus", cls.corpus),
            train_raw=data.get("train_raw", cls.train_raw),
            parallel=data.get("parallel", {}),
            sft=data.get("sft", {}),
            export=data.get("export", {}),
            qc=data.get("qc", {}),
            qc_v2=data.get("qc_v2", {}),
            sft_v2=data.get("sft_v2", {}),
        )

    def resolve(self, train_dir: Path) -> ExperimentConfig:
        """Return config with train-dir-relative paths expanded."""
        return ExperimentConfig(
            voice_ref=self.voice_ref,
            teacher=self.teacher,
            corpus=str(self._abs(train_dir, self.corpus)),
            train_raw=str(self._abs(train_dir, self.train_raw)),
            parallel=dict(self.parallel),
            sft=dict(self.sft),
            export={k: str(self._abs(train_dir, v)) for k, v in self.export.items()},
            qc={
                k: str(self._abs(train_dir, v)) if isinstance(v, str) and k.endswith("_dir") or k.endswith("_pruned") or k == "wav_dir" else v
                for k, v in self.qc.items()
            },
            qc_v2=dict(self.qc_v2),
            sft_v2={
                k: str(self._abs(train_dir, v)) if k == "merged" or k == "resume_checkpoint" else v
                for k, v in self.sft_v2.items()
            },
        )

    @staticmethod
    def _abs(train_dir: Path, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (train_dir / p).resolve()
