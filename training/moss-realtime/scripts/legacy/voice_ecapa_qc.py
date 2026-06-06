#!/usr/bin/env python3
"""ECAPA speaker-embedding QC (cos ref / cos teacher) for teacher WAV datasets."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
_LEGACY = Path(__file__).resolve().parent
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MIN_COS = 0.5


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def token_set(text: str) -> set[str]:
    from build_realtime_finetune_dataset import normalize

    return set(normalize(text).split())


def build_teacher_index(
    train_raw: Path,
    teacher_root: Path,
    *,
    root: Path = ROOT,
) -> list[tuple[str, Path, str, set[str]]]:
    rows: list[tuple[str, Path, str, set[str]]] = []
    if not train_raw.is_file():
        return rows
    for line in train_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = row.get("id", "")
        convs = row.get("conversations") or []
        if not convs:
            continue
        text = str(convs[0].get("text", ""))
        wav_rel = str(convs[0].get("wav", ""))
        wav_path = (root / wav_rel) if not Path(wav_rel).is_absolute() else Path(wav_rel)
        if not wav_path.is_file():
            stem = Path(wav_rel).name
            for alt in (
                teacher_root / stem,
                teacher_root / f"{tid.replace('_v15', '')}.wav",
            ):
                if alt.is_file():
                    wav_path = alt
                    break
            else:
                continue
        rows.append((tid, wav_path, text, token_set(text)))
    return rows


def find_teacher_match(
    ref_text: str,
    index: list[tuple[str, Path, str, set[str]]],
    *,
    exclude_wav: str | None = None,
    min_recall: float = 0.55,
) -> tuple[Path, str, float] | None:
    if not ref_text.strip() or not index:
        return None
    eval_tokens = token_set(ref_text)
    if not eval_tokens:
        return None
    best: tuple[float, str, Path, str] | None = None
    for tid, wav_path, text, teach_tokens in index:
        if exclude_wav and wav_path.name == exclude_wav:
            continue
        if not teach_tokens:
            continue
        overlap = len(eval_tokens & teach_tokens)
        recall = overlap / len(eval_tokens)
        precision = overlap / len(teach_tokens)
        score = recall * 0.85 + precision * 0.15
        if recall < min_recall:
            continue
        if best is None or score > best[0] or (score == best[0] and len(text) < len(best[3])):
            best = (score, tid, wav_path, text)
    if best is None:
        for tid, wav_path, text, teach_tokens in index:
            if exclude_wav and wav_path.name == exclude_wav:
                continue
            overlap = len(eval_tokens & teach_tokens)
            recall = overlap / len(eval_tokens) if eval_tokens else 0.0
            if best is None or recall > best[0]:
                best = (recall, tid, wav_path, text)
        if best is None or best[0] < 0.25:
            return None
    return best[2], best[1], float(best[0])


class EcapaEncoder:
    def __init__(self, device: str, cache_dir: Path) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self._lock = threading.Lock()
        self.device = device
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            savedir=str(cache_dir),
            run_opts={"device": device},
        )

    def embed(self, wav_path: Path) -> np.ndarray:
        with self._lock:
            signal = self.model.load_audio(str(wav_path))
            emb = self.model.encode_batch(signal)
            return emb.squeeze().detach().cpu().numpy()


class VoiceQcGate:
    """Score teacher WAVs vs enrollment ref and corpus teacher pool (bench-aligned)."""

    def __init__(
        self,
        *,
        ref_wav: Path,
        teacher_index: list[tuple[str, Path, str, set[str]]],
        device: str,
        cache_dir: Path,
        min_cos_ref: float = DEFAULT_MIN_COS,
        min_cos_teacher: float = DEFAULT_MIN_COS,
        min_teacher_recall: float = 0.55,
        use_prototype: bool = True,
        prototype_max: int = 256,
    ) -> None:
        self.min_cos_ref = min_cos_ref
        self.min_cos_teacher = min_cos_teacher
        self.min_teacher_recall = min_teacher_recall
        self.teacher_index = teacher_index
        self.encoder = EcapaEncoder(device, cache_dir)
        self.ref_emb = self.encoder.embed(ref_wav)
        self._teacher_emb_cache: dict[str, np.ndarray] = {}
        self.prototype_emb: np.ndarray | None = None
        if use_prototype and teacher_index:
            self.prototype_emb = self._build_prototype(teacher_index, prototype_max)

    def _teacher_emb(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._teacher_emb_cache:
            self._teacher_emb_cache[key] = self.encoder.embed(path)
        return self._teacher_emb_cache[key]

    def _build_prototype(
        self,
        index: list[tuple[str, Path, str, set[str]]],
        max_n: int,
    ) -> np.ndarray:
        paths = [p for _, p, _, _ in index[:max_n]]
        embs = [self._teacher_emb(p) for p in paths]
        return np.median(np.stack(embs, axis=0), axis=0)

    def score_file(
        self,
        wav_path: Path,
        ref_text: str,
    ) -> tuple[float, float | None, str | None]:
        emb = self.encoder.embed(wav_path)
        cos_ref = cos_sim(emb, self.ref_emb)

        cos_teacher: float | None = None
        teacher_note: str | None = None
        match = find_teacher_match(
            ref_text,
            self.teacher_index,
            exclude_wav=wav_path.name,
            min_recall=self.min_teacher_recall,
        )
        if match:
            t_path, tid, _ = match
            cos_teacher = cos_sim(emb, self._teacher_emb(t_path))
            teacher_note = f"matched:{tid}"
        elif self.prototype_emb is not None:
            cos_teacher = cos_sim(emb, self.prototype_emb)
            teacher_note = "prototype"

        return cos_ref, cos_teacher, teacher_note

    def check(
        self,
        wav_path: Path,
        ref_text: str,
    ) -> tuple[float, float | None, str | None, list[str]]:
        cos_ref, cos_teacher, note = self.score_file(wav_path, ref_text)
        reasons: list[str] = []
        if cos_ref < self.min_cos_ref:
            reasons.append(f"low_cos_ref:{cos_ref:.3f}")
        if cos_teacher is not None and cos_teacher < self.min_cos_teacher:
            reasons.append(f"low_cos_teacher:{cos_teacher:.3f}")
        return cos_ref, cos_teacher, note, reasons


def default_device() -> str:
    return os.environ.get("VOICE_QC_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
