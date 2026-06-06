"""Descriptive teacher WAV filenames from corpus row metadata."""

from __future__ import annotations

import re
from pathlib import Path


def _slug(value: str, *, max_len: int = 24) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return (s[:max_len] if s else "misc")


def wav_filename_for_row(row: dict, *, turn_idx: int | None = None) -> str:
    """
    Encode gap_category (emotion bucket), style, and length in the filename.

    Example: loli3_st_60000__emotion__excited__short.wav
    """
    cid = row["id"]
    gap = _slug(row.get("gap_category") or row.get("type") or "misc")
    style = _slug(row.get("style") or "cheerful")
    length = _slug(row.get("length") or "med", max_len=12)
    core = f"{cid}__{gap}__{style}__{length}"
    if turn_idx is not None:
        core += f"__a{turn_idx:02d}"
    return f"{core}.wav"


def corpus_id_from_wav_name(wav_name: str) -> str:
    """Strip descriptive suffix to recover corpus id (loli3_st_60000)."""
    stem = Path(wav_name).stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    if "_a" in stem:
        base, suffix = stem.rsplit("_a", 1)
        if suffix.isdigit():
            return base
    return stem
