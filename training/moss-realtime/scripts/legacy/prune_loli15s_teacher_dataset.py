#!/usr/bin/env python3
"""
QC + tail-prune MOSS v1.5 teacher WAVs for loli_15s SFT.

Uses STT with word timestamps to trim garbage after the last spoken word,
then runs duration / level / WER / repetition checks. Emits JSON + HTML reports.

Requires STT at STT_API (OpenAI-compatible whisper, e.g. :8603) with verbose_json
+ word timestamps, or falls back to local openai-whisper if installed.

Examples:
  # Report only (safe default)
  python3 scripts/prune_loli15s_teacher_dataset.py

  # Write pruned WAVs + quarantine rejects
  python3 scripts/prune_loli15s_teacher_dataset.py --apply \\
    --wav-dir training/loli_15s/wavs/v15 \\
    --out-dir training/loli_15s/wavs/v15_pruned \\
    --quarantine-dir training/loli_15s/wavs/v15_quarantine

  # In-place trim (backs up to .bak before overwrite)
  python3 scripts/prune_loli15s_teacher_dataset.py --apply --in-place
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

import os
import sys

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
_LEGACY = Path(__file__).resolve().parent
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from build_realtime_finetune_dataset import (  # noqa: E402
    load_corpus,
    load_id_set,
    normalize,
    word_error_rate,
)
from corpus_wav_names import corpus_id_from_wav_name, wav_filename_for_row  # noqa: E402
from loli15s_wav_analysis import (  # noqa: E402
    NormProfile,
    WaveformFeatures,
    extract_waveform_features,
    fit_norm_profile,
    load_profile,
    save_profile,
    score_waveform_outlier,
)

STT_API = os.environ.get("STT_API", "http://localhost:8603/v1/audio/transcriptions")
STT_MODEL = os.environ.get("STT_MODEL", "base")
REF_PAD_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


@dataclass
class WordStamp:
    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class ClipResult:
    wav: str
    corpus_id: str
    ref_text: str
    action: str  # pass | trim | quarantine | stt_pending | skip
    reasons: list[str] = field(default_factory=list)
    orig_duration_s: float = 0.0
    new_duration_s: float = 0.0
    trim_removed_s: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    tail_rms_ratio: float = 0.0
    wer: float | None = None
    hyp_text: str = ""
    n_words: int = 0
    last_word: str = ""
    last_word_end_s: float = 0.0
    ref_last_word: str = ""
    hyp_last_word: str = ""
    last_word_match: bool = False
    likely_cutoff: bool = False
    missing_tail_words: int = 0
    tail_gap_s: float | None = None
    out_path: str = ""
    wav_features: dict = field(default_factory=dict)
    wav_outlier_score: float = 0.0
    wav_outlier_flags: list[str] = field(default_factory=list)
    cos_ref: float | None = None
    cos_teacher: float | None = None
    voice_teacher_match: str = ""


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def rms_energy(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def tail_rms_ratio(x: np.ndarray, sr: int, cut_s: float, tail_window_s: float = 0.5) -> float:
    """RMS in last tail_window after cut vs full-clip RMS."""
    full = rms_energy(x)
    if full < 1e-8:
        return 0.0
    start = int(cut_s * sr)
    tail = x[start : start + int(tail_window_s * sr)]
    if len(tail) < int(0.05 * sr):
        return 0.0
    return rms_energy(tail) / full


def repetition_flag(text: str, min_phrase_words: int = 3, min_repeats: int = 3) -> bool:
    words = normalize(text).split()
    if len(words) < min_phrase_words * min_repeats:
        return False
    for n in range(min_phrase_words, min(8, len(words) // min_repeats + 1)):
        for i in range(len(words) - n * min_repeats + 1):
            phrase = tuple(words[i : i + n])
            count = 0
            j = i
            while j + n <= len(words) and tuple(words[j : j + n]) == phrase:
                count += 1
                j += n
            if count >= min_repeats:
                return True
    return False


def build_wav_index(corpus_path: Path, train_ids_path: Path | None) -> dict[str, dict]:
    corpus = load_corpus(corpus_path)
    train_ids = load_id_set(train_ids_path) if train_ids_path else None
    index: dict[str, dict] = {}
    for row in corpus:
        cid = row["id"]
        if train_ids is not None and cid not in train_ids:
            continue
        base = {"corpus_id": cid, "style": row.get("style"), "type": row.get("type")}
        if row.get("type") == "single" or "turns" not in row:
            name = wav_filename_for_row(row)
            index[name] = {**base, "text": row["text"], "gap_category": row.get("gap_category")}
        else:
            for ti, turn in enumerate(row["turns"]):
                if turn["role"] != "assistant":
                    continue
                turn_row = {**row, "style": turn.get("style", row.get("style")), "text": turn["text"]}
                name = wav_filename_for_row(turn_row, turn_idx=ti)
                index[name] = {**base, "text": turn["text"], "turn_idx": ti}
    return index


def parse_stt_words(payload: dict) -> tuple[str, list[WordStamp]]:
    text = (payload.get("text") or "").strip()
    words: list[WordStamp] = []
    if "words" in payload and isinstance(payload["words"], list):
        for w in payload["words"]:
            word = str(w.get("word", "")).strip()
            if not word:
                continue
            words.append(
                WordStamp(
                    word=word,
                    start=float(w.get("start", 0)),
                    end=float(w.get("end", w.get("start", 0))),
                    probability=w.get("probability"),
                )
            )
        return text, words
    for seg in payload.get("segments") or []:
        if "words" in seg:
            for w in seg["words"]:
                word = str(w.get("word", "")).strip()
                if not word:
                    continue
                words.append(
                    WordStamp(
                        word=word,
                        start=float(w.get("start", 0)),
                        end=float(w.get("end", w.get("start", 0))),
                        probability=w.get("probability"),
                    )
                )
        else:
            seg_text = str(seg.get("text", "")).strip()
            if seg_text:
                words.append(
                    WordStamp(
                        word=seg_text,
                        start=float(seg.get("start", 0)),
                        end=float(seg.get("end", seg.get("start", 0))),
                    )
                )
    if not text and words:
        text = " ".join(w.word for w in words)
    return text, words


_whisper_model = None


def transcribe_local_whisper(wav_path: Path, model_name: str) -> tuple[str, list[WordStamp]]:
    global _whisper_model
    import whisper  # type: ignore

    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    result = _whisper_model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=True,
        verbose=False,
    )
    text = (result.get("text") or "").strip()
    words: list[WordStamp] = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            word = str(w.get("word", "")).strip()
            if not word:
                continue
            words.append(
                WordStamp(
                    word=word,
                    start=float(w["start"]),
                    end=float(w["end"]),
                    probability=w.get("probability"),
                )
            )
    return text, words


def transcribe_api(wav_path: Path, api: str, model: str) -> tuple[str, list[WordStamp]]:
    base_data = {
        "model": model,
        "language": "en",
        "response_format": "verbose_json",
    }
    data_variants = [
        {**base_data, "timestamp_granularities[]": "word"},
        {**base_data, "timestamp_granularities": "word"},
        base_data,
    ]
    last_err = ""
    for payload in data_variants:
        with wav_path.open("rb") as f:
            files = {"file": (wav_path.name, f, "audio/wav")}
            try:
                r = requests.post(api, files=files, data=payload, timeout=180)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                return parse_stt_words(r.json())
            except requests.RequestException as exc:
                last_err = str(exc)
    raise RuntimeError(last_err or "STT request failed")


def _transient_stt_error(detail: str) -> bool:
    s = detail.lower()
    return any(
        x in s
        for x in (
            "connection refused",
            "connection aborted",
            "max retries exceeded",
            "timed out",
            "timeout",
            "remotedisconnected",
        )
    )


def transcribe_with_words(
    wav_path: Path,
    api: str,
    model: str,
    allow_local: bool,
    max_retries: int = 1,
) -> tuple[str, list[WordStamp]]:
    last_exc: Exception | None = None
    retries = max(1, max_retries)
    for attempt in range(retries):
        try:
            return transcribe_api(wav_path, api, model)
        except Exception as api_exc:
            last_exc = api_exc
            if attempt >= retries - 1 or not _transient_stt_error(str(api_exc)):
                break
            time.sleep(min(30, 3 * (attempt + 1)))
    assert last_exc is not None
    if not allow_local:
        raise last_exc
    try:
        text, words = transcribe_local_whisper(wav_path, model)
        return text, words
    except Exception as local_exc:
        raise RuntimeError(f"API: {last_exc}; local: {local_exc}") from local_exc


def analyze_completeness(
    ref_text: str,
    hyp: str,
    audio_s: float,
    last_word_end_s: float | None,
    *,
    tail_gap_fail_s: float = 0.30,
) -> dict:
    """Detect teacher clips that end before the reference last word (MOSS cutoff)."""
    ref_w = normalize(ref_text).split()
    hyp_w = normalize(hyp).split()
    ref_last = ref_w[-1] if ref_w else ""
    hyp_last = hyp_w[-1] if hyp_w else ""
    last_word_match = bool(ref_last and hyp_last and ref_last == hyp_last)
    missing_tail_words = max(0, len(ref_w) - len(hyp_w))
    tail_gap_s = None
    likely_cutoff = False
    if last_word_end_s is not None and audio_s > 0:
        tail_gap_s = round(max(0.0, audio_s - last_word_end_s), 3)
        likely_cutoff = tail_gap_s < tail_gap_fail_s and not last_word_match
    return {
        "ref_last_word": ref_last,
        "hyp_last_word": hyp_last,
        "last_word_match": last_word_match,
        "likely_cutoff": likely_cutoff,
        "missing_tail_words": missing_tail_words,
        "tail_gap_s": tail_gap_s,
    }


def meaningful_words(words: list[WordStamp]) -> list[WordStamp]:
    out: list[WordStamp] = []
    for w in words:
        core = re.sub(r"[^\w']", "", w.word)
        if len(core) >= 1:
            out.append(w)
    return out


def compute_cut_time(
    words: list[WordStamp],
    buffer_s: float,
    max_end_s: float,
) -> tuple[float, str, int]:
    mw = meaningful_words(words)
    if not mw:
        return 0.0, "", 0
    last = mw[-1]
    cut = min(last.end + buffer_s, max_end_s)
    return cut, last.word, len(mw)


def trim_audio(x: np.ndarray, sr: int, cut_s: float, min_keep_s: float) -> np.ndarray:
    n = int(cut_s * sr)
    n = max(n, int(min_keep_s * sr))
    n = min(n, len(x))
    return x[:n].copy()


def _reason_tag(r: str) -> str:
    return r.split(":")[0]


def _is_hard_reason(
    r: str,
    *,
    trim_only: bool = False,
    wer_quarantine_threshold: float | None = None,
) -> bool:
    if trim_only:
        if wer_quarantine_threshold is not None:
            if r.startswith("high_wer"):
                return True
        tag = _reason_tag(r)
        if tag in {
            "low_cos_ref", "low_cos_teacher", "likely_cutoff",
            "last_word_mismatch", "missing_tail_words",
        }:
            return True
        return tag in {"file_too_small", "low_peak"} or tag.startswith("stt_failed")
    if r.startswith("high_wer"):
        return True
    tag = _reason_tag(r)
    hard = {
        "file_too_small", "low_peak", "too_short", "too_long", "no_stt_words",
        "stt_failed", "low_confidence", "repetition", "still_too_long_after_trim",
        "noisy_tail_remaining", "large_tail_removed", "low_cos_ref", "low_cos_teacher",
        "likely_cutoff", "missing_tail_words",
    }
    if tag in hard:
        return True
    return tag.startswith("wav_artifact") or tag.startswith("wav_outlier")


def evaluate_clip(
    wav_path: Path,
    meta: dict,
    *,
    api: str,
    model: str,
    allow_local: bool,
    buffer_s: float,
    min_keep_s: float,
    min_dur: float,
    max_dur: float,
    max_wer: float,
    min_peak: float,
    max_tail_removed_s: float,
    min_word_prob: float | None,
    wav_profile: NormProfile | None,
    wav_z_threshold: float,
    wav_score_threshold: float,
    skip_stt_if_wav_outlier: bool,
    pre_feat: WaveformFeatures | None = None,
    wav_only: bool = False,
    trim_only: bool = False,
    wer_quarantine_threshold: float | None = None,
    voice_gate: object | None = None,
    quarantine_cutoff: bool = True,
    tail_gap_fail_s: float = 0.30,
    min_missing_tail_words: int = 2,
    defer_stt_fail: bool = False,
    stt_max_retries: int = 5,
) -> tuple[ClipResult, np.ndarray, int]:
    name = wav_path.name
    ref = REF_PAD_SUFFIX.sub("", meta.get("text", ""))
    res = ClipResult(
        wav=name,
        corpus_id=meta.get("corpus_id", ""),
        ref_text=ref,
        action="quarantine",
    )

    if wav_path.stat().st_size < 1024:
        res.reasons.append("file_too_small")
        return res, np.array([], dtype=np.float32), 24000

    x, sr = load_audio_mono(wav_path)
    res.orig_duration_s = len(x) / sr if sr else 0.0
    res.peak = float(np.max(np.abs(x))) if len(x) else 0.0
    res.rms = rms_energy(x)

    if not trim_only:
        feat = pre_feat if pre_feat is not None else extract_waveform_features(x, sr)
        res.wav_features = feat.as_dict()
        if wav_profile is not None and wav_profile.n_fit > 0:
            score, wflags = score_waveform_outlier(
                feat, wav_profile,
                z_threshold=wav_z_threshold,
                score_threshold=wav_score_threshold,
            )
            res.wav_outlier_score = score
            res.wav_outlier_flags = wflags
            for f in wflags:
                if f.startswith("wav_artifact_") or f.startswith("wav_outlier"):
                    res.reasons.append(f)

        if res.peak < min_peak:
            res.reasons.append("low_peak")
        if res.orig_duration_s < min_dur:
            res.reasons.append("too_short")
        if res.orig_duration_s > max_dur:
            res.reasons.append("too_long")
    elif res.peak < min_peak:
        res.reasons.append("low_peak")

    if wav_only:
        trimmed = x
        res.new_duration_s = res.orig_duration_s
        if any(
            _is_hard_reason(
                r, trim_only=trim_only, wer_quarantine_threshold=wer_quarantine_threshold,
            )
            for r in res.reasons
        ):
            res.action = "quarantine"
        else:
            res.action = "pass"
        return res, trimmed, sr

    skip_stt = skip_stt_if_wav_outlier and any(
        f.startswith("wav_artifact_") or f == "wav_outlier_combined"
        for f in res.wav_outlier_flags
    )

    if skip_stt:
        res.reasons.append("wav_outlier_skip_stt")
        trimmed = x
        res.new_duration_s = res.orig_duration_s
        res.action = "quarantine"
        return res, trimmed, sr

    try:
        hyp, words = transcribe_with_words(
            wav_path, api, model, allow_local, max_retries=stt_max_retries,
        )
    except Exception as exc:
        res.reasons.append(f"stt_failed:{exc}")
        res.hyp_text = ""
        res.action = "stt_pending" if defer_stt_fail else "quarantine"
        return res, x, sr

    res.hyp_text = hyp
    res.n_words = len(meaningful_words(words))
    cut_s, last_w, _ = compute_cut_time(words, buffer_s, res.orig_duration_s)
    res.last_word = last_w
    res.last_word_end_s = cut_s - buffer_s if cut_s > buffer_s else 0.0

    if ref and quarantine_cutoff:
        comp = analyze_completeness(
            ref, hyp, res.orig_duration_s, res.last_word_end_s, tail_gap_fail_s=tail_gap_fail_s,
        )
        res.ref_last_word = comp["ref_last_word"]
        res.hyp_last_word = comp["hyp_last_word"]
        res.last_word_match = comp["last_word_match"]
        res.likely_cutoff = comp["likely_cutoff"]
        res.missing_tail_words = comp["missing_tail_words"]
        res.tail_gap_s = comp["tail_gap_s"]
        if comp["likely_cutoff"]:
            res.reasons.append("likely_cutoff")
        if comp["missing_tail_words"] >= min_missing_tail_words:
            res.reasons.append(f"missing_tail_words:{comp['missing_tail_words']}")

    if res.n_words == 0:
        res.reasons.append("no_stt_words")

    if min_word_prob is not None:
        probs = [w.probability for w in meaningful_words(words) if w.probability is not None]
        if probs and float(np.mean(probs)) < min_word_prob:
            res.reasons.append("low_confidence")

    if ref and not trim_only:
        res.wer = word_error_rate(ref, hyp)
        if res.wer > max_wer:
            res.reasons.append(f"high_wer:{res.wer:.2f}")
        if repetition_flag(hyp):
            res.reasons.append("repetition")
    elif ref:
        res.wer = word_error_rate(ref, hyp)
        if wer_quarantine_threshold is not None and res.wer > wer_quarantine_threshold:
            res.reasons.append(f"high_wer:{res.wer:.2f}")

    trimmed = trim_audio(x, sr, cut_s, min_keep_s) if res.n_words > 0 else x
    res.new_duration_s = len(trimmed) / sr if sr else 0.0
    res.trim_removed_s = max(0.0, res.orig_duration_s - res.new_duration_s)
    res.tail_rms_ratio = tail_rms_ratio(x, sr, res.new_duration_s)

    if not trim_only:
        # Re-score trimmed body (garbage tail should shrink outlier score).
        feat_after = extract_waveform_features(trimmed, sr)
        res.wav_features["after_trim"] = feat_after.as_dict()
        if wav_profile is not None and wav_profile.n_fit > 0:
            score2, wflags2 = score_waveform_outlier(
                feat_after, wav_profile,
                z_threshold=wav_z_threshold,
                score_threshold=wav_score_threshold,
            )
            res.wav_features["outlier_score_after_trim"] = score2
            if score2 < res.wav_outlier_score:
                res.wav_outlier_score = score2
                res.wav_outlier_flags = wflags2

        if res.trim_removed_s > max_tail_removed_s:
            res.reasons.append(f"large_tail_removed:{res.trim_removed_s:.1f}s")

        if res.new_duration_s > max_dur:
            res.reasons.append("still_too_long_after_trim")

        if res.tail_rms_ratio > 0.35 and res.trim_removed_s < 0.05:
            res.reasons.append("noisy_tail_remaining")

    if voice_gate is not None and len(trimmed) > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), trimmed, sr, subtype="PCM_16")
            cos_r, cos_t, note, vreasons = voice_gate.check(tmp_path, ref)
            res.cos_ref = round(cos_r, 4)
            res.cos_teacher = round(cos_t, 4) if cos_t is not None else None
            res.voice_teacher_match = note or ""
            res.reasons.extend(vreasons)
        finally:
            tmp_path.unlink(missing_ok=True)

    if any(
        _is_hard_reason(
            r, trim_only=trim_only, wer_quarantine_threshold=wer_quarantine_threshold,
        )
        for r in res.reasons
    ):
        res.action = "quarantine"
    elif res.trim_removed_s > 0.05:
        res.action = "trim"
    else:
        res.action = "pass"

    return res, trimmed, sr


def apply_result(
    res: ClipResult,
    wav_path: Path,
    trimmed: np.ndarray,
    sr: int,
    *,
    apply: bool,
    in_place: bool,
    out_dir: Path | None,
    quarantine_dir: Path | None,
    backup: bool,
) -> None:
    if not apply:
        return

    if res.action == "stt_pending":
        return

    if res.action == "quarantine":
        if quarantine_dir is not None:
            dest = quarantine_dir / wav_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, dest)
            res.out_path = str(dest)
        return

    if in_place:
        target = wav_path
    elif out_dir is not None:
        target = out_dir / wav_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        return

    if res.action == "pass":
        if not in_place:
            shutil.copy2(wav_path, target)
        res.out_path = str(target)
    else:
        if backup and in_place and not Path(str(wav_path) + ".bak").exists():
            shutil.copy2(wav_path, str(wav_path) + ".bak")
        sf.write(str(target), trimmed, sr, subtype="PCM_16")
        res.out_path = str(target)

    if quarantine_dir is not None:
        stale = quarantine_dir / wav_path.name
        stale.unlink(missing_ok=True)


def render_html(summary: dict, results: list[ClipResult], out_path: Path) -> None:
    by_action = Counter(r.action for r in results)
    wer_vals = [r.wer for r in results if r.wer is not None]
    trim_vals = [r.trim_removed_s for r in results if r.trim_removed_s > 0.05]

    def pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        i = int(len(xs) * p / 100)
        return xs[min(i, len(xs) - 1)]

    quarantine_samples = [r for r in results if r.action == "quarantine"][:40]
    trim_samples = sorted(
        [r for r in results if r.action == "trim"],
        key=lambda r: r.trim_removed_s,
        reverse=True,
    )[:40]

    reason_counts = Counter()
    for r in results:
        for reason in r.reasons:
            reason_counts[reason.split(":")[0]] += 1

    voice_low = [r for r in results if any(
        x.startswith("low_cos_ref") or x.startswith("low_cos_teacher") for x in r.reasons
    )]
    cos_refs = [r.cos_ref for r in results if r.cos_ref is not None]
    cos_tchrs = [r.cos_teacher for r in results if r.cos_teacher is not None]

    rows_q = "\n".join(
        f"<tr><td>{r.wav}</td><td>{r.orig_duration_s:.2f}</td>"
        f"<td>{r.cos_ref if r.cos_ref is not None else '—'}</td>"
        f"<td>{r.cos_teacher if r.cos_teacher is not None else '—'}</td>"
        f"<td>{r.wer if r.wer is not None else '—'}</td>"
        f"<td>{', '.join(r.reasons)}</td><td>{r.hyp_text[:80]!r}</td></tr>"
        for r in quarantine_samples
    )
    rows_t = "\n".join(
        f"<tr><td>{r.wav}</td><td>{r.orig_duration_s:.2f}</td><td>{r.new_duration_s:.2f}</td>"
        f"<td>{r.trim_removed_s:.2f}</td><td>{r.last_word!r}</td></tr>"
        for r in trim_samples
    )

    wav_outliers = sorted(
        [r for r in results if r.wav_outlier_score > 0],
        key=lambda r: r.wav_outlier_score,
        reverse=True,
    )[:40]
    rows_w = "\n".join(
        f"<tr><td>{r.wav}</td><td>{r.wav_outlier_score:.1f}</td>"
        f"<td>{r.wav_features.get('low_energy_tail_s', 0):.2f}</td>"
        f"<td>{r.wav_features.get('spectral_flatness', 0):.3f}</td>"
        f"<td>{', '.join(r.wav_outlier_flags[:4])}</td></tr>"
        for r in wav_outliers
    )

    norm_rows = ""
    profile = summary.get("wav_norm_profile") or {}
    med = profile.get("medians") or {}
    mad = profile.get("mads") or {}
    for key in sorted(med.keys()):
        norm_rows += f"<tr><td>{key}</td><td>{med[key]:.4g}</td><td>{mad[key]:.4g}</td></tr>"

    n_wav_flagged = sum(1 for r in results if r.wav_outlier_flags)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>loli_15s teacher dataset QC — {summary.get('generated_at', '')}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f1419; color: #e6edf3; }}
    h1, h2 {{ color: #7ee787; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; min-width: 140px; }}
    .card b {{ font-size: 1.6rem; display: block; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #30363d; padding: 0.45rem 0.6rem; text-align: left; }}
    th {{ background: #21262d; }}
    tr:nth-child(even) {{ background: #161b22; }}
    .muted {{ color: #8b949e; }}
  </style>
</head>
<body>
  <h1>loli_15s v15 teacher QC report</h1>
  <p class="muted">Generated {summary.get('generated_at', '')} · STT {summary.get('stt', '')} ·
  apply={summary.get('apply', False)} · wav_dir={summary.get('wav_dir', '')}</p>

  <div class="cards">
    <div class="card"><span class="muted">Scanned</span><b>{summary['n_scanned']}</b></div>
    <div class="card"><span class="muted">Pass</span><b>{by_action.get('pass', 0)}</b></div>
    <div class="card"><span class="muted">Trimmed</span><b>{by_action.get('trim', 0)}</b></div>
    <div class="card"><span class="muted">Quarantine</span><b>{by_action.get('quarantine', 0)}</b></div>
    <div class="card"><span class="muted">STT pending</span><b>{by_action.get('stt_pending', 0)}</b></div>
    <div class="card"><span class="muted">WER median</span><b>{pct(wer_vals, 50):.2f}</b></div>
    <div class="card"><span class="muted">Tail cut median</span><b>{pct(trim_vals, 50):.2f}s</b></div>
    <div class="card"><span class="muted">WAV outliers</span><b>{n_wav_flagged}</b></div>
    <div class="card"><span class="muted">Voice QC fail</span><b>{len(voice_low)}</b></div>
    <div class="card"><span class="muted">cos(ref) med</span><b>{pct(cos_refs, 50) if cos_refs else 0:.3f}</b></div>
    <div class="card"><span class="muted">cos(tchr) med</span><b>{pct(cos_tchrs, 50) if cos_tchrs else 0:.3f}</b></div>
  </div>
  <p class="muted">Voice floor: cos(ref) and cos(teacher) ≥ {summary.get('config', {}).get('min_cos_ref', 0.5)} (quarantine below).</p>

  <h2>Corpus waveform norms (robust median / MAD)</h2>
  <p class="muted">Clips that look like speech build the baseline; per-feature robust-z &gt; {summary.get('config', {}).get('wav_z_threshold', 3.5)} flags outliers.</p>
  <table><tr><th>Feature</th><th>Median</th><th>MAD</th></tr>
  {norm_rows or '<tr><td colspan="3">No profile</td></tr>'}
  </table>

  <h2>Waveform outliers (top 40 by score)</h2>
  <table>
    <tr><th>WAV</th><th>Score</th><th>Low-energy tail (s)</th><th>Flatness</th><th>Flags</th></tr>
    {rows_w or '<tr><td colspan="5">None</td></tr>'}
  </table>

  <h2>Failure reasons</h2>
  <table><tr><th>Reason</th><th>Count</th></tr>
  {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in reason_counts.most_common())}
  </table>

  <h2>Quarantine samples (first 40)</h2>
  <table>
    <tr><th>WAV</th><th>Dur (s)</th><th>cos(ref)</th><th>cos(tchr)</th><th>WER</th><th>Reasons</th><th>STT (snippet)</th></tr>
    {rows_q or '<tr><td colspan="5">None</td></tr>'}
  </table>

  <h2>Largest tail trims (top 40)</h2>
  <table>
    <tr><th>WAV</th><th>Before</th><th>After</th><th>Removed</th><th>Last word</th></tr>
    {rows_t or '<tr><td colspan="5">None</td></tr>'}
  </table>

  <h2>Config</h2>
  <pre>{json.dumps(summary.get('config', {}), indent=2)}</pre>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="QC and tail-prune loli_15s teacher WAVs")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--wav-dir", type=Path, default=ROOT / "training/loli_15s/wavs/v15")
    parser.add_argument("--corpus", type=Path, default=ROOT / "training/loli_15s/corpus/texts.jsonl")
    parser.add_argument("--train-ids", type=Path, default=ROOT / "training/loli_15s/corpus/train_ids.txt")
    parser.add_argument("--qc-dir", type=Path, default=ROOT / "training/loli_15s/qc")
    parser.add_argument("--out-dir", type=Path, default=None, help="Pruned WAV output (with --apply)")
    parser.add_argument("--quarantine-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Write trimmed / quarantine outputs")
    parser.add_argument("--in-place", action="store_true", help="Overwrite wav-dir (keeps .bak once)")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stt-api", default=STT_API)
    parser.add_argument("--stt-model", default=STT_MODEL)
    parser.add_argument(
        "--local-whisper-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fallback to local openai-whisper (often broken on long clips; default off)",
    )
    parser.add_argument(
        "--stt-max-retries",
        type=int,
        default=int(os.environ.get("STT_MAX_RETRIES", "5")),
        help="Retry transient STT API errors before marking failure",
    )
    parser.add_argument(
        "--defer-stt-fail",
        action="store_true",
        help="On STT failure after retries, mark stt_pending (not quarantine) for later mop-up",
    )
    parser.add_argument(
        "--retry-stt-from-manifest",
        type=Path,
        default=None,
        help="Only process WAVs flagged stt_failed in a prior prune_manifest.jsonl",
    )
    parser.add_argument(
        "--end-buffer-ms",
        type=int,
        default=500,
        help="Keep this much audio after STT last-word end (ms). "
        "200ms often clipped final syllables; 500ms is safer for SFT.",
    )
    parser.add_argument("--min-keep-ms", type=int, default=400)
    parser.add_argument("--min-dur", type=float, default=0.8)
    parser.add_argument("--max-dur", type=float, default=22.0)
    parser.add_argument("--max-wer", type=float, default=0.35)
    parser.add_argument("--min-peak", type=float, default=0.01)
    parser.add_argument("--max-tail-removed", type=float, default=8.0,
                        help="Flag/quarantine if more than this removed (legacy 14s junk)")
    parser.add_argument("--min-word-prob", type=float, default=None)
    parser.add_argument("--wav-z-threshold", type=float, default=3.5,
                        help="Robust-z per feature vs corpus norms")
    parser.add_argument("--wav-score-threshold", type=float, default=18.0,
                        help="Sum of robust-z above this → wav_outlier_combined")
    parser.add_argument("--wav-profile", type=Path, default=None,
                        help="Reuse saved norm profile (skip fit pass)")
    parser.add_argument("--skip-stt-on-wav-outlier", action="store_true",
                        help="Skip STT for obvious waveform artifacts (faster)")
    parser.add_argument("--wav-only", action="store_true",
                        help="Waveform QC only (no STT)")
    parser.add_argument(
        "--trim-only",
        action="store_true",
        help="STT word-timestamp tail trim only; no WER/wav-outlier quarantine",
    )
    parser.add_argument(
        "--wer-quarantine-threshold",
        type=float,
        default=None,
        help="With --trim-only: quarantine only when WER exceeds this (e.g. 0.75). "
        "No wav-outlier quarantine.",
    )
    parser.add_argument(
        "--quarantine-cutoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quarantine clips with missing tail words / likely last-word cutoff (default on)",
    )
    parser.add_argument(
        "--tail-gap-fail-s",
        type=float,
        default=float(os.environ.get("QC_TAIL_GAP_FAIL_S", "0.30")),
        help="Audio ends within this many seconds of STT last word + mismatch → likely_cutoff",
    )
    parser.add_argument(
        "--min-missing-tail-words",
        type=int,
        default=int(os.environ.get("QC_MIN_MISSING_TAIL_WORDS", "2")),
        help="Quarantine when STT hyp is this many words shorter than reference",
    )
    parser.add_argument(
        "--voice-qc",
        action="store_true",
        help="ECAPA cos(ref) vs enrollment + cos(teacher) vs matched/pr prototype pool",
    )
    parser.add_argument(
        "--no-voice-qc",
        action="store_true",
        help="Disable voice QC even if --voice-qc was set via env default",
    )
    parser.add_argument(
        "--ref-wav",
        type=Path,
        default=ROOT / "data/voices/loli/loli_15s.wav",
        help="Enrollment reference for cos(ref)",
    )
    parser.add_argument(
        "--teacher-train-raw",
        type=Path,
        default=ROOT / "training/loli_15s/train_raw.jsonl",
        help="train_raw for text-matched cos(teacher)",
    )
    parser.add_argument(
        "--teacher-pool",
        type=Path,
        default=ROOT / "training/loli_15s/wavs/v15_pruned",
        help="Directory of accepted teacher WAVs for matching / prototype",
    )
    parser.add_argument(
        "--min-cos-ref",
        type=float,
        default=float(os.environ.get("MIN_COS_REF", "0.5")),
        help="Quarantine when cos(ref) below this (default 0.5)",
    )
    parser.add_argument(
        "--min-cos-teacher",
        type=float,
        default=float(os.environ.get("MIN_COS_TEACHER", "0.5")),
        help="Quarantine when cos(teacher) below this (default 0.5)",
    )
    parser.add_argument(
        "--voice-qc-device",
        default=os.environ.get("VOICE_QC_DEVICE", ""),
        help="ECAPA device (default cuda:0 or cpu)",
    )
    parser.add_argument(
        "--voice-qc-cache",
        type=Path,
        default=ROOT / "training/loli_15s/.cache/ecapa",
    )
    args = parser.parse_args()
    if args.no_voice_qc:
        args.voice_qc = False
    elif os.environ.get("VOICE_QC", "").strip() in ("1", "true", "yes"):
        args.voice_qc = True

    wav_dir = args.wav_dir
    if not wav_dir.is_dir():
        raise SystemExit(f"Missing wav dir: {wav_dir}")

    out_dir = args.out_dir
    if args.apply and not args.in_place and out_dir is None:
        out_dir = args.root / "training/loli_15s/wavs/v15_pruned"
    quarantine_dir = args.quarantine_dir
    if (
        args.trim_only
        and args.wer_quarantine_threshold is None
        and not args.voice_qc
        and not args.quarantine_cutoff
    ):
        quarantine_dir = None
    elif args.apply and quarantine_dir is None:
        quarantine_dir = args.root / "training/loli_15s/wavs/v15_quarantine"

    index = build_wav_index(args.corpus, args.train_ids)
    wavs = sorted(wav_dir.glob("*.wav"))
    if args.retry_stt_from_manifest and args.retry_stt_from_manifest.is_file():
        retry_names: set[str] = set()
        for line in args.retry_stt_from_manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if any(str(r).startswith("stt_failed") for r in row.get("reasons") or []):
                retry_names.add(row["wav"])
        wavs = [p for p in wavs if p.name in retry_names]
        print(f"Retry STT failures only: {len(wavs)} WAVs from {args.retry_stt_from_manifest}", flush=True)
    if args.limit:
        wavs = wavs[: args.limit]

    buffer_s = args.end_buffer_ms / 1000.0
    min_keep_s = args.min_keep_ms / 1000.0
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {len(wavs)} WAVs in {wav_dir}", flush=True)
    if not args.wav_only:
        print(f"STT: {args.stt_api} (model={args.stt_model})", flush=True)

    profile_path = args.wav_profile or (args.qc_dir / "wav_norm_profile.json")
    feat_cache: dict[str, WaveformFeatures] = {}

    def extract_only(path: Path) -> tuple[str, WaveformFeatures]:
        x, sr = load_audio_mono(path)
        return path.name, extract_waveform_features(x, sr)

    voice_gate = None
    if args.voice_qc:
        from voice_ecapa_qc import (  # noqa: E402
            VoiceQcGate,
            build_teacher_index,
            default_device,
        )

        ref_wav = args.ref_wav if args.ref_wav.is_absolute() else args.root / args.ref_wav
        if not ref_wav.is_file():
            raise SystemExit(f"Missing --ref-wav: {ref_wav}")
        train_raw = (
            args.teacher_train_raw
            if args.teacher_train_raw.is_absolute()
            else args.root / args.teacher_train_raw
        )
        teacher_root = (
            args.teacher_pool
            if args.teacher_pool.is_absolute()
            else args.root / args.teacher_pool
        )
        teacher_index = build_teacher_index(train_raw, teacher_root, root=args.root)
        device = args.voice_qc_device or default_device()
        print(
            f"Voice QC: ECAPA on {device}, cos(ref)>={args.min_cos_ref}, "
            f"cos(teacher)>={args.min_cos_teacher}, pool={len(teacher_index)} teachers",
            flush=True,
        )
        voice_gate = VoiceQcGate(
            ref_wav=ref_wav,
            teacher_index=teacher_index,
            device=device,
            cache_dir=args.voice_qc_cache,
            min_cos_ref=args.min_cos_ref,
            min_cos_teacher=args.min_cos_teacher,
        )

    wav_profile: NormProfile | None = None
    if args.trim_only:
        msg = "Trim-only: STT tail trim"
        if args.wer_quarantine_threshold is not None:
            msg += f", quarantine WER>{args.wer_quarantine_threshold}"
        if args.voice_qc:
            msg += f", voice cos(ref/tchr)>={args.min_cos_ref}"
        if args.quarantine_cutoff:
            msg += (
                f", quarantine cutoff (tail_gap<{args.tail_gap_fail_s}s, "
                f"missing_words>={args.min_missing_tail_words})"
            )
        print(msg, flush=True)
    else:
        print("Phase 1: waveform features + corpus norms...", flush=True)
        if args.workers <= 1:
            for p in wavs:
                name, feat = extract_only(p)
                feat_cache[name] = feat
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for name, feat in pool.map(extract_only, wavs):
                    feat_cache[name] = feat

        if args.wav_profile and args.wav_profile.is_file():
            wav_profile = load_profile(args.wav_profile)
            print(f"Loaded wav profile: {args.wav_profile} (n_fit={wav_profile.n_fit})", flush=True)
        else:
            wav_profile = fit_norm_profile(list(feat_cache.values()))
            save_profile(wav_profile, profile_path)
            print(f"Fitted wav profile on {wav_profile.n_fit} clips → {profile_path}", flush=True)

    results: list[ClipResult] = []
    t0 = time.time()

    def work(path: Path) -> tuple[ClipResult, np.ndarray, int]:
        meta = index.get(path.name)
        if meta is None:
            meta = {"text": "", "corpus_id": corpus_id_from_wav_name(path.name)}
        return evaluate_clip(
            path,
            meta,
            api=args.stt_api,
            model=args.stt_model,
            allow_local=args.local_whisper_fallback,
            buffer_s=buffer_s,
            min_keep_s=min_keep_s,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            max_wer=args.max_wer,
            min_peak=args.min_peak,
            max_tail_removed_s=args.max_tail_removed,
            min_word_prob=args.min_word_prob,
            wav_profile=wav_profile,
            wav_z_threshold=args.wav_z_threshold,
            wav_score_threshold=args.wav_score_threshold,
            skip_stt_if_wav_outlier=args.skip_stt_on_wav_outlier,
            pre_feat=feat_cache.get(path.name),
            wav_only=args.wav_only,
            trim_only=args.trim_only,
            wer_quarantine_threshold=args.wer_quarantine_threshold,
            voice_gate=voice_gate,
            quarantine_cutoff=args.quarantine_cutoff,
            tail_gap_fail_s=args.tail_gap_fail_s,
            min_missing_tail_words=args.min_missing_tail_words,
            defer_stt_fail=args.defer_stt_fail,
            stt_max_retries=args.stt_max_retries,
        )

    if args.workers <= 1:
        for i, p in enumerate(wavs, 1):
            res, trimmed, sr = work(p)
            apply_result(
                res, p, trimmed, sr,
                apply=args.apply, in_place=args.in_place,
                out_dir=out_dir, quarantine_dir=quarantine_dir, backup=args.backup,
            )
            results.append(res)
            if i % 25 == 0:
                print(f"  [{i}/{len(wavs)}] {res.action} {p.name}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(work, p): p for p in wavs}
            done = 0
            for fut in as_completed(futs):
                p = futs[fut]
                res, trimmed, sr = fut.result()
                apply_result(
                    res, p, trimmed, sr,
                    apply=args.apply, in_place=args.in_place,
                    out_dir=out_dir, quarantine_dir=quarantine_dir, backup=args.backup,
                )
                results.append(res)
                done += 1
                if done % 25 == 0:
                    print(f"  [{done}/{len(wavs)}] last={res.action} {p.name}", flush=True)

    elapsed = time.time() - t0
    by_action = Counter(r.action for r in results)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_scanned": len(results),
        "by_action": dict(by_action),
        "elapsed_s": round(elapsed, 1),
        "wav_dir": str(wav_dir),
        "apply": args.apply,
        "in_place": args.in_place,
        "out_dir": str(out_dir) if out_dir else None,
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else None,
        "stt": f"{args.stt_api} model={args.stt_model}",
        "wav_norm_profile": wav_profile.to_dict() if wav_profile is not None else None,
        "config": {
            "end_buffer_ms": args.end_buffer_ms,
            "max_wer": args.max_wer,
            "max_dur": args.max_dur,
            "max_tail_removed": args.max_tail_removed,
            "wav_z_threshold": args.wav_z_threshold,
            "wav_score_threshold": args.wav_score_threshold,
            "wav_only": args.wav_only,
            "trim_only": args.trim_only,
            "wer_quarantine_threshold": args.wer_quarantine_threshold,
            "voice_qc": args.voice_qc,
            "min_cos_ref": args.min_cos_ref,
            "min_cos_teacher": args.min_cos_teacher,
            "ref_wav": str(args.ref_wav),
            "teacher_pool": str(args.teacher_pool),
            "quarantine_cutoff": args.quarantine_cutoff,
            "tail_gap_fail_s": args.tail_gap_fail_s,
            "min_missing_tail_words": args.min_missing_tail_words,
        },
    }

    json_path = args.qc_dir / "prune_report.json"
    manifest_path = args.qc_dir / "prune_manifest.jsonl"
    html_path = args.qc_dir / "prune_report.html"

    json_path.write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    with manifest_path.open("w") as mf:
        for r in results:
            mf.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    render_html(summary, results, html_path)

    pass_list = args.qc_dir / "pass_ids.txt"
    trim_list = args.qc_dir / "trim_ids.txt"
    quarantine_list = args.qc_dir / "quarantine_ids.txt"
    pass_list.write_text("\n".join(r.wav for r in results if r.action == "pass") + "\n")
    trim_list.write_text("\n".join(r.wav for r in results if r.action == "trim") + "\n")
    quarantine_list.write_text("\n".join(r.wav for r in results if r.action == "quarantine") + "\n")
    stt_pending_list = args.qc_dir / "stt_pending_wavs.txt"
    stt_pending_list.write_text(
        "\n".join(r.wav for r in results if r.action == "stt_pending") + "\n"
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {html_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
