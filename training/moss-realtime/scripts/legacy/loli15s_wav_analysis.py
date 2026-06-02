"""Waveform feature extraction + robust outlier scoring for teacher WAV QC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WaveformFeatures:
    duration_s: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    crest_factor: float = 0.0
    dynamic_range_db: float = 0.0
    zcr_mean: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_flatness: float = 0.0
    hf_energy_ratio: float = 0.0
    speech_fraction: float = 0.0
    speech_to_total_rms: float = 0.0
    envelope_std: float = 0.0
    silence_prefix_s: float = 0.0
    silence_suffix_s: float = 0.0
    low_energy_tail_s: float = 0.0
    click_index: float = 0.0
    dc_offset: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# Features used for multivariate outlier distance (speech-like clips).
NORM_KEYS = (
    "duration_s",
    "crest_factor",
    "zcr_mean",
    "spectral_centroid_hz",
    "spectral_flatness",
    "hf_energy_ratio",
    "speech_fraction",
    "envelope_std",
    "silence_suffix_s",
    "low_energy_tail_s",
    "click_index",
)


@dataclass
class NormProfile:
    medians: dict[str, float] = field(default_factory=dict)
    mads: dict[str, float] = field(default_factory=dict)
    n_fit: int = 0

    def to_dict(self) -> dict:
        return {"medians": self.medians, "mads": self.mads, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, d: dict) -> NormProfile:
        return cls(medians=d.get("medians", {}), mads=d.get("mads", {}), n_fit=d.get("n_fit", 0))


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.array([float(np.sqrt(np.mean(x * x)))]) if len(x) else np.array([0.0])
    n_frames = 1 + (len(x) - frame) // hop
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        chunk = x[start : start + frame]
        out[i] = np.sqrt(np.mean(chunk * chunk))
    return out


def _spectral_features_frame(frame: np.ndarray, sr: int) -> tuple[float, float, float]:
    if len(frame) < 64:
        return 0.0, 0.0, 0.0
    win = np.hanning(len(frame)) * frame
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    power = spec * spec
    total = float(power.sum()) + 1e-12
    centroid = float((freqs * power).sum() / total)
    log_spec = np.log(spec + 1e-12)
    flatness = float(np.exp(log_spec.mean()) / (spec.mean() + 1e-12))
    hf = float(power[freqs >= 4000].sum() / total)
    return centroid, flatness, hf


def extract_waveform_features(x: np.ndarray, sr: int) -> WaveformFeatures:
    feat = WaveformFeatures()
    if len(x) == 0 or sr <= 0:
        return feat

    x = x.astype(np.float64)
    feat.duration_s = len(x) / sr
    feat.peak = float(np.max(np.abs(x)))
    feat.rms = float(np.sqrt(np.mean(x * x)))
    feat.crest_factor = feat.peak / (feat.rms + 1e-12)
    feat.dc_offset = float(np.mean(x))

    p99 = float(np.percentile(np.abs(x), 99)) + 1e-12
    p10 = float(np.percentile(np.abs(x), 10)) + 1e-12
    feat.dynamic_range_db = float(20 * np.log10(p99 / p10))

    signs = np.sign(x)
    signs[signs == 0] = 1
    zc = np.where(signs[:-1] * signs[1:] < 0)[0]
    feat.zcr_mean = float(len(zc) / max(len(x) - 1, 1) * sr)

    frame = min(2048, max(256, len(x) // 8))
    hop = frame // 4
    rms_frames = _frame_rms(x, frame, hop)
    hop_s = hop / sr

    if len(rms_frames) == 0:
        return feat

    noise_floor = float(np.percentile(rms_frames, 15))
    speech_thr = max(noise_floor * 2.5, float(np.median(rms_frames)) * 0.12, 1e-5)
    speech_mask = rms_frames >= speech_thr
    feat.speech_fraction = float(speech_mask.mean())
    speech_rms = rms_frames[speech_mask]
    feat.speech_to_total_rms = float(speech_rms.mean() / (feat.rms + 1e-12)) if len(speech_rms) else 0.0
    if len(speech_rms) > 1:
        feat.envelope_std = float(np.std(speech_rms))

    centroids: list[float] = []
    flatnesses: list[float] = []
    hfs: list[float] = []
    for i in range(len(rms_frames)):
        if not speech_mask[i]:
            continue
        start = i * hop
        chunk = x[start : start + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        c, fl, hf = _spectral_features_frame(chunk, sr)
        centroids.append(c)
        flatnesses.append(fl)
        hfs.append(hf)

    if centroids:
        feat.spectral_centroid_hz = float(np.median(centroids))
        feat.spectral_flatness = float(np.median(flatnesses))
        feat.hf_energy_ratio = float(np.median(hfs))

    # prefix / suffix silence
    idx = np.where(speech_mask)[0]
    if len(idx):
        feat.silence_prefix_s = float(idx[0] * hop_s)
        feat.silence_suffix_s = float((len(rms_frames) - 1 - idx[-1]) * hop_s)
        # Non-speech energy after last speech frame (typical garbage tail before STT trim).
        tail_frames = rms_frames[idx[-1] + 1 :]
        if len(tail_frames):
            junk = (tail_frames >= noise_floor) & (tail_frames < speech_thr)
            feat.low_energy_tail_s = float(junk.sum() * hop_s) if junk.any() else float(len(tail_frames) * hop_s)
        else:
            feat.low_energy_tail_s = 0.0
    else:
        feat.silence_prefix_s = feat.duration_s
        feat.silence_suffix_s = feat.duration_s
        feat.low_energy_tail_s = feat.duration_s

    # click pops: large sample-to-sample jumps vs median
    diff = np.abs(np.diff(x))
    if len(diff):
        med = float(np.median(diff)) + 1e-12
        feat.click_index = float(np.percentile(diff, 99) / med)

    return feat


def _robust_mad(vals: np.ndarray) -> float:
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med))) or 1e-9


def fit_norm_profile(
    features: list[WaveformFeatures],
    *,
    min_duration: float = 0.8,
    max_duration: float = 20.0,
    min_peak: float = 0.01,
    max_speech_fraction: float = 0.98,
) -> NormProfile:
    """Build corpus norms from clips that look like plausible speech."""
    pool: list[WaveformFeatures] = []
    for f in features:
        if f.duration_s < min_duration or f.duration_s > max_duration:
            continue
        if f.peak < min_peak:
            continue
        if f.speech_fraction < 0.05 or f.speech_fraction > max_speech_fraction:
            continue
        pool.append(f)

    if len(pool) < 20:
        pool = features

    profile = NormProfile(n_fit=len(pool))
    for key in NORM_KEYS:
        vals = np.array([getattr(f, key) for f in pool], dtype=np.float64)
        profile.medians[key] = float(np.median(vals))
        profile.mads[key] = _robust_mad(vals)
    return profile


def robust_z(value: float, median: float, mad: float) -> float:
    return abs(value - median) / (1.4826 * mad + 1e-12)


def score_waveform_outlier(
    feat: WaveformFeatures,
    profile: NormProfile,
    *,
    z_threshold: float = 3.5,
    score_threshold: float = 18.0,
) -> tuple[float, list[str]]:
    """Return combined robust-z score and human-readable flags."""
    flags: list[str] = []
    zs: list[float] = []

    for key in NORM_KEYS:
        if key not in profile.medians:
            continue
        z = robust_z(getattr(feat, key), profile.medians[key], profile.mads[key])
        zs.append(z)
        if z >= z_threshold:
            flags.append(f"wav_outlier_{key}")

    # Rule-based artifact detectors (independent of corpus)
    if feat.low_energy_tail_s > 1.5 and feat.silence_suffix_s < feat.duration_s * 0.4:
        flags.append("wav_artifact_noise_tail")
    if feat.spectral_flatness > 0.55 and feat.speech_fraction < 0.5:
        flags.append("wav_artifact_flat_spectrum")
    if feat.click_index > 25:
        flags.append("wav_artifact_clicks")
    if feat.hf_energy_ratio > 0.45:
        flags.append("wav_artifact_hf_noise")
    if feat.zcr_mean > profile.medians.get("zcr_mean", 0) + 4 * profile.mads.get("zcr_mean", 1):
        flags.append("wav_artifact_high_zcr")

    score = float(sum(zs))
    if score >= score_threshold and "wav_outlier" not in " ".join(flags):
        flags.append("wav_outlier_combined")

    return score, flags


def save_profile(profile: NormProfile, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")


def load_profile(path: str | Path) -> NormProfile:
    return NormProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
