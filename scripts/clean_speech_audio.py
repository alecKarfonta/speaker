#!/usr/bin/env python3
"""
Clean up speech recordings using an Audacity-style processing chain.

Recommended order (default pipeline):
  high-pass → noise reduction → EQ → de-click → compression → de-ess → normalize → limiter

Usage:
    # Single file (writes alongside input as name_cleaned.wav unless -o is set)
    ./scripts/clean_speech_audio.py data/voices/major/major_2_01.wav

    # Explicit output, female voice preset (higher HPF)
    ./scripts/clean_speech_audio.py input.wav -o output.wav --voice female

    # Batch directory
    ./scripts/clean_speech_audio.py data/voices/major/*.wav --output-dir data/voices/major/cleaned

    # Manual noise profile (seconds) from a noise-only segment
    ./scripts/clean_speech_audio.py input.wav --noise-start 0.0 --noise-end 0.5

    # Lighter processing
    ./scripts/clean_speech_audio.py input.wav --noise-db 6 --no-deess --no-declick

Dependencies: numpy, scipy, soundfile (see requirements-stable.txt).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # (samples, channels)
    return data, sr


def save_audio(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return data[:, 0]
    return np.mean(data, axis=1)


def ensure_2d(mono: np.ndarray) -> np.ndarray:
    return mono[:, np.newaxis] if mono.ndim == 1 else mono


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------


def remove_dc(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y, axis=0, keepdims=True)


def db_to_linear(db: float) -> float:
    return float(10 ** (db / 20.0))


def linear_to_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def butter_sos(
    sr: int,
    cutoff_hz: float,
    btype: str,
    order: int = 4,
) -> np.ndarray:
    nyq = sr / 2.0
    wn = min(max(cutoff_hz / nyq, 1e-5), 0.999)
    return signal.butter(order, wn, btype=btype, output="sos")


def apply_sos(y: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return signal.sosfiltfilt(sos, y)
    out = np.zeros_like(y)
    for ch in range(y.shape[1]):
        out[:, ch] = signal.sosfiltfilt(sos, y[:, ch])
    return out


def high_pass(y: np.ndarray, sr: int, cutoff_hz: float, rolloff_db: int = 24) -> np.ndarray:
    order = max(2, int(rolloff_db / 6))
    sos = butter_sos(sr, cutoff_hz, "highpass", order=order)
    return apply_sos(y, sos)


def bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    lo = min(max(low_hz / nyq, 1e-5), 0.999)
    hi = min(max(high_hz / nyq, lo + 1e-5), 0.999)
    sos = signal.butter(order, [lo, hi], btype="bandpass", output="sos")
    return apply_sos(y, sos)


def peaking_eq(
    y: np.ndarray,
    sr: int,
    freq_hz: float,
    gain_db: float,
    q: float = 1.4,
) -> np.ndarray:
    """RBJ peaking EQ (gain in dB)."""
    if abs(gain_db) < 0.05:
        return y
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * a
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a

    b = np.array([b0, b1, b2]) / a0
    a_coef = np.array([1.0, a1 / a0, a2 / a0])
    if y.ndim == 1:
        return signal.lfilter(b, a_coef, y)
    out = np.zeros_like(y)
    for ch in range(y.shape[1]):
        out[:, ch] = signal.lfilter(b, a_coef, y[:, ch])
    return out


def speech_eq(y: np.ndarray, sr: int) -> np.ndarray:
    """Gentle speech-clarity curve: cut mud, boost presence and air."""
    y = peaking_eq(y, sr, 250.0, -3.0, q=1.2)
    y = peaking_eq(y, sr, 3000.0, 2.5, q=1.5)
    y = peaking_eq(y, sr, 6500.0, 1.5, q=2.0)
    return y


def find_noise_segment(
    y: np.ndarray,
    sr: int,
    duration_s: float = 0.5,
    frame_ms: float = 20.0,
) -> Tuple[int, int]:
    """Pick the quietest contiguous window (likely background noise)."""
    mono = to_mono(y) if y.ndim > 1 else y
    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, len(mono) // frame)
    rms = np.array(
        [
            np.sqrt(np.mean(mono[i * frame : (i + 1) * frame] ** 2) + 1e-12)
            for i in range(n_frames)
        ]
    )
    win_frames = max(1, int(duration_s * 1000.0 / frame_ms))
    if n_frames <= win_frames:
        return 0, len(mono)

    kernel = np.ones(win_frames)
    rolling = np.convolve(rms, kernel, mode="valid") / win_frames
    start_frame = int(np.argmin(rolling))
    start = start_frame * frame
    end = min(len(mono), start + int(duration_s * sr))
    return start, end


def noise_reduce(
    y: np.ndarray,
    sr: int,
    noise_start: int,
    noise_end: int,
    noise_db: float = 9.0,
    sensitivity: float = 1.0,
    freq_smooth_bands: int = 4,
    n_fft: int = 2048,
    hop: Optional[int] = None,
) -> np.ndarray:
    """
    Spectral noise reduction using a noise profile (Audacity-style).

    noise_db: strength (6–12 typical). Maps to magnitude attenuation.
    sensitivity: threshold multiplier (>1 treats more as noise).
    freq_smooth_bands: smooth mask across frequency to reduce musical noise.
    """
    hop = hop or n_fft // 4
    prop_decrease = min(1.0, max(0.0, noise_db / 20.0))

    def _reduce_channel(chan: np.ndarray) -> np.ndarray:
        n0 = chan[noise_start:noise_end]
        if len(n0) < hop:
            n0 = chan[: min(len(chan), n_fft)]

        _, _, noise_stft = signal.stft(
            n0, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros"
        )
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        _, _, stft = signal.stft(
            chan, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros"
        )
        mag = np.abs(stft)
        phase = np.angle(stft)

        n_thresh = noise_mag * (1.0 + sensitivity)
        mask = (mag - n_thresh) / (mag + 1e-12)
        mask = np.clip(mask, 0.0, 1.0)
        mask = 1.0 - prop_decrease * (1.0 - mask)

        if freq_smooth_bands > 1:
            k = max(1, int(freq_smooth_bands))
            mask = uniform_filter1d(mask, size=k, axis=0, mode="nearest")

        cleaned = mask * mag * np.exp(1j * phase)
        _, out = signal.istft(
            cleaned, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros"
        )
        return out[: len(chan)].astype(np.float32)

    if y.ndim == 1:
        return _reduce_channel(y)
    out = np.zeros_like(y)
    for ch in range(y.shape[1]):
        out[:, ch] = _reduce_channel(y[:, ch])
    return out


def remove_clicks(
    y: np.ndarray,
    sr: int,
    threshold_factor: float = 3.0,
    max_click_ms: float = 0.5,
    max_repairs_fraction: float = 0.001,
) -> np.ndarray:
    """
    Repair isolated impulsive clicks (mouth pops). Conservative: only extreme
    outliers are touched, and each repair spans at most a few milliseconds.
    """
    max_half = max(1, int(sr * max_click_ms / 2000.0))
    merge_gap = max(2, int(sr * 0.0002))  # ~0.2 ms — same click only

    def _channel(chan: np.ndarray) -> np.ndarray:
        d = np.diff(chan, prepend=chan[0])
        abs_d = np.abs(d)
        rms = float(np.sqrt(np.mean(chan**2) + 1e-12))
        # Outliers only: well above typical sample-to-sample change
        p99 = float(np.percentile(abs_d, 99.9))
        med = float(np.median(abs_d) + 1e-12)
        thresh = max(p99 * threshold_factor, med * 25.0, rms * 0.25)
        idx = np.where(abs_d > thresh)[0]
        if len(idx) == 0 or len(idx) > len(chan) * max_repairs_fraction:
            return chan

        out = chan.copy()
        i = 0
        while i < len(idx):
            center = int(idx[i])
            start = max(1, center - max_half)
            end = min(len(chan) - 1, center + max_half + 1)
            j = i + 1
            while j < len(idx) and idx[j] - idx[j - 1] <= merge_gap:
                end = min(len(chan) - 1, int(idx[j]) + max_half + 1)
                j += 1
            if end - start >= 2:
                out[start:end] = np.linspace(chan[start - 1], chan[end], end - start)
            i = j
        return out

    if y.ndim == 1:
        return _channel(y)
    out = np.zeros_like(y)
    for ch in range(y.shape[1]):
        out[:, ch] = _channel(y[:, ch])
    return out


def fix_clipping(y: np.ndarray, clip_level: float = 0.99) -> np.ndarray:
    """Soft repair for samples near full scale."""
    out = y.copy()
    if out.ndim == 1:
        clipped = np.abs(out) >= clip_level
        if not np.any(clipped):
            return out
        idx = np.where(clipped)[0]
        for i in idx:
            lo = max(0, i - 3)
            hi = min(len(out), i + 4)
            good = np.where(np.abs(out[lo:hi]) < clip_level)[0]
            if len(good) >= 2:
                out[i] = np.interp(i, lo + good, out[lo + good])
        return np.clip(out, -1.0, 1.0)

    for ch in range(out.shape[1]):
        out[:, ch] = fix_clipping(out[:, ch], clip_level)
    return out


def compress(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -15.0,
    ratio: float = 3.5,
    attack_ms: float = 15.0,
    release_ms: float = 150.0,
    makeup: bool = True,
) -> np.ndarray:
    """Feed-forward RMS compressor with optional make-up gain."""
    attack = np.exp(-1.0 / (sr * attack_ms / 1000.0 + 1e-12))
    release = np.exp(-1.0 / (sr * release_ms / 1000.0 + 1e-12))
    thresh_lin = db_to_linear(threshold_db)

    def _channel(chan: np.ndarray) -> np.ndarray:
        window = max(1, int(sr * 0.010))
        rms = np.sqrt(
            uniform_filter1d(chan.astype(np.float64) ** 2, size=window, mode="nearest")
            + 1e-12
        )
        env = np.zeros_like(rms)
        for i, level in enumerate(rms):
            coeff = attack if level > env[i - 1] else release if i else release
            if i == 0:
                env[i] = level
            else:
                env[i] = coeff * env[i - 1] + (1.0 - coeff) * level

        gain = np.ones_like(chan)
        over = env > thresh_lin
        if np.any(over):
            target = thresh_lin + (env[over] - thresh_lin) / ratio
            gain[over] = target / (env[over] + 1e-12)
        out = chan * gain
        if makeup:
            peak_in = np.max(np.abs(chan)) + 1e-12
            peak_out = np.max(np.abs(out)) + 1e-12
            out *= peak_in / peak_out
        return out.astype(np.float32)

    if y.ndim == 1:
        return _channel(y)
    out = np.zeros_like(y)
    for ch in range(y.shape[1]):
        out[:, ch] = _channel(y[:, ch])
    return out


def deess(
    y: np.ndarray,
    sr: int,
    low_hz: float = 5000.0,
    high_hz: float = 8000.0,
    reduction_db: float = 4.0,
    threshold_db: float = -32.0,
) -> np.ndarray:
    """Reduce harsh sibilance by ducking the 5–8 kHz band when it is hot."""
    if y.ndim == 1:
        channels = [y]
    else:
        channels = [y[:, c] for c in range(y.shape[1])]

    reduced = []
    for chan in channels:
        band = bandpass(chan, sr, low_hz, high_hz)
        env = uniform_filter1d(np.abs(band), size=max(1, int(sr * 0.005)), mode="nearest")
        thresh = db_to_linear(threshold_db)
        over = env > thresh
        if not np.any(over):
            reduced.append(chan)
            continue
        duck = np.ones_like(chan)
        excess = env[over] / thresh
        duck[over] = 1.0 / (1.0 + (excess - 1.0) * (1.0 - db_to_linear(-reduction_db)))
        rest = chan - band
        reduced.append(rest + band * duck)

    if len(reduced) == 1 and y.ndim == 1:
        return reduced[0].astype(np.float32)
    return np.stack(reduced, axis=1).astype(np.float32)


def normalize_peak(y: np.ndarray, peak_db: float = -1.0, remove_dc_offset: bool = True) -> np.ndarray:
    out = y.copy()
    if remove_dc_offset:
        out = remove_dc(out)
    peak = np.max(np.abs(out)) + 1e-12
    target = db_to_linear(peak_db)
    return (out * (target / peak)).astype(np.float32)


def limiter(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceiling = db_to_linear(ceiling_db)
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def truncate_silence(
    y: np.ndarray,
    sr: int,
    thresh_db: float = -40.0,
    min_silence_ms: float = 200.0,
    keep_ms: float = 80.0,
) -> np.ndarray:
    """Remove long silent gaps (Audacity Truncate Silence style)."""
    mono = to_mono(y) if y.ndim > 1 else y
    thresh = db_to_linear(thresh_db)
    frame = max(1, int(sr * 0.02))
    min_len = int(sr * min_silence_ms / 1000.0)
    keep = int(sr * keep_ms / 1000.0)

    active = []
    i = 0
    while i < len(mono):
        chunk = mono[i : i + frame]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        if rms >= thresh:
            active.append((i, min(len(mono), i + frame)))
        i += frame

    if not active:
        return y

    merged = [active[0]]
    for start, end in active[1:]:
        if start - merged[-1][1] < min_len:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    pieces = []
    for start, end in merged:
        s = max(0, start - keep)
        e = min(len(mono), end + keep)
        pieces.append(y[s:e])
    return np.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class CleanConfig:
    voice: str = "female"  # female | male
    highpass_hz: Optional[float] = None
    highpass_rolloff_db: int = 24
    noise_db: float = 9.0
    noise_sensitivity: float = 1.0
    noise_freq_smooth: int = 4
    noise_start: Optional[float] = None
    noise_end: Optional[float] = None
    eq: bool = True
    declick: bool = True
    declip: bool = True
    compress: bool = True
    threshold_db: float = -15.0
    ratio: float = 3.5
    attack_ms: float = 15.0
    release_ms: float = 150.0
    deess: bool = True
    deess_reduction_db: float = 4.0
    normalize: bool = True
    peak_db: float = -1.0
    limiter: bool = True
    limit_db: float = -1.0
    truncate_silence: bool = False
    skip_noise: bool = False
    skip_highpass: bool = False


def default_highpass_hz(voice: str) -> float:
    return 130.0 if voice == "female" else 90.0


def clean_speech(y: np.ndarray, sr: int, cfg: CleanConfig) -> np.ndarray:
    """Run the full speech cleanup chain on float32 audio (samples, channels)."""
    if y.ndim == 1:
        y = ensure_2d(y)

    y = remove_dc(y)

    if not cfg.skip_highpass:
        cutoff = cfg.highpass_hz or default_highpass_hz(cfg.voice)
        y = high_pass(y, sr, cutoff, rolloff_db=cfg.highpass_rolloff_db)

    if not cfg.skip_noise:
        if cfg.noise_start is not None and cfg.noise_end is not None:
            n0 = int(cfg.noise_start * sr)
            n1 = int(cfg.noise_end * sr)
        else:
            n0, n1 = find_noise_segment(y, sr)
        y = noise_reduce(
            y,
            sr,
            n0,
            n1,
            noise_db=cfg.noise_db,
            sensitivity=cfg.noise_sensitivity,
            freq_smooth_bands=cfg.noise_freq_smooth,
        )

    if cfg.eq:
        y = speech_eq(y, sr)

    if cfg.declick:
        y = remove_clicks(y, sr)

    if cfg.declip:
        y = fix_clipping(y)

    if cfg.compress:
        y = compress(
            y,
            sr,
            threshold_db=cfg.threshold_db,
            ratio=cfg.ratio,
            attack_ms=cfg.attack_ms,
            release_ms=cfg.release_ms,
            makeup=True,
        )

    if cfg.deess:
        y = deess(y, sr, reduction_db=cfg.deess_reduction_db)

    if cfg.normalize:
        y = normalize_peak(y, peak_db=cfg.peak_db, remove_dc_offset=True)

    if cfg.limiter:
        y = limiter(y, ceiling_db=cfg.limit_db)

    if cfg.truncate_silence:
        y = truncate_silence(y, sr)

    return np.clip(y, -1.0, 1.0).astype(np.float32)


def default_output_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    stem = input_path.stem + "_cleaned"
    if output_dir is not None:
        return output_dir / f"{stem}{input_path.suffix}"
    return input_path.with_name(f"{stem}{input_path.suffix}")


def iter_inputs(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("*.wav")))
            out.extend(sorted(path.glob("*.flac")))
            out.extend(sorted(path.glob("*.ogg")))
        elif path.exists():
            out.append(path)
        else:
            # glob from shell
            out.extend(sorted(Path().glob(p)))
    return out


def build_config(args: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        voice=args.voice,
        highpass_hz=args.highpass_hz,
        highpass_rolloff_db=args.highpass_rolloff,
        noise_db=args.noise_db,
        noise_sensitivity=args.noise_sensitivity,
        noise_freq_smooth=args.noise_freq_smooth,
        noise_start=args.noise_start,
        noise_end=args.noise_end,
        eq=not args.no_eq,
        declick=not args.no_declick,
        declip=not args.no_declip,
        compress=not args.no_compress,
        threshold_db=args.threshold_db,
        ratio=args.ratio,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
        deess=not args.no_deess,
        deess_reduction_db=args.deess_db,
        normalize=not args.no_normalize,
        peak_db=args.peak_db,
        limiter=not args.no_limiter,
        limit_db=args.limit_db,
        truncate_silence=args.truncate_silence,
        skip_noise=args.skip_noise,
        skip_highpass=args.skip_highpass,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean speech audio (noise reduction, EQ, compression, etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Audio file(s), directory, or glob",
    )
    p.add_argument("-o", "--output", type=Path, help="Output file (single input only)")
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for batch outputs (default: next to each input)",
    )
    p.add_argument(
        "--voice",
        choices=("female", "male"),
        default="female",
        help="Preset for high-pass cutoff when --highpass-hz is not set",
    )
    p.add_argument("--highpass-hz", type=float, default=None, help="High-pass cutoff Hz")
    p.add_argument("--highpass-rolloff", type=int, default=24, help="High-pass rolloff dB/oct")
    p.add_argument("--noise-db", type=float, default=9.0, help="Noise reduction strength (6–12)")
    p.add_argument("--noise-sensitivity", type=float, default=1.0, help="Noise gate sensitivity")
    p.add_argument("--noise-freq-smooth", type=int, default=4, help="Frequency smoothing bands")
    p.add_argument(
        "--noise-start",
        type=float,
        default=None,
        help="Noise profile start time (seconds); requires --noise-end",
    )
    p.add_argument("--noise-end", type=float, default=None, help="Noise profile end time (seconds)")
    p.add_argument("--threshold-db", type=float, default=-15.0, help="Compressor threshold dB")
    p.add_argument("--ratio", type=float, default=3.5, help="Compressor ratio")
    p.add_argument("--attack-ms", type=float, default=15.0)
    p.add_argument("--release-ms", type=float, default=150.0)
    p.add_argument("--deess-db", type=float, default=4.0, help="Sibilance reduction dB")
    p.add_argument("--peak-db", type=float, default=-1.0, help="Normalize peak dBFS")
    p.add_argument("--limit-db", type=float, default=-1.0, help="Limiter ceiling dBFS")
    p.add_argument("--truncate-silence", action="store_true", help="Remove long silent gaps")
    p.add_argument("--skip-noise", action="store_true")
    p.add_argument("--skip-highpass", action="store_true")
    p.add_argument("--no-eq", action="store_true")
    p.add_argument("--no-declick", action="store_true")
    p.add_argument("--no-declip", action="store_true")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--no-deess", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-limiter", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if (args.noise_start is None) ^ (args.noise_end is None):
        print("error: --noise-start and --noise-end must be used together", file=sys.stderr)
        return 2

    inputs = iter_inputs(args.inputs)
    if not inputs:
        print("error: no input files found", file=sys.stderr)
        return 1
    if args.output is not None and len(inputs) != 1:
        print("error: -o/--output only valid with a single input file", file=sys.stderr)
        return 2

    cfg = build_config(args)
    ok = 0
    for inp in inputs:
        try:
            data, sr = load_audio(inp)
            cleaned = clean_speech(data, sr, cfg)
            out = args.output if args.output else default_output_path(inp, args.output_dir)
            save_audio(out, cleaned, sr)
            if not args.quiet:
                print(f"{inp} -> {out}")
            ok += 1
        except Exception as exc:
            print(f"error: {inp}: {exc}", file=sys.stderr)
    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    sys.exit(main())
