"""Shared MOSS-TTS-Realtime full-text generation (stream + batch)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch

logger = logging.getLogger("moss-tts")

RT_PROFILE = os.environ.get("MOSS_RT_PROFILE", "false").lower() in ("1", "true", "yes")
RT_BATCH_CHUNK_DURATION = float(os.environ.get("MOSS_RT_BATCH_CHUNK_DURATION", "8"))


@dataclass
class RTTiming:
    infer_s: float = 0.0
    decode_s: float = 0.0
    n_infer: int = 0
    n_decode: int = 0
    n_audio_frames: int = 0
    cuda_infer_ms: float = 0.0
    cuda_decode_ms: float = 0.0

    def log(self, prefix: str = "Stream-RT") -> None:
        ms_per_frame = (
            (self.infer_s * 1000.0 / self.n_audio_frames)
            if self.n_audio_frames > 0
            else 0.0
        )
        msg = (
            f"[{prefix}][Timing] infer={self.infer_s:.2f}s ({self.n_infer} calls), "
            f"decode={self.decode_s:.2f}s ({self.n_decode} chunks), "
            f"frames={self.n_audio_frames}, ms/frame={ms_per_frame:.1f}"
        )
        if RT_PROFILE and self.cuda_infer_ms > 0:
            msg += (
                f", cuda_infer={self.cuda_infer_ms:.0f}ms"
                f", cuda_decode={self.cuda_decode_ms:.0f}ms"
            )
        logger.info(msg)


@dataclass
class RTChunkTuning:
    initial_text_chunk: int
    steady_text_chunk: int
    min_samples_first: int
    min_samples_steady: int
    decoder_chunk_frames: int


def default_chunk_tuning(
    *,
    sample_rate: int,
    initial_text_chunk: int,
    steady_text_chunk: int,
    min_samples_first_ms: float,
    min_samples_steady_ms: float,
    decoder_chunk_frames: int,
) -> RTChunkTuning:
    return RTChunkTuning(
        initial_text_chunk=initial_text_chunk,
        steady_text_chunk=steady_text_chunk,
        min_samples_first=int(sample_rate * min_samples_first_ms / 1000.0),
        min_samples_steady=int(sample_rate * min_samples_steady_ms / 1000.0),
        decoder_chunk_frames=decoder_chunk_frames,
    )


def make_token_sanitizer(
    worker: Any,
    audio_eos_token: Optional[int] = None,
    codebook_size: Optional[int] = None,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, bool]]:
    eos = audio_eos_token
    if eos is None:
        eos = int(getattr(worker.inferencer, "audio_eos_token", 1026))
    cbs = codebook_size
    if cbs is None:
        cbs = int(getattr(worker.codec, "codebook_size", 1024))

    def _sanitize(tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.numel() == 0:
            return tokens, False
        eos_rows = (tokens[:, 0] == eos).nonzero(as_tuple=False)
        invalid_rows = ((tokens < 0) | (tokens >= cbs)).any(dim=1)
        stop_idx = None
        if eos_rows.numel() > 0:
            stop_idx = int(eos_rows[0].item())
        if invalid_rows.any():
            inv_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
            stop_idx = inv_idx if stop_idx is None else min(stop_idx, inv_idx)
        if stop_idx is not None:
            return tokens[:stop_idx], True
        return tokens, False

    return _sanitize


class _CudaTimer:
    """Accumulate GPU time when MOSS_RT_PROFILE=1."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = RT_PROFILE and device.type == "cuda"
        self.total_ms = 0.0

    def __enter__(self):
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self._end.record()
            torch.cuda.synchronize(self.device)
            self.total_ms += self._start.elapsed_time(self._end)


@contextmanager
def _codec_streaming(codec: Any, batch_size: int = 1):
    """One codec streaming session per turn (MOSS fast_api pattern)."""
    streaming_fn = getattr(codec, "streaming", None)
    if streaming_fn is not None:
        with streaming_fn(batch_size=batch_size):
            yield
        return
    codec._start_streaming(batch_size=batch_size)
    try:
        yield
    finally:
        if hasattr(codec, "_stop_streaming"):
            codec._stop_streaming()


def iter_rt_pcm_chunks(
    worker: Any,
    session: Any,
    text: str,
    tune: RTChunkTuning,
    *,
    device: torch.device,
    prime_delay: bool,
    drain_batch_steps: int,
    buffer_for_stream: bool = True,
    codec: Any | None = None,
    decoder_chunk_frames: Optional[int] = None,
    decoder_overlap_frames: Optional[int] = None,
    decoder_initial_frames: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Push full text through session; yield PCM float32 chunks.

    When buffer_for_stream=True, first chunk ships ASAP then steady buffer applies.
    When False (/tts batch), decode once at flush for seamless timbre.
    """
    from mossttsrealtime.streaming_mossttsrealtime import AudioStreamDecoder

    sanitize = make_token_sanitizer(worker)
    timing = RTTiming()
    active_codec = codec if codec is not None else worker.codec

    overlap = (
        decoder_overlap_frames
        if decoder_overlap_frames is not None
        else int(os.environ.get("MOSS_RT_STREAM_DECODER_OVERLAP_FRAMES", "0"))
    )
    if buffer_for_stream:
        chunk_frames = decoder_chunk_frames or tune.decoder_chunk_frames
        if decoder_initial_frames is not None:
            initial_frames = decoder_initial_frames
        else:
            stream_initial = os.environ.get("MOSS_RT_STREAM_DECODER_INITIAL_FRAMES", "none")
            initial_frames = (
                None
                if stream_initial.lower() in ("none", "null", "")
                else int(stream_initial)
            )
    else:
        # Batch /tts: single flush decode — no chunk-boundary timbre shifts.
        chunk_frames = 1_000_000
        initial_frames = None
        overlap = 0

    decoder = AudioStreamDecoder(
        active_codec,
        chunk_frames=chunk_frames,
        overlap_frames=overlap,
        initial_chunk_frames=initial_frames,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )

    def decode_frames(audio_frames) -> Iterator[np.ndarray]:
        nonlocal timing
        for frame in audio_frames:
            tokens = frame
            if tokens.dim() == 3:
                tokens = tokens[0]
            tokens, _ = sanitize(tokens)
            if tokens.numel() == 0:
                continue
            timing.n_audio_frames += 1
            decoder.push_tokens(tokens.detach())
            with _CudaTimer(device) as ct:
                chunks = list(decoder.audio_chunks())
            timing.cuda_decode_ms += ct.total_ms
            for wav_chunk in chunks:
                if wav_chunk.numel() == 0:
                    continue
                yield wav_chunk.detach().cpu().numpy().reshape(-1)

    def flush_dec() -> Iterator[np.ndarray]:
        with _CudaTimer(device) as ct:
            final = decoder.flush()
        timing.cuda_decode_ms += ct.total_ms
        if final is not None and final.numel() > 0:
            yield final.detach().cpu().numpy().reshape(-1)

    for _name, module in active_codec.named_modules():
        if hasattr(module, "_streaming_state"):
            module._streaming_state = None

    with _codec_streaming(active_codec, batch_size=1):
        with torch.inference_mode():
            steady_chunk = tune.steady_text_chunk

            def _emit_pcm(chunk: np.ndarray) -> Iterator[np.ndarray]:
                if chunk.size == 0:
                    return
                timing.n_decode += 1
                yield chunk

            text_tokens = worker.tokenizer.encode(text, add_special_tokens=False)
            if not text_tokens:
                return

            i = 0
            if prime_delay:
                prime_n = min(worker.processor.delay_tokens_len, len(text_tokens))
                if prime_n > 0:
                    t0 = time.perf_counter()
                    with _CudaTimer(device) as ct:
                        audio_frames = session.push_text_tokens(text_tokens[:prime_n])
                    timing.cuda_infer_ms += ct.total_ms
                    timing.infer_s += time.perf_counter() - t0
                    timing.n_infer += 1
                    t0 = time.perf_counter()
                    for pcm in decode_frames(audio_frames):
                        timing.decode_s += time.perf_counter() - t0
                        yield from _emit_pcm(pcm)
                        t0 = time.perf_counter()
                    timing.decode_s += time.perf_counter() - t0
                    i = prime_n

            while i < len(text_tokens):
                token_chunk = text_tokens[i : i + steady_chunk]
                i += len(token_chunk)
                t0 = time.perf_counter()
                with _CudaTimer(device) as ct:
                    audio_frames = session.push_text_tokens(token_chunk)
                timing.cuda_infer_ms += ct.total_ms
                timing.infer_s += time.perf_counter() - t0
                timing.n_infer += 1
                t0 = time.perf_counter()
                for pcm in decode_frames(audio_frames):
                    timing.decode_s += time.perf_counter() - t0
                    yield from _emit_pcm(pcm)
                    t0 = time.perf_counter()
                timing.decode_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            with _CudaTimer(device) as ct:
                audio_frames = session.end_text()
            timing.cuda_infer_ms += ct.total_ms
            timing.infer_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            for pcm in decode_frames(audio_frames):
                timing.decode_s += time.perf_counter() - t0
                yield from _emit_pcm(pcm)
                t0 = time.perf_counter()
            timing.decode_s += time.perf_counter() - t0

            while True:
                t0 = time.perf_counter()
                with _CudaTimer(device) as ct:
                    audio_frames = session.drain(max_steps=drain_batch_steps)
                timing.cuda_infer_ms += ct.total_ms
                timing.infer_s += time.perf_counter() - t0
                timing.n_infer += 1
                if not audio_frames:
                    break
                t0 = time.perf_counter()
                for pcm in decode_frames(audio_frames):
                    timing.decode_s += time.perf_counter() - t0
                    yield from _emit_pcm(pcm)
                    t0 = time.perf_counter()
                timing.decode_s += time.perf_counter() - t0
                if session.inferencer.is_finished:
                    break

            for pcm in flush_dec():
                yield from _emit_pcm(pcm)

            timing.log()


def _stack_generated_codes(worker: Any, inferencer: Any) -> torch.Tensor | None:
    """Stack inferencer tokens to [C, T] for one-shot batch codec decode."""
    tokens_list = inferencer._generated_tokens
    if not tokens_list:
        return None

    stacked = torch.stack(tokens_list, dim=0)
    if stacked.dim() == 3:
        stacked = stacked[:, 0, :]
    elif stacked.dim() == 2 and stacked.shape[0] == 1:
        stacked = stacked.squeeze(0).unsqueeze(0)

    if stacked.numel() == 0:
        return None

    sanitize = make_token_sanitizer(worker)
    stacked, _ = sanitize(stacked)
    if stacked.numel() == 0:
        return None
    return stacked.permute(1, 0).contiguous()


def _run_rt_generation(
    worker: Any,
    session: Any,
    text: str,
    tune: RTChunkTuning,
    *,
    device: torch.device,
    prime_delay: bool,
    drain_batch_steps: int,
    timing: RTTiming,
) -> None:
    """Token generation only — no codec decode (official batch infer.py pattern)."""
    text_tokens = worker.tokenizer.encode(text, add_special_tokens=False)
    if not text_tokens:
        raise RuntimeError("Text tokenized to empty — nothing to generate")

    i = 0
    if prime_delay:
        prime_n = min(worker.processor.delay_tokens_len, len(text_tokens))
        if prime_n > 0:
            t0 = time.perf_counter()
            with _CudaTimer(device) as ct:
                session.push_text_tokens(text_tokens[:prime_n])
            timing.cuda_infer_ms += ct.total_ms
            timing.infer_s += time.perf_counter() - t0
            timing.n_infer += 1
            i = prime_n

    steady_chunk = tune.steady_text_chunk
    while i < len(text_tokens):
        token_chunk = text_tokens[i : i + steady_chunk]
        i += len(token_chunk)
        t0 = time.perf_counter()
        with _CudaTimer(device) as ct:
            session.push_text_tokens(token_chunk)
        timing.cuda_infer_ms += ct.total_ms
        timing.infer_s += time.perf_counter() - t0
        timing.n_infer += 1

    t0 = time.perf_counter()
    with _CudaTimer(device) as ct:
        session.end_text()
    timing.cuda_infer_ms += ct.total_ms
    timing.infer_s += time.perf_counter() - t0

    while True:
        t0 = time.perf_counter()
        with _CudaTimer(device) as ct:
            audio_frames = session.drain(max_steps=drain_batch_steps)
        timing.cuda_infer_ms += ct.total_ms
        timing.infer_s += time.perf_counter() - t0
        timing.n_infer += 1
        if not audio_frames:
            break
        timing.n_audio_frames += len(audio_frames)
        if session.inferencer.is_finished:
            break


def decode_rt_codes_batch(worker: Any, codes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """One-shot batch decode — no codec.streaming(), no AudioStreamDecoder."""
    with torch.inference_mode():
        codes_dev = codes.to(device)
        result = worker.codec.decode(codes_dev, chunk_duration=RT_BATCH_CHUNK_DURATION)
        if isinstance(result, dict):
            wav = result["audio"][0]
        else:
            wav = result
        return wav.reshape(-1).detach().cpu().float()


def collect_rt_audio(
    worker: Any,
    session: Any,
    text: str,
    tune: RTChunkTuning,
    *,
    device: torch.device,
    prime_delay: bool,
    drain_batch_steps: int,
) -> torch.Tensor:
    """Non-streaming: generate tokens, then official one-shot codec decode."""
    timing = RTTiming()
    with torch.inference_mode():
        _run_rt_generation(
            worker,
            session,
            text,
            tune,
            device=device,
            prime_delay=prime_delay,
            drain_batch_steps=drain_batch_steps,
            timing=timing,
        )

        codes = _stack_generated_codes(worker, session.inferencer)
        if codes is None:
            raise RuntimeError("Realtime model produced no audio tokens")

        t0 = time.perf_counter()
        with _CudaTimer(device) as ct:
            wav = decode_rt_codes_batch(worker, codes, device)
        timing.decode_s = time.perf_counter() - t0
        timing.cuda_decode_ms += ct.total_ms
        timing.n_decode = 1
        timing.n_audio_frames = codes.shape[1]
        timing.log("Batch-RT")

    if wav.numel() == 0:
        raise RuntimeError("Realtime model produced no audio")
    return wav
