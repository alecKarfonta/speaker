"""
True streaming audio generation for Qwen3-TTS.

Uses the Qwen3-TTS-streaming fork to provide real-time incremental
audio generation with two-phase streaming, torch.compile + CUDA graphs,
and Hann crossfade for click-free chunk boundaries.

Expected TTFA: ~208ms (vs 570ms baseline).
"""

import asyncio
import struct
import time
import threading
import queue
from typing import AsyncIterator, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import json as _json

if TYPE_CHECKING:
    from app.backends.qwen_tts import QwenTTSBackend


@dataclass
class AudioChunk:
    """A chunk of streaming audio.

    Attributes:
        data: Raw audio bytes (int16 PCM)
        sample_rate: Audio sample rate (typically 24000)
        is_final: Whether this is the last chunk
        chunk_index: Zero-based chunk number
        timestamp_ms: Position in audio stream (milliseconds)
        audio_duration: Duration of this chunk in seconds
    """
    data: bytes
    sample_rate: int
    is_final: bool
    chunk_index: int
    timestamp_ms: float
    audio_duration: float = 0.0

    def to_metadata(self) -> dict:
        """Return metadata dict for streaming protocol."""
        return {
            "sample_rate": self.sample_rate,
            "is_final": self.is_final,
            "chunk_index": self.chunk_index,
            "timestamp_ms": self.timestamp_ms,
            "audio_duration": round(self.audio_duration, 3),
            "streaming": "qwen-realtime",
        }


class QwenTTSStreamingHandler:
    """
    Handles true streaming audio generation from Qwen3-TTS.

    Uses the streaming fork's stream_generate_voice_clone() method
    with two-phase streaming for optimal TTFA + quality balance.
    Falls back to chunked batch generation if streaming is unavailable.
    """

    def __init__(
        self,
        backend: "QwenTTSBackend",
    ):
        """Initialize streaming handler.

        Args:
            backend: Initialized QwenTTSBackend instance
        """
        self.backend = backend

    @property
    def supports_true_streaming(self) -> bool:
        """Check if true streaming is available."""
        return getattr(self.backend, '_streaming_enabled', False)

    def stream_synthesis_sync(
        self,
        text: str,
        voice_name: str,
        language: str = "Auto",
        **kwargs
    ):
        """
        Synchronous generator for streaming audio.

        If true streaming is available, uses stream_generate_voice_clone().
        Otherwise falls back to batch generation + chunking.

        Yields:
            Tuple of (audio_np_chunk, sample_rate, metadata_dict)
        """
        if self.supports_true_streaming:
            yield from self._stream_true(text, voice_name, language, **kwargs)
        else:
            yield from self._stream_chunked_fallback(text, voice_name, language, **kwargs)

    def _stream_true(self, text, voice_name, language, **kwargs):
        """True streaming via stream_generate_voice_clone()."""
        chunk_index = 0
        total_samples = 0

        for chunk_np, sr in self.backend.generate_speech_streaming(
            text=text,
            voice_name=voice_name,
            language=language,
            **kwargs
        ):
            timestamp_ms = (total_samples / sr) * 1000
            audio_duration = len(chunk_np) / sr
            total_samples += len(chunk_np)

            metadata = {
                "sample_rate": sr,
                "is_final": False,
                "chunk_index": chunk_index,
                "timestamp_ms": timestamp_ms,
                "audio_duration": round(audio_duration, 3),
                "streaming": "qwen-realtime",
            }

            chunk_index += 1
            yield chunk_np, sr, metadata

        # Mark final
        if chunk_index > 0:
            # Re-yield nothing, just let caller know stream ended
            pass

    def _stream_chunked_fallback(self, text, voice_name, language, **kwargs):
        """Fallback: generate full audio then chunk it."""
        CHUNK_SIZE = 4096  # ~170ms at 24kHz

        audio, sr = self.backend.generate_speech(
            text=text,
            voice_name=voice_name,
            language=language,
            mode="auto",
            **kwargs
        )

        total_samples = len(audio)
        total_chunks = (total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_index, i in enumerate(range(0, total_samples, CHUNK_SIZE)):
            chunk_data = audio[i:i + CHUNK_SIZE]
            is_final = (chunk_index == total_chunks - 1)

            metadata = {
                "sample_rate": sr,
                "is_final": is_final,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "timestamp_ms": (i / sr) * 1000,
                "audio_duration": round(len(chunk_data) / sr, 3),
                "streaming": "chunked-fallback",
            }

            yield chunk_data, sr, metadata


def build_framed_chunk(wav_np: np.ndarray, sr: int, meta: dict) -> bytes:
    """Build a framed binary chunk for the streaming protocol.

    Format: [4B audio_len][4B meta_len][WAV bytes][JSON metadata]

    Args:
        wav_np: Audio samples as float32 numpy array
        sr: Sample rate
        meta: Metadata dict

    Returns:
        Binary frame bytes
    """
    # Convert to int16 WAV in-memory
    pcm16 = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    raw = pcm16.tobytes()
    data_size = len(raw)

    # 44-byte WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sr, sr * 2, 2, 16,
        b'data', data_size,
    )
    wav_bytes = header + raw

    meta_bytes = _json.dumps(meta).encode()
    frame_header = struct.pack("<II", len(wav_bytes), len(meta_bytes))
    return frame_header + wav_bytes + meta_bytes
