"""
Streaming audio generation for Qwen3-TTS.

Qwen3-TTS supports dual-track hybrid streaming with ~97ms first-packet latency.
This module provides chunked streaming output from the Qwen3-TTS backend.
"""

import asyncio
from typing import AsyncIterator, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

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
    """
    data: bytes
    sample_rate: int
    is_final: bool
    chunk_index: int
    timestamp_ms: float
    
    def to_metadata(self) -> dict:
        """Return metadata dict for streaming protocol."""
        return {
            "sample_rate": self.sample_rate,
            "is_final": self.is_final,
            "chunk_index": self.chunk_index,
            "timestamp_ms": self.timestamp_ms
        }


class QwenTTSStreamingHandler:
    """
    Handles streaming audio generation from Qwen3-TTS.
    
    Qwen3-TTS's dual-track architecture enables:
    - First audio packet after single character input
    - End-to-end latency as low as 97ms
    
    Note: Current implementation uses chunked output from full generation.
    For true real-time streaming, use vLLM-Omni when available.
    """
    
    DEFAULT_CHUNK_SIZE = 4096  # samples per chunk (~170ms at 24kHz)
    
    def __init__(
        self, 
        backend: "QwenTTSBackend",
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ):
        """Initialize streaming handler.
        
        Args:
            backend: Initialized QwenTTSBackend instance
            chunk_size: Number of audio samples per chunk
        """
        self.backend = backend
        self.chunk_size = chunk_size
    
    async def stream_synthesis(
        self,
        text: str,
        voice_name: str = "Vivian",
        language: str = "Auto",
        mode: str = "custom_voice",
        instruct: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio generation chunk by chunk.
        
        This generates the full audio and yields it in chunks for streaming.
        The chunking provides a streaming interface even though the underlying
        generation is non-incremental.
        
        Args:
            text: Text to synthesize
            voice_name: Speaker name for custom_voice mode
            language: Target language or "Auto"
            mode: Generation mode (custom_voice, voice_design, voice_clone)
            instruct: Optional instruction for style control
            **kwargs: Additional generation parameters
            
        Yields:
            AudioChunk objects containing audio data and metadata
        """
        # Generate full audio first (current qwen-tts doesn't support true streaming)
        audio, sample_rate = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.backend.generate_speech(
                text=text,
                voice_name=voice_name,
                language=language,
                mode=mode,
                instruct=instruct,
                **kwargs
            )
        )
        
        # Stream chunks
        chunk_index = 0
        total_samples = len(audio)
        
        for i in range(0, total_samples, self.chunk_size):
            chunk_data = audio[i:i + self.chunk_size]
            is_final = (i + self.chunk_size >= total_samples)
            
            yield AudioChunk(
                data=self._to_bytes(chunk_data),
                sample_rate=sample_rate,
                is_final=is_final,
                chunk_index=chunk_index,
                timestamp_ms=(i / sample_rate) * 1000
            )
            
            chunk_index += 1
            
            # Yield control to event loop between chunks
            await asyncio.sleep(0)
    
    def stream_synthesis_sync(
        self,
        text: str,
        voice_name: str = "Vivian",
        language: str = "Auto",
        mode: str = "custom_voice",
        instruct: Optional[str] = None,
        **kwargs
    ):
        """
        Synchronous generator version of stream_synthesis.
        
        For use with FastAPI's StreamingResponse which expects a sync generator.
        
        Yields:
            Tuple of (audio_bytes, sample_rate, metadata_dict)
        """
        # Generate full audio
        audio, sample_rate = self.backend.generate_speech(
            text=text,
            voice_name=voice_name,
            language=language,
            mode=mode,
            instruct=instruct,
            **kwargs
        )
        
        total_samples = len(audio)
        total_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        for chunk_index, i in enumerate(range(0, total_samples, self.chunk_size)):
            chunk_data = audio[i:i + self.chunk_size]
            is_final = (chunk_index == total_chunks - 1)
            
            metadata = {
                "sample_rate": sample_rate,
                "is_final": is_final,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "timestamp_ms": (i / sample_rate) * 1000
            }
            
            yield chunk_data, sample_rate, metadata
    
    def _to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to int16 PCM bytes.
        
        Args:
            audio: Float audio samples in range [-1, 1]
            
        Returns:
            Bytes in int16 little-endian format
        """
        # Clip to valid range and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()
