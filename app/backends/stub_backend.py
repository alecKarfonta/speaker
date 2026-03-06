"""
Stub TTS backend — returns silence. Used for testing audiobook/visual features
without GPU dependencies for the TTS model.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from app.tts_backend_base import TTSBackendBase


class StubTTSBackend(TTSBackendBase):
    """Stub backend that returns silence. For testing without GPU."""

    @property
    def backend_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-silence"

    @property
    def sample_rate(self) -> int:
        return 22050

    def initialize_model(self) -> None:
        self.logger.info("Stub TTS backend initialized (returns silence)")

    def load_voice(self, voice_name: str, voice_path: str) -> None:
        from app.tts_backend_base import Voice
        self.voices[voice_name] = Voice(name=voice_name, path=voice_path)
        self.logger.info(f"Stub loaded voice: {voice_name}")

    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Return silence matching approximate speech duration."""
        # ~150 words per minute, ~5 chars per word
        duration_sec = max(1.0, len(text) / 750.0 * 60.0)
        num_samples = int(duration_sec * self.sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        return silence, self.sample_rate
