"""
MOSS-TTS HTTP Proxy Backend.
Forwards TTS requests to a remote MOSS-TTS service over HTTP.
Used when MOSS runs in a separate container and tts-api needs to call it.
"""

import io
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf

from ..tts_backend_base import TTSBackendBase, Voice


class MossProxyBackend(TTSBackendBase):
    """TTS backend that proxies to a remote MOSS-TTS HTTP service."""

    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(logger=logger, config=config)
        self._moss_url = os.environ.get(
            "MOSS_PROXY_URL",
            os.environ.get("TTS_BACKEND_HOST", "moss-tts:8000"),
        )
        # Ensure URL has scheme
        if not self._moss_url.startswith("http"):
            self._moss_url = f"http://{self._moss_url}"
        self._sample_rate = 24000  # MOSS default; updated from response headers

    # -- TTSBackendBase abstract properties -----------------------------------

    @property
    def backend_name(self) -> str:
        return "moss-proxy"

    @property
    def model_name(self) -> str:
        return "MOSS-TTS (remote)"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # -- Lifecycle ------------------------------------------------------------

    def initialize_model(self) -> None:
        """No local model to initialize — MOSS runs remotely."""
        self.logger.info(f"MOSS proxy backend initialized → {self._moss_url}")

    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """No-op: voices are managed by the remote MOSS service."""
        self.voices[voice_name] = Voice(name=voice_name, file_paths=[voice_path])

    def load_voices(self, voices_dir: str = "data/voices") -> None:
        """Fetch voice list from the remote MOSS service."""
        try:
            import requests
            resp = requests.get(f"{self._moss_url}/voices", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                voice_names = data
            elif isinstance(data, dict):
                voice_names = data.get("voices", [])
            else:
                voice_names = []
            for name in voice_names:
                self.voices[name] = Voice(name=name, file_paths=[])
            self.logger.info(f"Loaded {len(voice_names)} voices from MOSS at {self._moss_url}")
        except Exception as e:
            self.logger.warning(f"Failed to fetch voices from MOSS: {e}")
            # Fall back to local scan
            super().load_voices(voices_dir)

    # -- Core TTS -------------------------------------------------------------

    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech by calling the remote MOSS /tts endpoint.

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        import requests

        payload = {
            "text": text,
            "voice_name": voice_name or None,
            "language": language,
        }

        self.logger.debug(f"MOSS proxy: generating speech for '{text[:60]}...' (voice={voice_name})")

        try:
            resp = requests.post(
                f"{self._moss_url}/tts",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MOSS TTS request failed: {e}") from e

        wav_bytes = resp.content
        if len(wav_bytes) < 1000:
            raise RuntimeError("MOSS TTS returned empty or invalid audio")

        sr = int(resp.headers.get("X-Sample-Rate", self._sample_rate))
        audio_data, chunk_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        self._sample_rate = chunk_sr or sr

        duration = len(audio_data) / self._sample_rate
        self.logger.debug(f"MOSS proxy: received {duration:.1f}s audio ({self._sample_rate}Hz)")

        return audio_data, self._sample_rate

    # -- Voice design (optional, used by audiobook character voices) ----------

    def design_voice(self, voice_name: str, description: str) -> Optional[str]:
        """
        Design a voice on the remote MOSS service using /tts/design.

        Generates sample audio from a voice description, then saves it as a
        named reference voice on the remote MOSS service for clip synthesis.

        Args:
            voice_name: Identifier for the new voice
            description: Voice description prompt

        Returns:
            Path to generated voice reference audio, or None
        """
        import requests

        design_url = os.environ.get("MOSS_DESIGN_URL", self._moss_url)
        if not design_url.startswith("http"):
            design_url = f"http://{design_url}"

        sample_text = (
            "Hello. This is a short voice sample so you can hear how I sound. "
            "I will be reading your audiobook with this voice."
        )
        design_mode = os.environ.get("MOSS_DESIGN_MODE", "voice_gen").lower()
        use_realtime = design_mode in ("realtime", "rt", "native")

        if use_realtime:
            payload = {"text": sample_text, "language": "en"}
            design_endpoint = f"{design_url}/tts"
        else:
            payload = {"text": sample_text, "instruction": description}
            design_endpoint = f"{design_url}/tts/design"

        self.logger.info(
            f"MOSS proxy: designing voice '{voice_name}' via {design_mode} "
            f"({len(description)} chars)"
        )

        try:
            resp = requests.post(
                design_endpoint,
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            wav_bytes = resp.content
            if len(wav_bytes) < 1000:
                raise RuntimeError("MOSS voice design returned empty or invalid audio")

            upload_resp = requests.post(
                f"{self._moss_url}/voices",
                params={"voice_name": voice_name},
                files={"file": ("reference.wav", wav_bytes, "audio/wav")},
                timeout=60,
            )
            upload_resp.raise_for_status()

            voices_dir = os.environ.get("VOICES_DIR", "data/voices")
            voice_path = os.path.join(voices_dir, voice_name)
            self.voices[voice_name] = Voice(name=voice_name, file_paths=[voice_path])
            self.logger.info(f"MOSS proxy: designed and saved voice '{voice_name}' → {voice_path}")
            return voice_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"MOSS voice design failed for '{voice_name}': {e}")
            raise RuntimeError(f"MOSS voice design failed: {e}") from e
