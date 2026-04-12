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
            voice_names = resp.json()
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
        Generate speech by calling the remote MOSS /tts/stream endpoint.

        Uses the streaming endpoint because the non-streaming /tts endpoint
        has a known CUDA assertion issue with the MOSS-TTS-Realtime codec
        decode. The streaming endpoint handles codec decode correctly.

        The binary framing format is:
            4-byte audio_len (LE) + 4-byte metadata_len (LE) + audio_bytes + metadata_json

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        import requests
        import struct

        payload = {
            "text": text,
            "voice_name": voice_name or None,
            "language": language,
        }

        self.logger.debug(f"MOSS proxy: generating speech for '{text[:60]}...' (voice={voice_name})")

        try:
            resp = requests.post(
                f"{self._moss_url}/tts/stream",
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MOSS TTS request failed: {e}") from e

        # Read the full streaming response and reassemble audio from
        # binary-framed chunks: [4B audio_len][4B meta_len][audio][meta] ...
        audio_chunks = []
        sr = self._sample_rate
        buf = b""

        for chunk in resp.iter_content(chunk_size=65536):
            buf += chunk
            # Parse as many complete frames as available
            while len(buf) >= 8:
                audio_len, meta_len = struct.unpack_from("<II", buf, 0)
                frame_size = 8 + audio_len + meta_len
                if len(buf) < frame_size:
                    break  # incomplete frame, wait for more data
                audio_bytes = buf[8 : 8 + audio_len]
                # meta_bytes = buf[8 + audio_len : frame_size]  # not needed
                buf = buf[frame_size:]
                if audio_len > 0:
                    audio_chunks.append(audio_bytes)

        # Concatenate all WAV chunks and decode
        if not audio_chunks:
            raise RuntimeError("MOSS TTS stream returned no audio data")

        # Each chunk is a complete WAV file; decode and concatenate
        all_audio = []
        for wav_bytes in audio_chunks:
            data, chunk_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            sr = chunk_sr
            all_audio.append(data)

        audio_data = np.concatenate(all_audio, axis=0) if len(all_audio) > 1 else all_audio[0]
        self._sample_rate = sr

        duration = len(audio_data) / sr
        self.logger.debug(f"MOSS proxy: received {duration:.1f}s audio ({sr}Hz, {len(audio_chunks)} chunks)")

        return audio_data, sr

    # -- Voice design (optional, used by audiobook character voices) ----------

    def design_voice(self, voice_name: str, description: str) -> Optional[str]:
        """
        Design a voice on the remote MOSS service using /tts/design.

        Args:
            voice_name: Identifier for the new voice
            description: Voice description prompt

        Returns:
            Path to generated voice reference audio, or None
        """
        import requests

        payload = {
            "voice_name": voice_name,
            "description": description,
        }

        self.logger.info(f"MOSS proxy: designing voice '{voice_name}' ({len(description)} chars)")

        try:
            resp = requests.post(
                f"{self._moss_url}/tts/design",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()

            # Response may be JSON with voice info
            try:
                data = resp.json()
                voice_path = data.get("voice_path") or data.get("path")
            except Exception:
                voice_path = None

            # Register the voice locally
            self.voices[voice_name] = Voice(name=voice_name, file_paths=[voice_path] if voice_path else [])
            self.logger.info(f"MOSS proxy: designed voice '{voice_name}' → {voice_path}")
            return voice_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"MOSS voice design failed for '{voice_name}': {e}")
            raise RuntimeError(f"MOSS voice design failed: {e}") from e
