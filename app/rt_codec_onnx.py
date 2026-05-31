"""
ONNX MOSS-Audio-Tokenizer wrapper for MOSS-TTS-Realtime streaming.

Uses OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX (encoder.onnx + decoder.onnx) via
ONNX Runtime. Decode is stateless (no causal streaming KV in the ONNX graph);
quality matches batch decode for each chunk AudioStreamDecoder emits.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger("moss-tts")

N_QUANTIZERS_ONNX = 32  # ONNX graph fixed input depth
N_QUANTIZERS_RT = int(os.environ.get("MOSS_RT_N_QUANTIZERS", "16"))  # MOSS-Realtime AR output
# Inactive RVQ rows: use 0 (valid codebook index); 1024 is text-channel pad only.
AUDIO_PAD_CODE = int(os.environ.get("MOSS_RT_AUDIO_PAD_CODE", "0"))
DOWNSAMPLE_RATE = 1920
SAMPLE_RATE = 24000


def _ort_providers(use_gpu: bool) -> list[str]:
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_session(onnx_path: str, use_gpu: bool) -> ort.InferenceSession:
    path = str(Path(onnx_path).expanduser().resolve())
    sess = ort.InferenceSession(path, providers=_ort_providers(use_gpu))
    logger.info(
        "ONNX session %s providers=%s",
        Path(path).name,
        sess.get_providers(),
    )
    return sess


class MossRTOnnxCodec:
    """Drop-in subset of MOSS PyTorch codec API for realtime decode/encode."""

    codebook_size = 1024
    sampling_rate = SAMPLE_RATE
    downsample_rate = DOWNSAMPLE_RATE

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        *,
        device: str = "cuda:0",
        use_gpu_ort: bool = True,
        n_quantizers: int = N_QUANTIZERS_RT,
    ):
        self.device = torch.device(device)
        self.n_quantizers = n_quantizers  # active RVQ levels for realtime
        self._enc = _load_session(encoder_path, use_gpu_ort)
        self._dec = _load_session(decoder_path, use_gpu_ort)
        self._enc_in = [i.name for i in self._enc.get_inputs()]
        self._enc_out = [o.name for o in self._enc.get_outputs()]
        self._dec_in = [i.name for i in self._dec.get_inputs()]
        self._dec_out = [o.name for o in self._dec.get_outputs()]

    def named_modules(self) -> Iterator[tuple[str, Any]]:
        return iter([])

    def _start_streaming(self, batch_size: int = 1) -> None:
        pass

    def _stop_streaming(self) -> None:
        pass

    @contextmanager
    def streaming(self, batch_size: int = 1):
        yield

    @torch.inference_mode()
    def encode(
        self,
        waveform: torch.Tensor,
        chunk_duration: float | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del chunk_duration, kwargs
        wav = waveform.detach().float().cpu().numpy()
        if wav.ndim == 1:
            wav = wav[np.newaxis, np.newaxis, :]
        elif wav.ndim == 2:
            wav = wav[np.newaxis, :]
        t = wav.shape[-1]
        padded = ((t + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != t:
            wav = np.concatenate(
                [wav, np.zeros((wav.shape[0], wav.shape[1], padded - t), dtype=np.float32)],
                axis=-1,
            )
        nq = np.array(self.n_quantizers, dtype=np.int64)
        r = self._enc.run(
            self._enc_out,
            {self._enc_in[0]: wav.astype(np.float32), self._enc_in[1]: nq},
        )
        codes_out = r[0][: self.n_quantizers, 0, : int(r[1][0])].T.astype(np.int64)
        codes_t = torch.from_numpy(codes_out).to(self.device)
        return {"audio_codes": codes_t}

    def _to_onnx_codes(self, audio_codes: torch.Tensor) -> np.ndarray:
        """[nq_active, 1, T] int64, padded to N_QUANTIZERS_ONNX rows for ONNX."""
        codes = audio_codes.detach().cpu().numpy()
        if codes.ndim == 2:
            if codes.shape[1] == self.n_quantizers and codes.shape[0] != self.n_quantizers:
                codes = codes.T
            codes = codes[:, np.newaxis, :]
        elif codes.ndim == 3 and codes.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected audio_codes shape {codes.shape}")
        n_active, _, t_len = codes.shape
        if n_active < N_QUANTIZERS_ONNX:
            pad = np.full(
                (N_QUANTIZERS_ONNX - n_active, 1, t_len),
                AUDIO_PAD_CODE,
                dtype=np.int64,
            )
            codes = np.concatenate([codes.astype(np.int64), pad], axis=0)
        elif n_active > N_QUANTIZERS_ONNX:
            codes = codes[:N_QUANTIZERS_ONNX].astype(np.int64)
        return codes

    @torch.inference_mode()
    def decode(
        self,
        audio_codes: torch.Tensor,
        chunk_duration: float | None = None,
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        del chunk_duration, kwargs
        codes = self._to_onnx_codes(audio_codes)
        nq = np.array(self.n_quantizers, dtype=np.int64)
        r = self._dec.run(
            self._dec_out,
            {self._dec_in[0]: codes, self._dec_in[1]: nq},
        )
        length = int(r[1][0])
        wav = r[0][0, 0, :length].astype(np.float32)
        wav_t = torch.from_numpy(wav).to(self.device)
        return {"audio": [wav_t]}


def resolve_onnx_codec_paths(root: Path | None = None) -> tuple[Path, Path] | None:
    """Return (encoder.onnx, decoder.onnx) if both exist."""
    candidates = []
    if root is not None:
        candidates.append(root)
    repo = Path(__file__).resolve().parents[1]
    candidates.extend([
        repo / "training/weights/MOSS-Audio-Tokenizer-ONNX",
        repo / "weights/MOSS-Audio-Tokenizer-ONNX",
        Path(os.environ.get("MOSS_RT_ONNX_CODEC_DIR", "")).expanduser(),
    ])

    for base in candidates:
        if not base or not str(base):
            continue
        enc = base / "encoder.onnx"
        dec = base / "decoder.onnx"
        if enc.is_file() and dec.is_file():
            return enc, dec
    return None


def load_onnx_codec(device: str, onnx_dir: str | None = None) -> MossRTOnnxCodec:
    root = Path(onnx_dir).expanduser() if onnx_dir else None
    paths = resolve_onnx_codec_paths(root)
    if paths is None:
        raise FileNotFoundError(
            "MOSS ONNX codec not found. Run: "
            "hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX "
            "--local-dir training/weights/MOSS-Audio-Tokenizer-ONNX"
        )
    use_gpu = os.environ.get("MOSS_RT_ONNX_GPU", "true").lower() in ("1", "true", "yes")
    return MossRTOnnxCodec(
        str(paths[0]),
        str(paths[1]),
        device=device,
        use_gpu_ort=use_gpu,
    )
