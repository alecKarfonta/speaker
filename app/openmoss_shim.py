"""
Speaker-compatible API shim in front of pwilkin/openmoss moss-tts-server.

Runs the C++ GGML inference engine locally and exposes the same routes as
app/moss_api.py for voice library + TTS, without loading PyTorch/MOSS-TTS.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.moss_glm_compat import (
    _list_voice_names,
    _resolve_language_name,
    router as glm_compat_router,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openmoss-shim")

OPENMOSS_URL = os.environ.get("OPENMOSS_SERVER", "http://127.0.0.1:8081").rstrip("/")
MODEL_ID = os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS-v1.5")
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/data/voices"))

app = FastAPI(title="OpenMOSS TTS API (C++/GGML shim)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(glm_compat_router)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice_name: Optional[str] = None
    language: Optional[str] = "en"
    emotion: Optional[str] = None
    output_format: Optional[str] = "wav"
    max_new_tokens: int = Field(300, ge=256, le=16384)
    tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    sampling: Optional[int] = None


def _prepare_text(text: str, emotion: Optional[str]) -> str:
    if not emotion:
        return text
    tag = emotion if emotion.startswith("(") else f"({emotion})"
    return f"{tag} {text}"


def _resolve_reference_b64(voice_name: Optional[str]) -> Optional[str]:
    if not voice_name:
        return None
    names = _list_voice_names()
    if voice_name not in names:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found. Available: {', '.join(names)}",
        )
    voice_dir = VOICES_DIR / voice_name
    audio_files = sorted(
        f for f in voice_dir.iterdir()
        if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    )
    if not audio_files:
        raise HTTPException(status_code=404, detail=f"No audio files for voice '{voice_name}'")
    ref_path = audio_files[0]
    logger.info(f"[TTS] Using reference {ref_path}")
    return base64.b64encode(ref_path.read_bytes()).decode("ascii")


@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OPENMOSS_URL}/health")
            r.raise_for_status()
            info = await client.get(f"{OPENMOSS_URL}/info")
            info.raise_for_status()
            meta = info.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"openmoss server unavailable: {e}") from e

    return {
        "status": "healthy",
        "service": "TTS",
        "version": "1.0.0",
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "engine": "openmoss-ggml",
        "device": "cuda",
        "available_voices": len(_list_voice_names()),
        "openmoss": meta,
    }


@app.get("/voices")
async def list_voices(format: Optional[str] = None):
    names = _list_voice_names()
    if format == "flat":
        return names
    return {"voices": names, "total_count": len(names)}


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    text = _prepare_text(request.text, request.emotion)
    ref_b64 = _resolve_reference_b64(request.voice_name)

    payload: dict = {
        "text": text,
        "max_new_tokens": request.max_new_tokens,
    }
    # tokens = hard duration target (~12.5 frames/s). Only send when caller asks.
    if request.tokens is not None:
        payload["tokens"] = request.tokens
    lang = _resolve_language_name(request.language)
    if lang:
        payload["language"] = lang
    if ref_b64:
        payload["reference_wav_b64"] = ref_b64
    if request.temperature is not None or request.top_p is not None:
        payload["sampling"] = {}
        if request.temperature is not None:
            payload["sampling"]["audio_temperature"] = request.temperature
        if request.top_p is not None:
            payload["sampling"]["audio_top_p"] = request.top_p

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{OPENMOSS_URL}/tts", json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.text[:500])
            wav_bytes = r.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    gen_time = time.perf_counter() - t0
    gen_hdr = r.headers.get("X-MOSS-Generate-Seconds")
    decode_hdr = r.headers.get("X-MOSS-Decode-Seconds")

    duration = 0.0
    sr = 24000
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes))
        duration = len(data) / sr
    except Exception:
        pass

    headers = {
        "Content-Disposition": "attachment; filename=speech.wav",
        "Content-Length": str(len(wav_bytes)),
        "X-Sample-Rate": str(sr),
        "X-Audio-Duration": f"{duration:.2f}",
        "X-Generation-Time": gen_hdr or f"{gen_time:.2f}",
        "X-Model": MODEL_ID,
        "X-Engine": "openmoss-ggml",
    }
    if request.voice_name:
        headers["X-Voice"] = request.voice_name
    if duration > 0:
        headers["X-RTF"] = f"{float(gen_hdr or gen_time) / duration:.2f}"

    logger.info(
        f"[TTS] {duration:.1f}s audio in {gen_time:.1f}s "
        f"(openmoss gen={gen_hdr}, decode={decode_hdr})"
    )
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/audio/speech")
async def openai_speech(request: dict):
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{OPENMOSS_URL}/v1/audio/speech", json=request)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500])
        return Response(content=r.content, media_type=r.headers.get("content-type", "audio/wav"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
