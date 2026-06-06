"""
MOSS-SoundEffect v2 FastAPI service.

Text-to-audio sound effect generation via MossSoundEffectPipeline (DiT + Flow Matching).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moss-sfx")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

MODEL_ID = os.environ.get(
    "MOSS_SFX_MODEL_ID", "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
)
DEVICE = os.environ.get("MOSS_SFX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_DTYPE_NAME = os.environ.get("MOSS_SFX_TORCH_DTYPE", "float32").lower()
if _DTYPE_NAME == "float32":
    DTYPE = torch.float32
elif _DTYPE_NAME in ("float16", "fp16"):
    DTYPE = torch.float16
else:
    DTYPE = torch.bfloat16
MAX_SECONDS = float(os.environ.get("MOSS_SFX_MAX_SECONDS", "30"))


def _disable_moss_autocast() -> None:
    """Strip autocast from MossSoundEffectPipeline (mixed dtypes → noise on SM120)."""
    from pathlib import Path

    root = os.environ.get("MOSS_SFX_ROOT", "/app/MOSS-TTS/moss_soundeffect_v2")
    path = Path(root) / "pipeline_moss_soundeffect.py"
    if not path.is_file():
        logger.warning("pipeline_moss_soundeffect.py not found at %s", path)
        return
    text = path.read_text()
    markers = (
        "with torch.autocast(device_type, dtype=torch.bfloat16):",
        "with torch.autocast(device_type, dtype=torch.float32):",
    )
    if not any(m in text for m in markers):
        logger.info("MossSoundEffectPipeline autocast already disabled")
        return
    if "import contextlib" not in text:
        text = text.replace("import torch", "import contextlib\nimport torch", 1)
    for marker in markers:
        text = text.replace(marker, "with contextlib.nullcontext():")
    path.write_text(text)
    logger.info("Disabled MossSoundEffectPipeline autocast (%s)", path)


pipe = None
sample_rate = 48000
inference_semaphore = asyncio.Semaphore(1)

app = FastAPI(
    title="MOSS-SoundEffect API",
    description="Text-to-audio sound effect generation (MOSS-SoundEffect v2.0)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SfxRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096, description="Sound description")
    seconds: float = Field(10.0, ge=0.5, le=30.0, description="Output duration (max 30s)")
    num_inference_steps: int = Field(100, ge=10, le=150)
    cfg_scale: float = Field(4.0, ge=1.0, le=8.0)
    sigma_shift: float = Field(5.0, ge=0.0, le=10.0)
    seed: Optional[int] = Field(None, description="Random seed (optional)")


class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    sample_rate: int
    max_seconds: float
    gpu_memory_gb: Optional[float] = None


@app.on_event("startup")
async def load_model():
    global pipe, sample_rate

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

    # Must be set before moss_soundeffect_v2 imports wan_audio (torch.compile).
    _disable_moss_autocast()
    import moss_soundeffect_v2.diffsynth.pipelines.wan_audio as _wan_audio
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    compiled = getattr(_wan_audio, "model_fn_wan_video", None)
    if compiled is not None:
        orig = getattr(compiled, "_orig_mod", None) or getattr(
            compiled, "__wrapped__", None
        )
        if orig is not None:
            _wan_audio.model_fn_wan_video = orig
            logger.info("Using uncompiled model_fn_wan_video")
        else:
            logger.info(
                "model_fn_wan_video has no _orig_mod (compile may be no-op via TORCHDYNAMO_DISABLE)"
            )

    logger.info(
        f"Loading MOSS-SoundEffect from {MODEL_ID} on {DEVICE} (dtype={DTYPE}) ..."
    )
    t0 = time.perf_counter()
    pipe = MossSoundEffectPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device=DEVICE,
    )
    engine = pipe.engine
    # DiT checkpoint is float32; only move device (dit.to(dtype=...) corrupts RoPE buffers).
    engine.dit.to(device=DEVICE)
    engine.text_encoder.to(device=DEVICE, dtype=DTYPE)
    engine.vae.to(device=DEVICE)
    sample_rate = int(getattr(pipe, "sample_rate", 48000))
    logger.info(
        "DiT param dtype=%s text_encoder dtype=%s vae dtype=%s",
        next(engine.dit.parameters()).dtype,
        next(engine.text_encoder.parameters()).dtype,
        next(engine.vae.parameters()).dtype,
    )
    logger.info(
        f"Model ready in {time.perf_counter() - t0:.1f}s "
        f"(sample_rate={sample_rate})"
    )


def _audio_to_wav_bytes(audio: torch.Tensor, sr: int) -> bytes:
    """Encode (B,C,T) or (C,T) tensor to WAV — matches upstream channel layout."""
    wav = audio.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        data = wav.squeeze(0).numpy()
    else:
        data = wav.numpy().T
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _generate_sfx(request: SfxRequest) -> tuple[torch.Tensor, int]:
    if pipe is None:
        raise RuntimeError("Model not loaded")

    seconds = min(float(request.seconds), MAX_SECONDS)
    kwargs = {
        "prompt": request.prompt.strip(),
        "seconds": seconds,
        "num_inference_steps": request.num_inference_steps,
        "cfg_scale": request.cfg_scale,
        "sigma_shift": request.sigma_shift,
        "append_duration_suffix": True,
    }
    base_seed = int(request.seed) if request.seed is not None else 0
    seeds = [base_seed] + [(base_seed + i) % 10_000 for i in range(1, 3)]

    best_audio: torch.Tensor | None = None
    best_std = -1.0
    chosen_seed = base_seed

    for seed in seeds:
        kwargs["seed"] = seed
        with torch.no_grad():
            audio = pipe(**kwargs)
        if hasattr(audio, "audios"):
            audio = audio.audios
        std = float(audio.std())
        if std > best_std:
            best_std = std
            best_audio = audio
            chosen_seed = seed
        if std >= 0.05:
            logger.info("[SFX] seed=%s std=%.4f (accepted)", seed, std)
            break
    else:
        logger.warning(
            "[SFX] low amplitude after %d seeds; using best std=%.4f seed=%s",
            len(seeds),
            best_std,
            chosen_seed,
        )

    assert best_audio is not None
    logger.info(
        f"[SFX] tensor shape={tuple(best_audio.shape)} "
        f"min={float(best_audio.min()):.4f} max={float(best_audio.max()):.4f} "
        f"std={best_std:.4f} seed={chosen_seed}"
    )
    return best_audio, sample_rate


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "MOSS-SoundEffect v2 API",
        "model": MODEL_ID,
        "endpoints": {
            "health": "/health",
            "generate": "POST /sfx",
            "openai_compat": "POST /v1/audio/sfx",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_mem = None
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        gpu_mem = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return HealthResponse(
        status="ready" if pipe is not None else "loading",
        model_id=MODEL_ID,
        device=DEVICE,
        sample_rate=sample_rate,
        max_seconds=MAX_SECONDS,
        gpu_memory_gb=gpu_mem,
    )


@app.post("/sfx")
async def generate_sfx(request: SfxRequest):
    """Generate a sound effect from a text prompt. Returns WAV audio."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(
        f"[SFX] prompt='{request.prompt[:80]}...' "
        f"seconds={request.seconds} steps={request.num_inference_steps}"
    )
    t0 = time.perf_counter()
    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(_generate_sfx, request)
    except Exception as e:
        logger.error(f"[SFX] Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    wav_bytes = _audio_to_wav_bytes(audio, sr)
    wav = audio.detach().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    n_samples = wav.shape[-1]
    duration = n_samples / sr if n_samples else 0.0
    gen_time = time.perf_counter() - t0
    logger.info(f"[SFX] Done: {duration:.1f}s audio in {gen_time:.1f}s")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )


@app.post("/v1/audio/sfx")
async def generate_sfx_v1_compat(request: SfxRequest):
    """Alias for clients expecting a versioned audio route."""
    return await generate_sfx(request)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
