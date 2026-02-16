"""
MOSS-TTS FastAPI Service
Exposes voice generation, voice cloning, and streaming TTS endpoints.
"""

import asyncio
import importlib.util
import io
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moss-tts")

# ── Config ──
MODEL_ID = os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS")
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/data/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)
QUANTIZE = os.environ.get("MOSS_QUANTIZE", "4bit")  # "4bit", "8bit", or "none"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _resolve_attn_implementation() -> str:
    if (
        DEVICE == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and DTYPE in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if DEVICE == "cuda":
        return "sdpa"
    return "eager"


ATTN_IMPL = _resolve_attn_implementation()

# ── Semaphore: serialize GPU inference ──
inference_semaphore = asyncio.Semaphore(1)

# ── Global model/processor (loaded at startup) ──
model = None
processor = None
sample_rate = None


def load_model():
    """Load MOSS-TTS model and processor."""
    global model, processor, sample_rate
    from transformers import AutoModel, AutoProcessor

    # Disable broken cuDNN SDPA backend
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    logger.info(f"Loading processor from {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

    if DEVICE == "cuda":
        mem_after_tokenizer = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Audio tokenizer loaded: {mem_after_tokenizer:.2f} GB used")

    # Build model kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": ATTN_IMPL,
    }

    # Configure quantization based on available VRAM
    if QUANTIZE == "4bit" and DEVICE == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=DTYPE,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantization (NF4) with device_map=auto")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to no quantization")
            model_kwargs["torch_dtype"] = DTYPE
    elif QUANTIZE == "8bit" and DEVICE == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            logger.info("Using 8-bit quantization with device_map=auto")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to no quantization")
            model_kwargs["torch_dtype"] = DTYPE
    else:
        model_kwargs["torch_dtype"] = DTYPE
        logger.info(f"No quantization — loading in {DTYPE}")

    logger.info(f"Loading model from {MODEL_ID} (attn={ATTN_IMPL}) ...")
    model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs)

    # If not using device_map, manually move to GPU
    if "device_map" not in model_kwargs:
        model = model.to(DEVICE)

    model.eval()

    sample_rate = processor.model_config.sampling_rate
    logger.info(f"Model loaded. Sample rate: {sample_rate} Hz")

    if DEVICE == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU memory: {mem_gb:.2f} / {total_gb:.2f} GB")


# ── FastAPI app ──
app = FastAPI(
    title="MOSS-TTS API",
    description="Text-to-Speech API powered by MOSS-TTS (MossTTSDelay-8B)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_model()


# ── Request / Response models ──

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    language: Optional[str] = Field("en", description="Language code (informational)")
    max_new_tokens: int = Field(4096, ge=256, le=16384, description="Max audio tokens to generate")
    audio_temperature: Optional[float] = Field(None, description="Sampling temperature for audio tokens")
    audio_top_p: Optional[float] = Field(None, description="Top-p sampling for audio tokens")
    audio_top_k: Optional[int] = Field(None, description="Top-k sampling for audio tokens")
    tokens: Optional[int] = Field(None, description="Target duration in tokens (for duration control)")


class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    attention: str
    sample_rate: Optional[int] = None
    gpu_memory_gb: Optional[float] = None


class VoiceInfo(BaseModel):
    name: str
    files: list[str]


# ── Inference helpers ──

def _generate_audio(text: str, reference_path: Optional[str] = None,
                    max_new_tokens: int = 4096, tokens: Optional[int] = None,
                    **gen_kwargs) -> tuple[torch.Tensor, int]:
    """Synchronous generation — runs in thread pool."""
    kwargs = {}
    if reference_path:
        kwargs["reference"] = [reference_path]
    if tokens is not None:
        kwargs["tokens"] = tokens

    conversation = [processor.build_user_message(text=text, **kwargs)]
    batch = processor([conversation], mode="generation")

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    # Add optional generation hyperparameters if provided
    for k in ("audio_temperature", "audio_top_p", "audio_top_k"):
        if k in gen_kwargs and gen_kwargs[k] is not None:
            generate_kwargs[k] = gen_kwargs[k]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    messages = list(processor.decode(outputs))
    if not messages or not messages[0].audio_codes_list:
        raise RuntimeError("Model returned no audio")

    audio = messages[0].audio_codes_list[0]
    return audio, sample_rate


def _audio_to_wav_bytes(audio: torch.Tensor, sr: int) -> bytes:
    """Convert audio tensor to WAV bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, audio.unsqueeze(0).cpu(), sr, format="wav")
    buf.seek(0)
    return buf.read()


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_mem = None
    if DEVICE == "cuda":
        gpu_mem = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    return HealthResponse(
        status="ready" if model is not None else "loading",
        model_id=MODEL_ID,
        device=DEVICE,
        attention=ATTN_IMPL,
        sample_rate=sample_rate,
        gpu_memory_gb=gpu_mem,
    )


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List available reference voices."""
    voices = []
    if VOICES_DIR.exists():
        for voice_dir in sorted(VOICES_DIR.iterdir()):
            if voice_dir.is_dir():
                files = [f.name for f in voice_dir.iterdir()
                         if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
                if files:
                    voices.append(VoiceInfo(name=voice_dir.name, files=sorted(files)))
    return voices


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate speech from text. Returns WAV audio."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(f"[TTS] Generating: '{request.text[:80]}...'")
    t0 = time.perf_counter()

    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(
                _generate_audio,
                text=request.text,
                max_new_tokens=request.max_new_tokens,
                tokens=request.tokens,
                audio_temperature=request.audio_temperature,
                audio_top_p=request.audio_top_p,
                audio_top_k=request.audio_top_k,
            )
    except Exception as e:
        logger.error(f"[TTS] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    wav_bytes = _audio_to_wav_bytes(audio, sr)
    duration = audio.shape[-1] / sr
    gen_time = time.perf_counter() - t0

    logger.info(f"[TTS] Done: {duration:.1f}s audio in {gen_time:.1f}s (RTF={gen_time/duration:.2f}x)")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )


@app.post("/tts/clone")
async def clone_voice(
    text: str = Form(..., description="Text to synthesize"),
    reference: UploadFile = File(..., description="Reference audio file for voice cloning"),
    max_new_tokens: int = Form(4096, description="Max audio tokens"),
):
    """Generate speech cloning the voice from the uploaded reference audio."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Save uploaded reference to a temp file
    suffix = Path(reference.filename).suffix if reference.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await reference.read()
        tmp.write(content)

    logger.info(f"[Clone] Text: '{text[:80]}...', ref: {reference.filename} ({len(content)} bytes)")
    t0 = time.perf_counter()

    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(
                _generate_audio,
                text=text,
                reference_path=tmp_path,
                max_new_tokens=max_new_tokens,
            )
    except Exception as e:
        logger.error(f"[Clone] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    wav_bytes = _audio_to_wav_bytes(audio, sr)
    duration = audio.shape[-1] / sr
    gen_time = time.perf_counter() - t0

    logger.info(f"[Clone] Done: {duration:.1f}s audio in {gen_time:.1f}s")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )


@app.post("/tts/stream")
async def stream_speech(request: TTSRequest):
    """Generate speech and stream WAV chunks progressively.

    The full audio is generated first, then streamed in chunks.
    This allows the client to start playback before the entire
    response is buffered.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(f"[Stream] Generating: '{request.text[:80]}...'")

    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(
                _generate_audio,
                text=request.text,
                max_new_tokens=request.max_new_tokens,
                tokens=request.tokens,
                audio_temperature=request.audio_temperature,
                audio_top_p=request.audio_top_p,
                audio_top_k=request.audio_top_k,
            )
    except Exception as e:
        logger.error(f"[Stream] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Build complete WAV in memory, then stream in chunks
    wav_bytes = _audio_to_wav_bytes(audio, sr)
    chunk_size = 32 * 1024  # 32KB chunks

    async def wav_chunker():
        offset = 0
        while offset < len(wav_bytes):
            yield wav_bytes[offset:offset + chunk_size]
            offset += chunk_size

    duration = audio.shape[-1] / sr
    logger.info(f"[Stream] Streaming {duration:.1f}s ({len(wav_bytes)} bytes)")

    return StreamingResponse(
        wav_chunker(),
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Sample-Rate": str(sr),
            "X-Streaming": "true",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/voices/{voice_name}")
async def upload_voice(
    voice_name: str,
    file: UploadFile = File(..., description="Audio file for voice reference"),
):
    """Upload a reference audio file for a named voice."""
    voice_dir = VOICES_DIR / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "reference.wav"
    dest = voice_dir / filename

    content = await file.read()
    dest.write_bytes(content)

    logger.info(f"[Voice] Saved {filename} to voice '{voice_name}' ({len(content)} bytes)")

    return {"voice_name": voice_name, "filename": filename, "size_bytes": len(content)}


@app.post("/tts/clone/{voice_name}")
async def clone_from_saved_voice(
    voice_name: str,
    request: TTSRequest,
):
    """Generate speech using a previously uploaded voice as reference."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    voice_dir = VOICES_DIR / voice_name
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

    # Find the first audio file in the voice directory
    audio_files = [f for f in voice_dir.iterdir()
                   if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
    if not audio_files:
        raise HTTPException(status_code=404, detail=f"No audio files found for voice '{voice_name}'")

    ref_path = str(audio_files[0])
    logger.info(f"[Clone/{voice_name}] Text: '{request.text[:80]}...', ref: {ref_path}")
    t0 = time.perf_counter()

    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(
                _generate_audio,
                text=request.text,
                reference_path=ref_path,
                max_new_tokens=request.max_new_tokens,
                tokens=request.tokens,
                audio_temperature=request.audio_temperature,
                audio_top_p=request.audio_top_p,
                audio_top_k=request.audio_top_k,
            )
    except Exception as e:
        logger.error(f"[Clone/{voice_name}] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    wav_bytes = _audio_to_wav_bytes(audio, sr)
    duration = audio.shape[-1] / sr
    gen_time = time.perf_counter() - t0

    logger.info(f"[Clone/{voice_name}] Done: {duration:.1f}s audio in {gen_time:.1f}s")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )


# ── Entrypoint ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
