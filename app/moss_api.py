"""
MOSS-TTS FastAPI Service
Exposes voice generation, voice cloning, voice design, and streaming TTS endpoints.
Includes API shim for frontend compatibility (GLM/Qwen endpoint shapes).
"""

import asyncio
import importlib.util
import io
import json as _json
import logging
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moss-tts")

# ── Config ──
MODEL_ID = os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS")
VOICE_GEN_MODEL_ID = os.environ.get("MOSS_VOICE_GEN_MODEL", "OpenMOSS-Team/MOSS-VoiceGenerator")
ENABLE_VOICE_GEN = os.environ.get("MOSS_ENABLE_VOICE_GEN", "true").lower() == "true"
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
model_device = None  # actual device for input tensors (may differ from DEVICE with device_map)

# ── VoiceGenerator model (loaded separately for voice design) ──
voice_gen_model = None
voice_gen_processor = None
voice_gen_device = None
voice_gen_sample_rate = None

# ── MOSS-TTS-Realtime model (true streaming) ──
RT_MODEL_ID = os.environ.get("MOSS_RT_MODEL_ID", "OpenMOSS-Team/MOSS-TTS-Realtime")
RT_CODEC_ID = os.environ.get("MOSS_RT_CODEC_ID", "OpenMOSS-Team/MOSS-Audio-Tokenizer")
ENABLE_REALTIME = os.environ.get("MOSS_ENABLE_REALTIME", "true").lower() == "true"
rt_model = None
rt_tokenizer = None
rt_processor = None
rt_inferencer = None
rt_codec = None
rt_sample_rate = 24000


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
    # NOTE: audio_tokenizer stays on CPU here — moved to GPU AFTER model loads
    # to avoid OOM during quantized model loading (peak memory is higher than settled)

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
            model_kwargs["device_map"] = {"": 0}  # Force all layers on GPU 0
            model_kwargs["low_cpu_mem_usage"] = True
            logger.info("Using 4-bit quantization (NF4) on single GPU")
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

    # Detect the actual device for input tensors
    global model_device
    if hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        model_device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    else:
        model_device = next(model.parameters()).device
    logger.info(f"Model input device: {model_device}")

    # Now move audio tokenizer to GPU (deferred to avoid OOM during model loading)
    processor.audio_tokenizer = processor.audio_tokenizer.to(model_device)
    logger.info(f"Audio tokenizer moved to {model_device}")

    sample_rate = processor.model_config.sampling_rate
    logger.info(f"Model loaded. Sample rate: {sample_rate} Hz")

    _log_gpu_memory()


def _log_gpu_memory():
    """Log GPU memory usage for all visible devices."""
    if DEVICE == "cuda":
        for i in range(torch.cuda.device_count()):
            mem_gb = torch.cuda.memory_allocated(i) / 1e9
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {mem_gb:.2f} / {total_gb:.2f} GB")


def load_voice_generator():
    """Load MOSS-VoiceGenerator model on a separate GPU for voice design."""
    global voice_gen_model, voice_gen_processor, voice_gen_device, voice_gen_sample_rate
    from transformers import AutoModel, AutoProcessor

    # Determine which GPU to use (second available GPU, or CPU)
    num_gpus = torch.cuda.device_count() if DEVICE == "cuda" else 0
    if num_gpus >= 2:
        vg_gpu = 1  # container's second GPU
    elif num_gpus == 1:
        vg_gpu = 0  # share with TTS model (may be tight)
        logger.warning("Only 1 GPU available — VoiceGenerator will share GPU 0 with TTS")
    else:
        vg_gpu = None

    logger.info(f"Loading VoiceGenerator processor from {VOICE_GEN_MODEL_ID} ...")
    voice_gen_processor = AutoProcessor.from_pretrained(
        VOICE_GEN_MODEL_ID, trust_remote_code=True, normalize_inputs=True,
    )

    vg_device = torch.device(f"cuda:{vg_gpu}") if vg_gpu is not None else torch.device("cpu")
    vg_dtype = torch.bfloat16 if vg_device.type == "cuda" else torch.float32

    # Resolve attention impl for this device
    vg_attn = None
    if vg_device.type == "cuda":
        if importlib.util.find_spec("flash_attn") is not None and vg_dtype in {torch.float16, torch.bfloat16}:
            major, _ = torch.cuda.get_device_capability(vg_device)
            if major >= 8:
                vg_attn = "flash_attention_2"
        if vg_attn is None:
            vg_attn = "sdpa"
    else:
        vg_attn = "eager"

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": vg_attn,
    }

    # Apply same quantization strategy
    if QUANTIZE == "4bit" and vg_device.type == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=vg_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": vg_gpu}
            model_kwargs["low_cpu_mem_usage"] = True
            logger.info(f"VoiceGenerator: 4-bit NF4 on GPU {vg_gpu}")
        except ImportError:
            model_kwargs["torch_dtype"] = vg_dtype
    else:
        model_kwargs["torch_dtype"] = vg_dtype

    logger.info(f"Loading VoiceGenerator model from {VOICE_GEN_MODEL_ID} (attn={vg_attn}) ...")
    voice_gen_model = AutoModel.from_pretrained(VOICE_GEN_MODEL_ID, **model_kwargs)

    if "device_map" not in model_kwargs:
        voice_gen_model = voice_gen_model.to(vg_device)

    voice_gen_model.eval()

    # Detect actual device
    if hasattr(voice_gen_model, 'hf_device_map'):
        first_dev = next(iter(voice_gen_model.hf_device_map.values()))
        voice_gen_device = torch.device(f"cuda:{first_dev}" if isinstance(first_dev, int) else first_dev)
    else:
        voice_gen_device = next(voice_gen_model.parameters()).device

    if hasattr(voice_gen_processor, 'audio_tokenizer'):
        voice_gen_processor.audio_tokenizer = voice_gen_processor.audio_tokenizer.to(voice_gen_device)

    voice_gen_sample_rate = getattr(voice_gen_processor.model_config, 'sampling_rate', 24000)
    logger.info(f"VoiceGenerator loaded on {voice_gen_device}. Sample rate: {voice_gen_sample_rate} Hz")
    _log_gpu_memory()


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

# NOTE: Audiobook logic (projects, visuals, export) lives exclusively in `tts-api`.
# Nginx routes /audiobook/* -> tts-api:8000, /tts and /voices -> moss-tts:8000.
# This container only handles TTS inference — no audiobook router needed.


def load_realtime_model():
    """Load MOSS-TTS-Realtime model + codec for true token-level streaming."""
    global rt_model, rt_tokenizer, rt_processor, rt_inferencer, rt_codec
    import sys
    # Ensure the mossttsrealtime package is importable
    rt_pkg_dir = "/app/MOSS-TTS/moss_tts_realtime"
    if rt_pkg_dir not in sys.path:
        sys.path.insert(0, rt_pkg_dir)

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (
        MossTTSRealtimeInference as StreamingInferencer,
    )
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Loading MOSS-TTS-Realtime tokenizer from {RT_MODEL_ID} ...")
    rt_tokenizer = AutoTokenizer.from_pretrained(RT_MODEL_ID)
    rt_processor = MossTTSRealtimeProcessor(tokenizer=rt_tokenizer)

    logger.info(f"Loading MOSS-TTS-Realtime model from {RT_MODEL_ID} (attn={ATTN_IMPL}) ...")
    rt_model = MossTTSRealtime.from_pretrained(
        RT_MODEL_ID,
        attn_implementation=ATTN_IMPL,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    rt_model.eval()

    logger.info(f"Loading audio codec from {RT_CODEC_ID} ...")
    rt_codec = AutoModel.from_pretrained(RT_CODEC_ID, trust_remote_code=True).eval()
    rt_codec = rt_codec.to(DEVICE)

    # Streaming version of inferencer — no codec args (codec is passed to session)
    rt_inferencer = StreamingInferencer(
        rt_model, rt_tokenizer, max_length=5000,
    )

    logger.info("MOSS-TTS-Realtime loaded successfully")
    _log_gpu_memory()


@app.on_event("startup")
async def startup_event():
    if ENABLE_REALTIME:
        # Load ONLY the realtime model (base model won't fit alongside)
        try:
            load_realtime_model()
        except Exception as e:
            logger.error(f"Failed to load MOSS-TTS-Realtime: {e}", exc_info=True)
            logger.warning("Falling back to base MOSS-TTS model")
            load_model()
    else:
        load_model()

    if ENABLE_VOICE_GEN:
        try:
            load_voice_generator()
        except Exception as e:
            logger.error(f"Failed to load VoiceGenerator: {e}")
            logger.warning("Voice design endpoint will be unavailable")


# ── Request / Response models ──

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice_name: Optional[str] = Field(None, description="Voice name for cloning (uses saved reference)")
    language: Optional[str] = Field("en", description="Language code (informational)")
    output_format: Optional[str] = Field("wav", description="Output format (wav)")
    max_new_tokens: int = Field(4096, ge=256, le=16384, description="Max audio tokens to generate")
    audio_temperature: Optional[float] = Field(None, description="Sampling temperature for audio tokens")
    audio_top_p: Optional[float] = Field(None, description="Top-p sampling for audio tokens")
    audio_top_k: Optional[int] = Field(None, description="Top-k sampling for audio tokens")
    tokens: Optional[int] = Field(None, description="Target duration in tokens (for duration control)")
    # GLM-TTS compat params (silently accepted, not used)
    sampling: Optional[int] = Field(None, description="(GLM compat) Ignored")
    temperature: Optional[float] = Field(None, description="(GLM compat) Maps to audio_temperature")
    top_p: Optional[float] = Field(None, description="(GLM compat) Maps to audio_top_p")
    beam_size: Optional[int] = Field(None, description="(GLM compat) Ignored")
    repetition_penalty: Optional[float] = Field(None, description="(GLM compat) Ignored")
    sample_method: Optional[str] = Field(None, description="(GLM compat) Ignored")
    min_token_text_ratio: Optional[float] = Field(None, description="(GLM compat) Ignored")
    max_token_text_ratio: Optional[float] = Field(None, description="(GLM compat) Ignored")


class VoiceDesignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    instruction: Optional[str] = Field(None, description="Voice description instruction")
    instruct: Optional[str] = Field(None, description="Voice description (Qwen compat alias)")
    mode: Optional[str] = Field(None, description="Mode: 'voice_design' or 'custom_voice' (Qwen compat)")
    speaker: Optional[str] = Field(None, description="Speaker name for custom_voice mode (Qwen compat)")
    language: Optional[str] = Field("en", description="Language code (informational)")
    max_new_tokens: int = Field(4096, ge=256, le=16384, description="Max audio tokens")
    audio_temperature: Optional[float] = Field(1.5, description="Sampling temperature")
    audio_top_p: Optional[float] = Field(0.6, description="Top-p sampling")
    audio_top_k: Optional[int] = Field(50, description="Top-k sampling")
    audio_repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")


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

# ── Text splitting helpers ──

import re as _re

def _split_into_sentences(text: str, max_chunk: int = 300) -> list[str]:
    """Split text into sentence-level chunks for streaming generation.

    Uses sentence-ending punctuation (.!?) as primary split points.
    Falls back to commas/semicolons for overly long sentences.
    Guarantees every chunk is ≤ max_chunk characters.
    """
    text = text.strip()
    if not text:
        return [text] if text else [""]

    # Primary split: sentence boundaries (.!? followed by space or end)
    raw = _re.split(r'(?<=[.!?])\s+', text)

    # Secondary split: break long chunks on commas/semicolons
    chunks: list[str] = []
    for piece in raw:
        if len(piece) <= max_chunk:
            chunks.append(piece)
        else:
            # Try comma / semicolon
            sub = _re.split(r'(?<=[,;])\s+', piece)
            for s in sub:
                if len(s) <= max_chunk:
                    chunks.append(s)
                else:
                    # Hard split at max_chunk on word boundaries
                    words = s.split()
                    buf = ""
                    for w in words:
                        if buf and len(buf) + 1 + len(w) > max_chunk:
                            chunks.append(buf)
                            buf = w
                        else:
                            buf = f"{buf} {w}" if buf else w
                    if buf:
                        chunks.append(buf)

    # Filter empty and strip
    return [c.strip() for c in chunks if c.strip()]


def _list_voice_names() -> list[str]:
    """Return sorted list of available voice directory names."""
    if not VOICES_DIR.exists():
        return []
    return sorted(
        d.name for d in VOICES_DIR.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")
            for f in d.iterdir()
        )
    )


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

    input_ids = batch["input_ids"].to(model_device)
    attention_mask = batch["attention_mask"].to(model_device)

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
    import tempfile
    # torchaudio's torchcodec backend doesn't support BytesIO,
    # so we write to a temp file instead
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torchaudio.save(tmp_path, audio.unsqueeze(0).cpu(), sr, format="wav")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _generate_audio_realtime(text: str, reference_path: Optional[str] = None) -> tuple[torch.Tensor, int]:
    """Non-streaming generation using the MOSS-TTS-Realtime model.

    Used by /tts endpoint when the base model is not loaded.
    """
    device = torch.device(DEVICE)
    ref_audio_path = reference_path if reference_path else None

    with torch.inference_mode():
        result = rt_inferencer.generate(
            text=text,
            reference_audio_path=ref_audio_path,
            temperature=0.8,
            top_p=0.6,
            top_k=30,
            repetition_penalty=1.1,
            repetition_window=50,
            device=device,
        )

        # result is a list of token arrays, one per text input
        generated_tokens = result[0]  # numpy array [T, 16]
        output = torch.tensor(generated_tokens).to(device)
        decode_result = rt_codec.decode(output.permute(1, 0), chunk_duration=8)
        wav = decode_result["audio"][0].cpu().detach()  # [1, samples]

        if wav.dim() == 2:
            wav = wav.squeeze(0)  # [samples]

    return wav, rt_sample_rate


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_mem = None
    if DEVICE == "cuda":
        gpu_mem = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    is_ready = model is not None or rt_inferencer is not None
    active_model = RT_MODEL_ID if rt_inferencer is not None else MODEL_ID
    active_sr = rt_sample_rate if rt_inferencer is not None and sample_rate is None else sample_rate

    return HealthResponse(
        status="ready" if is_ready else "loading",
        model_id=active_model,
        device=DEVICE,
        attention=ATTN_IMPL,
        sample_rate=active_sr,
        gpu_memory_gb=gpu_mem,
    )


@app.get("/voices")
async def list_voices(format: Optional[str] = Query(None, description="'flat' for name-only list")):
    """List available reference voices. Default returns flat list for frontend compat."""
    voices = []
    if VOICES_DIR.exists():
        for voice_dir in sorted(VOICES_DIR.iterdir()):
            if voice_dir.is_dir():
                files = [f.name for f in voice_dir.iterdir()
                         if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
                if files:
                    voices.append(VoiceInfo(name=voice_dir.name, files=sorted(files)))
    # Default: return flat list of names (frontend expects ["name1", "name2"])
    if format == "detailed":
        return voices
    return JSONResponse(content=[v.name for v in voices])


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate speech from text. If voice_name is provided, auto-routes to cloning."""
    if model is None and rt_inferencer is None:
        raise HTTPException(status_code=503, detail="No TTS model loaded yet")

    # Map GLM compat params
    audio_temp = request.audio_temperature or request.temperature
    audio_tp = request.audio_top_p or request.top_p

    # Auto-route to voice cloning if voice_name provided
    ref_path = None
    if request.voice_name:
        voice_dir = VOICES_DIR / request.voice_name
        if voice_dir.exists():
            audio_files = [f for f in voice_dir.iterdir()
                           if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
            if audio_files:
                ref_path = str(audio_files[0])
                logger.info(f"[TTS] Auto-cloning with voice '{request.voice_name}': {ref_path}")

    logger.info(f"[TTS] Generating: '{request.text[:80]}...'")
    t0 = time.perf_counter()

    try:
        async with inference_semaphore:
            if model is not None:
                # Use base MOSS-TTS model
                audio, sr = await asyncio.to_thread(
                    _generate_audio,
                    text=request.text,
                    reference_path=ref_path,
                    max_new_tokens=request.max_new_tokens,
                    tokens=request.tokens,
                    audio_temperature=audio_temp,
                    audio_top_p=audio_tp,
                    audio_top_k=request.audio_top_k,
                )
            else:
                # Use realtime model's non-streaming generate()
                audio, sr = await asyncio.to_thread(
                    _generate_audio_realtime,
                    text=request.text,
                    reference_path=ref_path,
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
    """Stream speech audio in real-time using MOSS-TTS-Realtime.

    When the realtime model is loaded, text is pushed character-by-character
    and audio frames are decoded and streamed incrementally (~200ms latency).
    Falls back to sentence-level chunking if the realtime model is unavailable.

    Binary framing (same as GLM-TTS streaming):
        4-byte audio_len (LE) + 4-byte metadata_len (LE) + audio_bytes + metadata_json
    """
    if model is None and rt_inferencer is None:
        raise HTTPException(status_code=503, detail="No TTS model loaded yet")

    # ── Voice validation ─────────────────────────────────────────────
    ref_path = None
    if request.voice_name:
        voice_dir = VOICES_DIR / request.voice_name
        if not voice_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{request.voice_name}' not found. "
                       f"Available voices: {', '.join(_list_voice_names())}",
            )
        audio_files = [f for f in voice_dir.iterdir()
                       if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
        if audio_files:
            ref_path = str(audio_files[0])

    # ── Route: true realtime streaming or sentence-level fallback ─────
    if rt_inferencer is not None:
        return await _stream_realtime(request, ref_path)
    else:
        return await _stream_sentence_fallback(request, ref_path)


async def _stream_realtime(request: TTSRequest, ref_path: Optional[str]):
    """True token-level streaming using MOSS-TTS-Realtime."""
    import sys
    rt_pkg_dir = "/app/MOSS-TTS/moss_tts_realtime"
    if rt_pkg_dir not in sys.path:
        sys.path.insert(0, rt_pkg_dir)

    from mossttsrealtime.streaming_mossttsrealtime import (
        MossTTSRealtimeStreamingSession,
        AudioStreamDecoder,
    )

    device = torch.device(DEVICE)

    logger.info(
        f"[Stream-RT] True realtime streaming: "
        f"voice={request.voice_name or 'default'}, "
        f"text_len={len(request.text)}"
    )

    # ── Encode reference audio for voice cloning ─────────────────────
    prompt_tokens = None
    if ref_path:
        with torch.inference_mode():
            wav, sr_wav = torchaudio.load(ref_path)
            if sr_wav != rt_sample_rate:
                wav = torchaudio.functional.resample(wav, sr_wav, rt_sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            encode_result = rt_codec.encode(
                wav.unsqueeze(0).to(device),
                chunk_duration=0.24,
            )
            prompt_tokens = encode_result["audio_codes"].cpu().numpy().squeeze(1)

    async def realtime_streamer():
        t0 = time.perf_counter()
        chunk_idx = 0
        total_audio_duration = 0.0

        try:
            # ── Create streaming session ─────────────────────────
            session = MossTTSRealtimeStreamingSession(
                rt_inferencer,
                rt_processor,
                codec=rt_codec,
                codec_sample_rate=rt_sample_rate,
                codec_encode_kwargs={"chunk_duration": 0.24},
                prefill_text_len=rt_processor.delay_tokens_len,
                temperature=request.temperature or 0.8,
                top_p=request.top_p or 0.6,
                top_k=30,
                do_sample=True,
                repetition_penalty=1.1,
                repetition_window=50,
            )

            if prompt_tokens is not None:
                session.set_voice_prompt_tokens(prompt_tokens)

            # ── Build input_ids ──────────────────────────────────
            system_prompt = rt_processor.make_ensemble(prompt_tokens)
            assistant_prefix_ids = rt_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
            assistant_prefix = np.full(
                (len(assistant_prefix_ids), system_prompt.shape[1]),
                fill_value=rt_processor.audio_channel_pad,
                dtype=np.int64,
            )
            assistant_prefix[:, 0] = assistant_prefix_ids
            input_ids = np.concatenate([system_prompt, assistant_prefix], axis=0)

            rt_inferencer.reset_generation_state(keep_cache=False)
            session.reset_turn(
                input_ids=input_ids,
                include_system_prompt=False,
                reset_cache=True,
            )

            # ── Audio stream decoder ─────────────────────────────
            decoder = AudioStreamDecoder(
                rt_codec,
                chunk_frames=12,
                overlap_frames=0,
                decode_kwargs={"chunk_duration": -1},
                device=device,
            )

            codebook_size = int(getattr(rt_codec, "codebook_size", 1024))
            audio_eos_token = int(getattr(rt_inferencer, "audio_eos_token", 1026))

            def _sanitize_tokens(tokens):
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                if tokens.numel() == 0:
                    return tokens, False
                eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
                invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
                stop_idx = None
                if eos_rows.numel() > 0:
                    stop_idx = int(eos_rows[0].item())
                if invalid_rows.any():
                    inv_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
                    stop_idx = inv_idx if stop_idx is None else min(stop_idx, inv_idx)
                if stop_idx is not None:
                    return tokens[:stop_idx], True
                return tokens, False

            def decode_frames(audio_frames):
                for frame in audio_frames:
                    tokens = frame
                    if tokens.dim() == 3:
                        tokens = tokens[0]
                    tokens, _ = _sanitize_tokens(tokens)
                    if tokens.numel() == 0:
                        continue
                    decoder.push_tokens(tokens.detach())
                    for wav_chunk in decoder.audio_chunks():
                        if wav_chunk.numel() == 0:
                            continue
                        yield wav_chunk.detach().cpu().numpy().reshape(-1)

            def flush_dec():
                final = decoder.flush()
                if final is not None and final.numel() > 0:
                    yield final.detach().cpu().numpy().reshape(-1)

            # ── Run streaming generation in thread ───────────────
            # Batch small numpy chunks to reduce queue overhead
            MIN_SAMPLES = int(rt_sample_rate * 0.3)  # ~0.3s of audio

            # Force-reset any leftover codec streaming state from prior requests
            # (the codec's StreamingModule sets _streaming_state on sub-modules
            # which doesn't get cleared if the thread exits abnormally)
            if hasattr(rt_codec, 'is_streaming') and rt_codec.is_streaming():
                logger.warning("[Stream-RT] Force-resetting stuck codec streaming state")
                try:
                    rt_codec._stop_streaming()
                except Exception:
                    # Fallback: manually clear all _streaming_state
                    for name, module in rt_codec.named_modules():
                        if hasattr(module, '_streaming_state') and module._streaming_state is not None:
                            module._streaming_state = None

            def _run_stream():
                """Synchronous generator: push text → audio chunks.
                Matches the official Gradio app approach: pre-tokenize text,
                push 12 tokens at a time via push_text_tokens().
                """
                TEXT_CHUNK_TOKENS = 12  # Official Gradio default

                with torch.inference_mode():
                    with rt_codec.streaming(batch_size=1):
                        buf = []
                        buf_samples = 0

                        def _flush_buf():
                            nonlocal buf, buf_samples
                            if buf:
                                merged = np.concatenate(buf)
                                buf = []
                                buf_samples = 0
                                return merged
                            return None

                        # Pre-tokenize the full text (like the Gradio app does)
                        text_tokens = rt_tokenizer.encode(
                            request.text, add_special_tokens=False,
                        )
                        if not text_tokens:
                            return

                        # Push tokens in chunks of TEXT_CHUNK_TOKENS
                        for i in range(0, len(text_tokens), TEXT_CHUNK_TOKENS):
                            token_chunk = text_tokens[i:i + TEXT_CHUNK_TOKENS]
                            audio_frames = session.push_text_tokens(token_chunk)
                            for chunk in decode_frames(audio_frames):
                                buf.append(chunk)
                                buf_samples += len(chunk)
                                if buf_samples >= MIN_SAMPLES:
                                    yield _flush_buf()

                        audio_frames = session.end_text()
                        for chunk in decode_frames(audio_frames):
                            buf.append(chunk)
                            buf_samples += len(chunk)
                            if buf_samples >= MIN_SAMPLES:
                                yield _flush_buf()

                        while True:
                            audio_frames = session.drain(max_steps=1)
                            if not audio_frames:
                                break
                            for chunk in decode_frames(audio_frames):
                                buf.append(chunk)
                                buf_samples += len(chunk)
                                if buf_samples >= MIN_SAMPLES:
                                    yield _flush_buf()
                            if session.inferencer.is_finished:
                                break

                        for chunk in flush_dec():
                            buf.append(chunk)
                            buf_samples += len(chunk)

                        merged = _flush_buf()
                        if merged is not None:
                            yield merged

            # In-memory WAV header builder (avoids temp-file disk I/O)
            def _wav_bytes_from_pcm(pcm_float: np.ndarray, sr: int) -> bytes:
                pcm16 = np.clip(pcm_float, -1.0, 1.0)
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
                return header + raw

            import queue
            import threading

            q = queue.Queue()
            error_holder = [None]

            def _producer():
                try:
                    for wav_np in _run_stream():
                        q.put(wav_np)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    q.put(None)  # sentinel

            async with inference_semaphore:
                thread = threading.Thread(target=_producer, daemon=True)
                thread.start()

                while True:
                    wav_np = await asyncio.to_thread(q.get, True, 30.0)
                    if wav_np is None:
                        break

                    # In-memory WAV — no disk I/O
                    wav_bytes = _wav_bytes_from_pcm(wav_np, rt_sample_rate)

                    chunk_duration = len(wav_np) / rt_sample_rate
                    total_audio_duration += chunk_duration

                    meta = {
                        "chunk_index": chunk_idx,
                        "is_final": False,
                        "sample_rate": rt_sample_rate,
                        "audio_duration": round(chunk_duration, 3),
                        "total_audio_duration": round(total_audio_duration, 2),
                        "generation_time": round(time.perf_counter() - t0, 2),
                        "streaming": "realtime",
                    }
                    meta_bytes = _json.dumps(meta).encode()
                    header = struct.pack("<II", len(wav_bytes), len(meta_bytes))
                    yield header + wav_bytes + meta_bytes

                    chunk_idx += 1

                thread.join(timeout=10.0)
                if thread.is_alive():
                    logger.warning("[Stream-RT] Producer thread still alive after join! Codec may be stuck.")

            if error_holder[0]:
                logger.error(f"[Stream-RT] Error: {error_holder[0]}")
                meta = {"error": str(error_holder[0]), "is_final": True}
                meta_bytes = _json.dumps(meta).encode()
                yield struct.pack("<II", 0, len(meta_bytes)) + meta_bytes
                return

            # Send final metadata chunk
            meta = {
                "chunk_index": chunk_idx,
                "is_final": True,
                "total_audio_duration": round(total_audio_duration, 2),
                "total_generation_time": round(time.perf_counter() - t0, 2),
                "streaming": "realtime",
            }
            meta_bytes = _json.dumps(meta).encode()
            yield struct.pack("<II", 0, len(meta_bytes)) + meta_bytes

            logger.info(
                f"[Stream-RT] Done: {total_audio_duration:.1f}s audio, "
                f"{chunk_idx} chunks in {time.perf_counter() - t0:.1f}s"
            )

        except Exception as e:
            logger.error(f"[Stream-RT] Error: {e}", exc_info=True)
            meta = {"error": str(e), "is_final": True}
            meta_bytes = _json.dumps(meta).encode()
            yield struct.pack("<II", 0, len(meta_bytes)) + meta_bytes

    return StreamingResponse(
        realtime_streamer(),
        media_type="application/octet-stream",
        headers={
            "X-Streaming": "true",
            "X-Streaming-Mode": "realtime",
            "X-Sample-Rate": str(rt_sample_rate),
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


async def _stream_sentence_fallback(request: TTSRequest, ref_path: Optional[str]):
    """Fallback: sentence-level chunked streaming using base MOSS-TTS model."""
    audio_temp = request.audio_temperature or request.temperature
    audio_tp = request.audio_top_p or request.top_p
    sentences = _split_into_sentences(request.text)
    total_chunks = len(sentences)

    logger.info(f"[Stream-Fallback] {total_chunks} sentence chunks")

    async def sentence_streamer():
        for idx, chunk_text in enumerate(sentences):
            is_final = idx == total_chunks - 1
            t0 = time.perf_counter()
            chunk_max_tokens = min(max(512, len(chunk_text) * 30), request.max_new_tokens)

            try:
                async with inference_semaphore:
                    audio, sr = await asyncio.to_thread(
                        _generate_audio,
                        text=chunk_text,
                        reference_path=ref_path,
                        max_new_tokens=chunk_max_tokens,
                        tokens=request.tokens,
                        audio_temperature=audio_temp,
                        audio_top_p=audio_tp,
                        audio_top_k=request.audio_top_k,
                    )
            except Exception as e:
                logger.error(f"[Stream-Fallback] Chunk {idx} failed: {e}")
                meta = {"error": str(e), "is_final": True}
                meta_bytes = _json.dumps(meta).encode()
                yield struct.pack("<II", 0, len(meta_bytes)) + meta_bytes
                return

            gen_time = time.perf_counter() - t0
            duration = audio.shape[-1] / sr
            wav_bytes = _audio_to_wav_bytes(audio, sr)

            meta = {
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text[:200],
                "is_final": is_final,
                "sample_rate": sr,
                "audio_duration": round(duration, 2),
                "generation_time": round(gen_time, 2),
                "streaming": "sentence",
            }
            meta_bytes = _json.dumps(meta).encode()
            header = struct.pack("<II", len(wav_bytes), len(meta_bytes))
            yield header + wav_bytes + meta_bytes

            logger.info(
                f"[Stream-Fallback] Chunk {idx + 1}/{total_chunks}: "
                f"{duration:.1f}s audio in {gen_time:.1f}s"
            )

    return StreamingResponse(
        sentence_streamer(),
        media_type="application/octet-stream",
        headers={
            "X-Streaming": "true",
            "X-Streaming-Mode": "sentence",
            "X-Sample-Rate": str(sample_rate),
            "X-Total-Chunks": str(total_chunks),
            "Cache-Control": "no-cache",
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


# ── Frontend compatibility shim: POST /voices?voice_name=X ──
@app.post("/voices")
async def upload_voice_compat(
    voice_name: str = Query(..., description="Voice name"),
    file: UploadFile = File(..., description="Audio file for voice reference"),
    transcription: Optional[str] = Form(None, description="(GLM compat) Transcription text, ignored"),
):
    """Frontend-compatible voice upload (query param style)."""
    return await upload_voice(voice_name=voice_name, file=file)


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


# ── Voice Design (MOSS-VoiceGenerator) ──

def _generate_voice_design(text: str, instruction: str,
                           max_new_tokens: int = 4096, **gen_kwargs) -> tuple[torch.Tensor, int]:
    """Generate speech from voice description — runs in thread pool."""
    conversation = [voice_gen_processor.build_user_message(text=text, instruction=instruction)]
    batch = voice_gen_processor([conversation], mode="generation")

    input_ids = batch["input_ids"].to(voice_gen_device)
    attention_mask = batch["attention_mask"].to(voice_gen_device)

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    for k in ("audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty"):
        if k in gen_kwargs and gen_kwargs[k] is not None:
            generate_kwargs[k] = gen_kwargs[k]

    with torch.no_grad():
        outputs = voice_gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    messages = list(voice_gen_processor.decode(outputs))
    if not messages or not messages[0].audio_codes_list:
        raise RuntimeError("VoiceGenerator returned no audio")

    audio = messages[0].audio_codes_list[0]
    return audio, voice_gen_sample_rate


@app.post("/tts/design")
async def design_voice(request: VoiceDesignRequest):
    """Generate speech using voice description (no reference audio needed)."""
    if voice_gen_model is None:
        raise HTTPException(status_code=503, detail="VoiceGenerator not loaded")

    instruction = request.instruction or request.instruct
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction or instruct is required")

    logger.info(f"[Design] Text: '{request.text[:80]}...', instruction: '{instruction[:80]}...'")
    t0 = time.perf_counter()

    try:
        async with inference_semaphore:
            audio, sr = await asyncio.to_thread(
                _generate_voice_design,
                text=request.text,
                instruction=instruction,
                max_new_tokens=request.max_new_tokens,
                audio_temperature=request.audio_temperature,
                audio_top_p=request.audio_top_p,
                audio_top_k=request.audio_top_k,
                audio_repetition_penalty=request.audio_repetition_penalty,
            )
    except Exception as e:
        logger.error(f"[Design] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    wav_bytes = _audio_to_wav_bytes(audio, sr)
    duration = audio.shape[-1] / sr
    gen_time = time.perf_counter() - t0

    logger.info(f"[Design] Done: {duration:.1f}s audio in {gen_time:.1f}s")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )


# ── Qwen-compatible alias routes (frontend VoiceStudio) ──

@app.post("/api/v1/qwen/synthesize")
async def qwen_synthesize_compat(request: VoiceDesignRequest):
    """Frontend-compatible synthesize endpoint (Qwen API shape).
    Routes based on mode:
    - 'custom_voice' + speaker: clone from saved voice
    - 'voice_design' + instruct: generate with VoiceGenerator
    """
    mode = request.mode or ""
    instruct = request.instruction or request.instruct or ""

    if mode == "custom_voice" and request.speaker:
        # Route to TTS with voice cloning using saved speaker
        # Map speaker display name back to directory name
        voice_name = request.speaker.lower().replace(" ", "_")
        tts_request = TTSRequest(
            text=request.text,
            voice_name=voice_name,
            language=request.language or "en",
            max_new_tokens=request.max_new_tokens,
            audio_temperature=request.audio_temperature,
            audio_top_p=request.audio_top_p,
            audio_top_k=request.audio_top_k,
        )
        return await generate_speech(tts_request)
    else:
        # Route to voice design (VoiceGenerator)
        return await design_voice(request)


@app.post("/api/v1/qwen/clone")
async def qwen_clone_compat(
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query("en", description="Language code"),
    use_xvector_only: str = Query("false", description="(Qwen compat) Ignored"),
    ref_audio: UploadFile = File(..., description="Reference audio file"),
):
    """Frontend-compatible voice cloning endpoint (Qwen API shape)."""
    return await clone_voice(text=text, reference=ref_audio)


@app.post("/api/v1/qwen/voices/create-prompt")
async def qwen_create_prompt_compat(
    voice_id: str = Query(..., description="Voice name/ID"),
    ref_audio: UploadFile = File(..., description="Reference audio file"),
):
    """Frontend-compatible voice creation endpoint (Qwen API shape)."""
    return await upload_voice(voice_name=voice_id, file=ref_audio)


@app.get("/api/v1/qwen/speakers")
async def qwen_speakers_compat():
    """Frontend-compatible speakers list (Qwen API shape)."""
    voices = []
    if VOICES_DIR.exists():
        for voice_dir in sorted(VOICES_DIR.iterdir()):
            if voice_dir.is_dir():
                audio_files = [f for f in voice_dir.iterdir()
                               if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
                if audio_files:
                    voices.append({
                        "id": voice_dir.name,
                        "name": voice_dir.name.replace("_", " ").title(),
                        "description": f"Cloned voice ({len(audio_files)} reference files)",
                        "native_language": "English",
                        "gender": "unknown",
                    })
    return {"speakers": voices}


# ── Entrypoint ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

