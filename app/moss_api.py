"""
MOSS-TTS FastAPI Service
Exposes voice generation, voice cloning, voice design, and streaming TTS endpoints.
Includes API shim for frontend compatibility (GLM/Qwen endpoint shapes).
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


@app.on_event("startup")
async def startup_event():
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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

