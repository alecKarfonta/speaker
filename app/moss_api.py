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
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Match official Gradio app (prevent recompilation)
torch.set_float32_matmul_precision('high')  # Enable TF32 tensor cores for matmul

# SDPA backend configuration (optimization.md Priority 2)
torch.backends.cuda.enable_cudnn_sdp(False)    # Disable broken cuDNN backend
torch.backends.cuda.enable_flash_sdp(True)     # PyTorch's built-in flash kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Persistent compile caching (optimization.md Priority 1)
import os as _os
_os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
_os.environ.setdefault('TORCHINDUCTOR_AUTOGRAD_CACHE', '1')
_os.environ.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')  # 4 GiB PTX JIT cache

import torchaudio
import soundfile as sf


def _sf_load(path: str):
    """Load audio via soundfile (avoids torchcodec dependency in torchaudio 2.11+).
    Returns (waveform_tensor, sample_rate) matching torchaudio.load() signature.
    """
    data, sr = sf.read(path, dtype='float32')
    wav = torch.from_numpy(data).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, samples)
    elif wav.dim() == 2:
        wav = wav.T  # soundfile: (samples, ch) → (ch, samples)
    return wav, sr
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
        # FA2 doesn't support SM 120 (Blackwell consumer: RTX 5070/5080/5090)
        if major >= 8 and major < 12:
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
# Comma-separated GPU indices for realtime workers, e.g. "0,1"
RT_DEVICES = os.environ.get("MOSS_RT_DEVICES", "0").split(",")
RT_SAMPLE_RATE = 24000

# Worker pool for multi-GPU concurrent streaming
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RTWorker:
    """Holds per-GPU model state for concurrent streaming."""
    device: str              # e.g. "cuda:0"
    model: Any = None
    tokenizer: Any = None
    processor: Any = None
    inferencer: Any = None
    codec: Any = None
    sample_rate: int = 24000
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    prompt_cache: dict = field(default_factory=dict)

rt_workers: list[RTWorker] = []

# Backward compat: these point to the first worker (for non-streaming endpoints)
# They get set after loading.
rt_model = None
rt_tokenizer = None
rt_processor = None
rt_inferencer = None
rt_codec = None
rt_sample_rate = RT_SAMPLE_RATE


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


def _load_rt_worker(device: str) -> RTWorker:
    """Load MOSS-TTS-Realtime model + codec on a specific GPU device."""
    import sys
    rt_pkg_dir = "/app/MOSS-TTS/moss_tts_realtime"
    if rt_pkg_dir not in sys.path:
        sys.path.insert(0, rt_pkg_dir)

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (
        MossTTSRealtimeInference as StreamingInferencer,
    )
    from transformers import AutoTokenizer, AutoModel

    worker = RTWorker(device=device, sample_rate=RT_SAMPLE_RATE)

    logger.info(f"[{device}] Loading MOSS-TTS-Realtime tokenizer from {RT_MODEL_ID} ...")
    worker.tokenizer = AutoTokenizer.from_pretrained(RT_MODEL_ID)
    worker.processor = MossTTSRealtimeProcessor(tokenizer=worker.tokenizer)

    rt_attn = "sdpa"
    logger.info(f"[{device}] Loading MOSS-TTS-Realtime model from {RT_MODEL_ID} (attn={rt_attn}) ...")
    worker.model = MossTTSRealtime.from_pretrained(
        RT_MODEL_ID,
        attn_implementation=rt_attn,
        torch_dtype=DTYPE,
    ).to(device)
    worker.model.eval()

    logger.info(f"[{device}] Loading audio codec from {RT_CODEC_ID} ...")
    worker.codec = AutoModel.from_pretrained(RT_CODEC_ID, trust_remote_code=True).eval()
    worker.codec = worker.codec.to(device)

    worker.inferencer = StreamingInferencer(
        worker.model, worker.tokenizer, max_length=5000,
    )

    logger.info(f"[{device}] MOSS-TTS-Realtime loaded successfully")
    _log_gpu_memory()
    return worker


def load_realtime_models():
    """Load MOSS-TTS-Realtime on all configured GPUs."""
    global rt_workers, rt_model, rt_tokenizer, rt_processor, rt_inferencer, rt_codec, rt_sample_rate

    devices = [f"cuda:{idx.strip()}" for idx in RT_DEVICES if idx.strip()]
    if not devices:
        devices = ["cuda:0"]

    logger.info(f"Loading MOSS-TTS-Realtime on {len(devices)} GPU(s): {devices}")

    for dev in devices:
        try:
            worker = _load_rt_worker(dev)
            rt_workers.append(worker)
            logger.info(f"[{dev}] Worker ready")
        except Exception as e:
            logger.error(f"[{dev}] Failed to load worker: {e}", exc_info=True)

    if not rt_workers:
        raise RuntimeError("No RT workers loaded successfully")

    # Backward compat: point globals to first worker
    w0 = rt_workers[0]
    rt_model = w0.model
    rt_tokenizer = w0.tokenizer
    rt_processor = w0.processor
    rt_inferencer = w0.inferencer
    rt_codec = w0.codec
    rt_sample_rate = w0.sample_rate

    logger.info(f"{len(rt_workers)} RT worker(s) ready for concurrent streaming")


def _warmup_worker(worker: RTWorker):
    """Run model inference to trigger torch.compile + PTX JIT on local_head.
    
    We intentionally skip codec decoding — the CUDA device-side assert
    (indexSelectSmallIndex OOB) happens in codec.decode(), not in the model.
    torch.compile only wraps model.local_head, so we just need to run
    push_text_tokens/end_text/drain to trigger compilation.
    """
    import sys
    rt_pkg_dir = "/app/MOSS-TTS/moss_tts_realtime"
    if rt_pkg_dir not in sys.path:
        sys.path.insert(0, rt_pkg_dir)

    from mossttsrealtime.streaming_mossttsrealtime import (
        MossTTSRealtimeStreamingSession,
    )

    device = torch.device(worker.device)

    # Generate a dummy voice prompt (1s sine wave)
    with torch.inference_mode():
        t = torch.linspace(0, 1.0, worker.sample_rate, device=device)
        dummy_wav = (0.3 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0).unsqueeze(0)
        encode_result = worker.codec.encode(dummy_wav, chunk_duration=0.24)
        prompt_tokens = encode_result["audio_codes"].cpu().numpy().squeeze(1)

    session = MossTTSRealtimeStreamingSession(
        worker.inferencer, worker.processor,
        codec=worker.codec, codec_sample_rate=worker.sample_rate,
        codec_encode_kwargs={"chunk_duration": 0.24},
        prefill_text_len=worker.processor.delay_tokens_len,
        temperature=0.8, top_p=0.6, top_k=30,
        do_sample=True, repetition_penalty=1.1, repetition_window=50,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    system_prompt = worker.processor.make_ensemble(prompt_tokens)
    # Build input_ids with user text in proper chat template format
    # (matches official app._build_text_only_turn_input pattern)
    warmup_text = "Hello."
    user_prompt_text = (
        "<|im_end|>\n<|im_start|>user\n"
        + warmup_text
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    user_prompt_ids = worker.tokenizer.encode(user_prompt_text, add_special_tokens=False)
    user_prompt = np.full(
        (len(user_prompt_ids), system_prompt.shape[1]),
        fill_value=worker.processor.audio_channel_pad, dtype=np.int64,
    )
    user_prompt[:, 0] = np.array(user_prompt_ids, dtype=np.int64)
    input_ids = np.concatenate([system_prompt, user_prompt], axis=0)

    worker.inferencer.reset_generation_state(keep_cache=False)
    session.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)

    with torch.inference_mode():
        # Push warmup text — this triggers torch.compile on local_head
        warmup_tokens = worker.tokenizer.encode("Hello.", add_special_tokens=False)
        for tok in warmup_tokens:
            frames = session.push_text_tokens([tok])
            # Just discard frames — don't decode (codec decode causes CUDA assert)
            del frames

        # End text
        frames = session.end_text()
        del frames

        # Drain remaining
        for _ in range(100):
            frames = session.drain(max_steps=1)
            if not frames or session.inferencer.is_finished:
                break
            del frames

    # Reset inferencer state after warmup so first real request starts clean
    worker.inferencer.reset_generation_state(keep_cache=False)

    # Ensure all CUDA ops complete
    torch.cuda.synchronize()


@app.on_event("startup")
async def startup_event():
    if ENABLE_REALTIME:
        try:
            load_realtime_models()
        except Exception as e:
            logger.error(f"Failed to load MOSS-TTS-Realtime: {e}", exc_info=True)
            logger.warning("Falling back to base MOSS-TTS model")
            load_model()
            # Skip warmup if we fell back
            rt_workers.clear()

        # Warmup each worker to trigger torch.compile + PTX JIT
        for w in rt_workers:
            logger.info(f"[Warmup][{w.device}] Running dummy inference to trigger compilation...")
            t_warmup = time.perf_counter()
            try:
                _warmup_worker(w)
                logger.info(f"[Warmup][{w.device}] Done in {time.perf_counter() - t_warmup:.1f}s")
            except Exception as e:
                logger.error(f"[Warmup][{w.device}] Failed: {e}", exc_info=True)
                logger.warning(f"[Warmup][{w.device}] First real request will trigger compilation instead")
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
    """Convert audio tensor to WAV bytes via soundfile (no torchcodec dep)."""
    import io
    buf = io.BytesIO()
    # audio shape: (samples,) — soundfile expects (samples,) or (samples, channels)
    pcm = audio.cpu().numpy()
    if pcm.ndim == 0:
        return b''
    sf.write(buf, pcm, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def _generate_audio_realtime(text: str, reference_path: Optional[str] = None) -> tuple[torch.Tensor, int]:
    """Non-streaming generation using the MOSS-TTS-Realtime model.

    Used by /tts endpoint when the base model is not loaded.
    Uses the streaming session API to push text tokens, then collects
    all audio into a single tensor.
    """
    import sys
    rt_pkg_dir = "/app/MOSS-TTS/moss_tts_realtime"
    if rt_pkg_dir not in sys.path:
        sys.path.insert(0, rt_pkg_dir)

    from mossttsrealtime.streaming_mossttsrealtime import (
        MossTTSRealtimeStreamingSession,
        AudioStreamDecoder,
    )

    # Pick the first worker (globals are backward-compat aliases to worker[0])
    worker = rt_workers[0] if rt_workers else None
    if worker is None:
        raise RuntimeError("No realtime worker available")

    device = torch.device(worker.device)

    # Encode reference audio for voice cloning (cached)
    prompt_tokens = None
    if reference_path:
        cache_key = (str(reference_path), os.path.getmtime(reference_path))
        if cache_key in worker.prompt_cache:
            prompt_tokens = worker.prompt_cache[cache_key]
        else:
            with torch.inference_mode():
                wav_ref, sr_ref = _sf_load(reference_path)
                if sr_ref != worker.sample_rate:
                    wav_ref = torchaudio.functional.resample(wav_ref, sr_ref, worker.sample_rate)
                if wav_ref.shape[0] > 1:
                    wav_ref = wav_ref.mean(dim=0, keepdim=True)
                encode_result = worker.codec.encode(
                    wav_ref.unsqueeze(0).to(device),
                    chunk_duration=0.24,
                )
                prompt_tokens = encode_result["audio_codes"].cpu().numpy().squeeze(1)
            worker.prompt_cache[cache_key] = prompt_tokens

    # Create streaming session
    session = MossTTSRealtimeStreamingSession(
        worker.inferencer,
        worker.processor,
        codec=worker.codec,
        codec_sample_rate=worker.sample_rate,
        codec_encode_kwargs={"chunk_duration": 0.24},
        prefill_text_len=worker.processor.delay_tokens_len,
        temperature=0.8,
        top_p=0.6,
        top_k=30,
        do_sample=True,
        repetition_penalty=1.1,
        repetition_window=50,
    )

    if prompt_tokens is not None:
        session.set_voice_prompt_tokens(prompt_tokens)

    # Build input_ids with chat template (same as streaming path)
    system_prompt = worker.processor.make_ensemble(prompt_tokens)
    user_prompt_text = (
        "<|im_end|>\n<|im_start|>user\n"
        + text
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    user_prompt_ids = worker.tokenizer.encode(user_prompt_text, add_special_tokens=False)
    user_prompt = np.full(
        (len(user_prompt_ids), system_prompt.shape[1]),
        fill_value=worker.processor.audio_channel_pad,
        dtype=np.int64,
    )
    user_prompt[:, 0] = np.array(user_prompt_ids, dtype=np.int64)
    input_ids = np.concatenate([system_prompt, user_prompt], axis=0)

    worker.inferencer.reset_generation_state(keep_cache=False)
    session.reset_turn(
        input_ids=input_ids,
        include_system_prompt=False,
        reset_cache=True,
    )

    # Tokenize text
    text_tokens = worker.tokenizer.encode(text, add_special_tokens=False)
    if not text_tokens:
        raise RuntimeError("Text tokenized to empty — nothing to generate")

    # Audio EOS / codebook size for sanitization
    audio_eos_token = getattr(worker.processor, "audio_eos_token_id", None)
    codebook_size = getattr(worker.processor, "codebook_size", 2048)

    def _sanitize_tokens(tokens):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.numel() == 0:
            return tokens, False
        eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False) if audio_eos_token is not None else torch.tensor([])
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

    # Hard-reset codec streaming state
    for _name, module in worker.codec.named_modules():
        if hasattr(module, '_streaming_state'):
            module._streaming_state = None

    worker.codec._start_streaming(batch_size=1)
    try:
        decoder = AudioStreamDecoder(
            worker.codec,
            chunk_frames=12,
            overlap_frames=0,
            decode_kwargs={"chunk_duration": -1},
        )

        with torch.inference_mode():
            # Push text tokens in chunks
            CHUNK_SIZE = 12
            for i in range(0, len(text_tokens), CHUNK_SIZE):
                token_chunk = text_tokens[i:i + CHUNK_SIZE]
                audio_frames = session.push_text_tokens(token_chunk)
                for frame in audio_frames:
                    tokens = frame
                    if tokens.dim() == 3:
                        tokens = tokens[0]
                    tokens, _ = _sanitize_tokens(tokens)
                    if tokens.numel() > 0:
                        decoder.push_tokens(tokens.detach())

            # Signal end of text
            audio_frames = session.end_text()
            for frame in audio_frames:
                tokens = frame
                if tokens.dim() == 3:
                    tokens = tokens[0]
                tokens, _ = _sanitize_tokens(tokens)
                if tokens.numel() > 0:
                    decoder.push_tokens(tokens.detach())

            # Drain remaining audio
            for _ in range(200):
                audio_frames = session.drain(max_steps=1)
                if not audio_frames:
                    break
                for frame in audio_frames:
                    tokens = frame
                    if tokens.dim() == 3:
                        tokens = tokens[0]
                    tokens, _ = _sanitize_tokens(tokens)
                    if tokens.numel() > 0:
                        decoder.push_tokens(tokens.detach())
                if session.inferencer.is_finished:
                    break

            # Collect all audio chunks
            audio_chunks = list(decoder.audio_chunks())
            final = decoder.flush()
            if final is not None and final.numel() > 0:
                audio_chunks.append(final)

        if not audio_chunks:
            raise RuntimeError("Realtime model produced no audio")

        # Concatenate all chunks into a single tensor
        all_audio = torch.cat([c.cpu().reshape(-1) for c in audio_chunks], dim=0)

    finally:
        worker.codec._stop_streaming()
        # Reset inferencer state for next request
        worker.inferencer.reset_generation_state(keep_cache=False)

    return all_audio, worker.sample_rate


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_mem = None
    if DEVICE == "cuda":
        gpu_mem = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    is_ready = model is not None or len(rt_workers) > 0
    active_model = RT_MODEL_ID if rt_workers else MODEL_ID
    active_sr = RT_SAMPLE_RATE if rt_workers else sample_rate

    response = HealthResponse(
        status="ready" if is_ready else "loading",
        model_id=active_model,
        device=DEVICE,
        attention=ATTN_IMPL,
        sample_rate=active_sr,
        gpu_memory_gb=gpu_mem,
    )
    # Add multi-GPU worker info
    if rt_workers:
        response_dict = response.model_dump()
        response_dict["rt_workers"] = [
            {"device": w.device, "busy": w.semaphore.locked()}
            for w in rt_workers
        ]
        return response_dict
    return response


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
    if model is None and not rt_workers:
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
    if model is None and not rt_workers:
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
    if rt_workers:
        return await _stream_realtime(request, ref_path)
    elif rt_inferencer is not None:
        return await _stream_realtime(request, ref_path)
    else:
        return await _stream_sentence_fallback(request, ref_path)


async def _acquire_rt_worker() -> RTWorker:
    """Find the first available GPU worker (doesn't acquire semaphore)."""
    while True:
        for w in rt_workers:
            if not w.semaphore.locked():
                return w
        # All workers busy — wait briefly and retry
        await asyncio.sleep(0.01)


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

    # Acquire a GPU worker (blocks until one is free)
    worker = await _acquire_rt_worker()
    device = torch.device(worker.device)

    t_start = time.perf_counter()

    logger.info(
        f"[Stream-RT][{worker.device}] True realtime streaming: "
        f"voice={request.voice_name or 'default'}, "
        f"text_len={len(request.text)}"
    )

    # ── Encode reference audio for voice cloning (cached) ────────────
    prompt_tokens = None
    if ref_path:
        cache_key = (str(ref_path), os.path.getmtime(ref_path))
        if cache_key in worker.prompt_cache:
            prompt_tokens = worker.prompt_cache[cache_key]
            logger.info(f"[Stream-RT][{worker.device}] Voice prompt cache hit ({time.perf_counter()-t_start:.3f}s)")
        else:
            with torch.inference_mode():
                wav, sr_wav = _sf_load(ref_path)
                if sr_wav != worker.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr_wav, worker.sample_rate)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                encode_result = worker.codec.encode(
                    wav.unsqueeze(0).to(device),
                    chunk_duration=0.24,
                )
                prompt_tokens = encode_result["audio_codes"].cpu().numpy().squeeze(1)
            worker.prompt_cache[cache_key] = prompt_tokens
            logger.info(f"[Stream-RT][{worker.device}] Voice prompt encoded ({time.perf_counter()-t_start:.3f}s)")

    async def realtime_streamer():
        t0 = time.perf_counter()
        chunk_idx = 0
        total_audio_duration = 0.0

        try:
            # ── Create streaming session ─────────────────────────
            session = MossTTSRealtimeStreamingSession(
                worker.inferencer,
                worker.processor,
                codec=worker.codec,
                codec_sample_rate=worker.sample_rate,
                codec_encode_kwargs={"chunk_duration": 0.24},
                prefill_text_len=worker.processor.delay_tokens_len,
                temperature=request.temperature or 0.8,
                top_p=request.top_p or 0.6,
                top_k=30,
                do_sample=True,
                repetition_penalty=1.1,
                repetition_window=50,
            )

            if prompt_tokens is not None:
                session.set_voice_prompt_tokens(prompt_tokens)

            # ── Build input_ids with user text (official chat template) ─
            # The model needs the text-to-speak as the user message in chat
            # template format: system → user(text) → assistant.
            # push_text_tokens() then feeds the same text as the assistant
            # response tokens that the model aligns audio to.
            system_prompt = worker.processor.make_ensemble(prompt_tokens)
            user_prompt_text = (
                "<|im_end|>\n<|im_start|>user\n"
                + request.text
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            user_prompt_ids = worker.tokenizer.encode(user_prompt_text, add_special_tokens=False)
            user_prompt = np.full(
                (len(user_prompt_ids), system_prompt.shape[1]),
                fill_value=worker.processor.audio_channel_pad,
                dtype=np.int64,
            )
            user_prompt[:, 0] = np.array(user_prompt_ids, dtype=np.int64)
            input_ids = np.concatenate([system_prompt, user_prompt], axis=0)

            worker.inferencer.reset_generation_state(keep_cache=False)
            session.reset_turn(
                input_ids=input_ids,
                include_system_prompt=False,
                reset_cache=True,
            )
            logger.info(f"[Stream-RT][{worker.device}] Session setup + prefill: {time.perf_counter()-t0:.3f}s")

            # ── Audio stream decoder ─────────────────────────────
            decoder = AudioStreamDecoder(
                worker.codec,
                chunk_frames=12,
                overlap_frames=0,
                decode_kwargs={"chunk_duration": -1},
                device=device,
            )

            codebook_size = int(getattr(worker.codec, "codebook_size", 1024))
            audio_eos_token = int(getattr(worker.inferencer, "audio_eos_token", 1026))

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
            # Minimal batching for low TTFA — send chunks as soon as available
            MIN_SAMPLES = int(worker.sample_rate * 0.05)  # ~50ms minimum chunk

            def _run_stream():
                """Synchronous generator: push text tokens → audio chunks.
                Uses push_text_tokens() with adaptive chunk sizes for
                optimal throughput while maintaining fast TTFA.
                """
                INITIAL_CHUNK = 1   # 1 token at a time initially for fast TTFA
                STEADY_CHUNK = 12   # Switch to 12-token chunks once audio flows

                # Hard-reset ALL codec streaming state
                for _name, module in worker.codec.named_modules():
                    if hasattr(module, '_streaming_state'):
                        module._streaming_state = None

                # Timing instrumentation
                t_infer = 0.0
                t_decode = 0.0
                n_infer = 0
                n_decode = 0

                # Use manual _start/_stop instead of codec.streaming() context
                # to avoid FX tracing conflicts and codec recompilation overhead
                worker.codec._start_streaming(batch_size=1)
                try:
                    with torch.inference_mode():
                        buf = []
                        buf_samples = 0
                        first_audio_sent = False

                        def _flush_buf():
                            nonlocal buf, buf_samples, first_audio_sent
                            if buf:
                                merged = np.concatenate(buf)
                                buf = []
                                buf_samples = 0
                                first_audio_sent = True
                                return merged
                            return None

                        # Pre-tokenize the full text
                        text_tokens = worker.tokenizer.encode(
                            request.text, add_special_tokens=False,
                        )
                        if not text_tokens:
                            return

                        # Push tokens — small chunks first for TTFA, then big for throughput
                        i = 0
                        while i < len(text_tokens):
                            chunk_size = STEADY_CHUNK if first_audio_sent else INITIAL_CHUNK
                            token_chunk = text_tokens[i:i + chunk_size]
                            i += len(token_chunk)
                            _t0 = time.perf_counter()
                            audio_frames = session.push_text_tokens(token_chunk)
                            t_infer += time.perf_counter() - _t0
                            n_infer += 1
                            _t0 = time.perf_counter()
                            for chunk in decode_frames(audio_frames):
                                t_decode += time.perf_counter() - _t0
                                n_decode += 1
                                buf.append(chunk)
                                buf_samples += len(chunk)
                                if buf_samples >= MIN_SAMPLES:
                                    yield _flush_buf()
                                _t0 = time.perf_counter()
                            t_decode += time.perf_counter() - _t0

                        _t0 = time.perf_counter()
                        audio_frames = session.end_text()
                        t_infer += time.perf_counter() - _t0
                        _t0 = time.perf_counter()
                        for chunk in decode_frames(audio_frames):
                            t_decode += time.perf_counter() - _t0
                            n_decode += 1
                            buf.append(chunk)
                            buf_samples += len(chunk)
                            if buf_samples >= MIN_SAMPLES:
                                yield _flush_buf()
                            _t0 = time.perf_counter()
                        t_decode += time.perf_counter() - _t0

                        while True:
                            _t0 = time.perf_counter()
                            audio_frames = session.drain(max_steps=1)
                            t_infer += time.perf_counter() - _t0
                            n_infer += 1
                            if not audio_frames:
                                break
                            _t0 = time.perf_counter()
                            for chunk in decode_frames(audio_frames):
                                t_decode += time.perf_counter() - _t0
                                n_decode += 1
                                buf.append(chunk)
                                buf_samples += len(chunk)
                                if buf_samples >= MIN_SAMPLES:
                                    yield _flush_buf()
                                _t0 = time.perf_counter()
                            t_decode += time.perf_counter() - _t0
                            if session.inferencer.is_finished:
                                break

                        for chunk in flush_dec():
                            buf.append(chunk)
                            buf_samples += len(chunk)

                        merged = _flush_buf()
                        if merged is not None:
                            yield merged

                        logger.info(
                            f"[Stream-RT][Timing] infer={t_infer:.2f}s ({n_infer} calls), "
                            f"decode={t_decode:.2f}s ({n_decode} chunks)"
                        )
                finally:
                    worker.codec._stop_streaming()

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
                    import traceback
                    logger.error(f"[Stream-RT] Producer error: {type(e).__name__}: {e}")
                    logger.error(f"[Stream-RT] Traceback: {traceback.format_exc()}")
                    error_holder[0] = e
                finally:
                    q.put(None)  # sentinel

            async with worker.semaphore:
                thread = threading.Thread(target=_producer, daemon=True)
                thread.start()

                while True:
                    # 300s timeout to accommodate torch.compile first-run (~80-120s)
                    wav_np = await asyncio.to_thread(q.get, True, 300.0)
                    if wav_np is None:
                        break

                    # In-memory WAV — no disk I/O
                    wav_bytes = _wav_bytes_from_pcm(wav_np, worker.sample_rate)

                    chunk_duration = len(wav_np) / worker.sample_rate
                    total_audio_duration += chunk_duration

                    meta = {
                        "chunk_index": chunk_idx,
                        "is_final": False,
                        "sample_rate": worker.sample_rate,
                        "audio_duration": round(chunk_duration, 3),
                        "total_audio_duration": round(total_audio_duration, 2),
                        "generation_time": round(time.perf_counter() - t0, 2),
                        "streaming": "realtime",
                        "gpu": worker.device,
                    }
                    meta_bytes = _json.dumps(meta).encode()
                    header = struct.pack("<II", len(wav_bytes), len(meta_bytes))
                    yield header + wav_bytes + meta_bytes

                    chunk_idx += 1

                # Wait for producer thread to fully exit
                thread.join()

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

