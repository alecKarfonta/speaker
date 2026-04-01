import asyncio
import os
import asyncio

# ── Force torchaudio soundfile backend ──────────────────────────────────────
# torchaudio >= 2.9.1 defaults to torchcodec which is not installed / broken.
# This must happen BEFORE any module imports torchaudio (e.g. GLM-TTS frontend).
os.environ.setdefault("TORCHAUDIO_BACKEND", "soundfile")
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass  # Older torchaudio versions may not have set_audio_backend
# ─────────────────────────────────────────────────────────────────────────────

import logging
import numpy as np
import io
import soundfile as sf
import time
from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel
from pydantic import BaseModel, Field, constr
from fastapi import UploadFile, File, Form, HTTPException, Depends, Request
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from prometheus_client import generate_latest, REGISTRY, CONTENT_TYPE_LATEST

from app.tts_factory import TTSBackendFactory, register_default_backends
from app.tts_backend_base import TTSBackendBase
from app.config import settings, get_backend_config
from app.log_util import ColoredFormatter
from app.monitoring import metrics_collector, health_monitor, audit_logger, get_metrics, get_health
from app.models import (
    TTSRequest, VoiceUploadResponse, VoiceDeleteResponse, LanguageListResponse,
    HealthResponse, APIInfo, MetricsResponse, ErrorResponse, ErrorDetail,
    OpenAISpeechRequest, OpenAIAudioFormat
)
import app.version as version
from app.audiobook_router import router as audiobook_router, set_tts_service as set_audiobook_tts
from app.audiobook_ws import router as audiobook_ws_router, set_tts_service as set_audiobook_ws_tts

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter and handler
formatter = ColoredFormatter()
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add handler to logger
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
logger.addHandler(handler)

# Register available TTS backends
register_default_backends()

# TTS Service - Create backend based on configuration
# Can be set via TTS_BACKEND environment variable or settings.tts_backend
backend_name = os.environ.get("TTS_BACKEND", settings.tts_backend)
backend_config = get_backend_config()
logger.info(f"Initializing TTS backend: {backend_name}")
tts_service: TTSBackendBase = TTSBackendFactory.create_backend(
    backend_name,
    logger=logger,
    config=backend_config
)

# Load available voices into the TTS service
try:
    tts_service.load_voices()
except Exception as e:
    logger.error(f"Failed to load voices: {e}")

# Inject TTS service into audiobook router
set_audiobook_tts(tts_service)
set_audiobook_ws_tts(tts_service)

# Semaphore to serialize GPU inference — only 1 inference thread at a time.
# Additional requests wait on the semaphore (non-blocking to event loop)
# rather than spawning competing threads that fight over VLLM/GPU resources.
tts_inference_semaphore = asyncio.Semaphore(1)

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, List[float]] = {}

def check_rate_limit(request: Request):
    """Simple rate limiting: 100 requests per minute per IP"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    
    # Remove requests older than 1 minute
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip] 
        if current_time - req_time < 60
    ]
    
    if len(rate_limit_storage[client_ip]) >= 100:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 100 requests per minute."
        )
    
    rate_limit_storage[client_ip].append(current_time)

app = FastAPI(
    title="Speaker TTS API", 
    description="""
    High-quality Text-to-Speech API with XTTS v2 voice cloning capabilities.
    
    ## Features
    * **Voice Cloning**: Upload audio samples to create custom voices
    * **Multi-language Support**: Generate speech in multiple languages
    * **Emotion Control**: Add emotion tags to control speech style
    * **Real-time Generation**: Fast audio generation with configurable parameters
    
    ## Authentication
    Currently, this API is open access. Rate limiting is applied to prevent abuse.
    
    ## Rate Limits
    * 100 requests per minute per IP address
    """,
    version="1.0.0",
    contact={
        "name": "Speaker TTS API Support",
        "url": "https://github.com/your-repo/speaker",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    debug=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register audiobook routers
app.include_router(audiobook_router)
app.include_router(audiobook_ws_router)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log error for audit
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent")
    audit_logger.log_error(
        f"HTTP_{exc.status_code}", 
        exc.detail, 
        client_ip, 
        user_agent
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Log error for audit
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent")
    audit_logger.log_error("UNHANDLED_EXCEPTION", str(exc), client_ip, user_agent)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            }
        }
    )

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# API versioning middleware
@app.middleware("http")
async def add_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = "1.0.0"
    return response

# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Extract metrics from request
    voice_name = None
    language = None
    text_length = None
    text = None
    
    # Try to extract TTS-specific metrics
    if request.url.path == "/tts" and request.method == "POST":
        try:
            # Get the request body for TTS requests
            body = await request.body()
            # Note: This is a simplified approach. In production, you might want to
            # parse the JSON body more carefully
            if b"voice_name" in body:
                # Extract basic metrics from request
                voice_name = "extracted_from_request"  # Simplified
                language = "extracted_from_request"    # Simplified
                text_length = len(body)  # Simplified
                text = "extracted_from_request"  # Simplified
        except:
            pass
    
    # Record metrics with enhanced tracking
    metrics_collector.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        response_time=response_time,
        voice_name=voice_name,
        language=language,
        text_length=text_length,
        text=text
    )
    
    # Update system metrics for Prometheus
    metrics_collector.update_system_metrics()
    
    return response

@app.get("/", response_model=APIInfo)
async def read_root():
    """
    Root endpoint providing API information and status.
    """
    return APIInfo(
        message="Welcome to the Speaker TTS API",
        version="1.0.0",
        model=tts_service.model_name,
        status="operational",
        documentation="/docs",
        health_check="/health"
    )

@app.get("/voices", response_model=Dict[str, List[str]])
async def get_voices():
    """
    Get list of available voices for cloning.
    
    Returns a dictionary with available voice names that can be used for TTS generation.
    """
    return {"voices": tts_service.get_voices()}

@app.get("/voices/{voice_name}/details")
async def get_voice_details(voice_name: str):
    """
    Get detailed information about a specific voice including all its files.
    
    - **voice_name**: Name of the voice
    
    Returns detailed file information for the voice.
    """
    voice_dir = f"data/voices/{voice_name}"
    
    if not os.path.exists(voice_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{voice_name}' not found"
        )
    
    files = []
    total_size = 0
    
    for filename in os.listdir(voice_dir):
        if filename.lower().endswith(('.wav', '.mp3')):
            filepath = os.path.join(voice_dir, filename)
            file_stat = os.stat(filepath)
            
            # Get audio duration
            duration = None
            try:
                audio_data, sample_rate = sf.read(filepath)
                duration = len(audio_data) / sample_rate
            except:
                pass
            
            files.append({
                "name": filename,
                "path": filepath,
                "size": file_stat.st_size,
                "duration": duration,
                "modified": file_stat.st_mtime,
            })
            total_size += file_stat.st_size
    
    return {
        "voice_name": voice_name,
        "files": files,
        "total_files": len(files),
        "total_size": total_size,
    }

@app.get("/voices/{voice_name}/files/{filename}")
async def download_voice_file(voice_name: str, filename: str):
    """
    Download a specific audio file from a voice.
    
    - **voice_name**: Name of the voice
    - **filename**: Name of the file to download
    """
    voice_dir = f"data/voices/{voice_name}"
    filepath = os.path.join(voice_dir, filename)
    
    # Security check: ensure the file is within the voice directory
    if not os.path.abspath(filepath).startswith(os.path.abspath(voice_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in voice '{voice_name}'"
        )
    
    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=filename
    )

@app.delete("/voices/{voice_name}/files/{filename}")
async def delete_voice_file(voice_name: str, filename: str, request: Request):
    """
    Delete a specific audio file from a voice.
    
    - **voice_name**: Name of the voice
    - **filename**: Name of the file to delete
    """
    check_rate_limit(request)
    
    voice_dir = f"data/voices/{voice_name}"
    filepath = os.path.join(voice_dir, filename)
    
    # Security check: ensure the file is within the voice directory
    if not os.path.abspath(filepath).startswith(os.path.abspath(voice_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in voice '{voice_name}'"
        )
    
    try:
        os.remove(filepath)
        
        # Check if voice directory is empty and remove it if so
        remaining_files = [f for f in os.listdir(voice_dir) if f.lower().endswith(('.wav', '.mp3'))]
        if len(remaining_files) == 0:
            import shutil
            shutil.rmtree(voice_dir)
        
        # Reload voices
        tts_service.load_voices()
        
        return {"message": f"File '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )

@app.post("/voices/combine")
async def combine_voice_files(
    voice_name: str,
    source_voices: List[str],
    request: Request
):
    """
    Combine audio files from multiple voices into a new voice.
    
    - **voice_name**: Name for the new combined voice
    - **source_voices**: List of voice names to combine
    """
    check_rate_limit(request)
    
    if len(source_voices) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 source voices are required to combine"
        )
    
    # Validate voice name
    if not voice_name.replace('_', '').isalnum():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice name must contain only letters, numbers, and underscores"
        )
    
    # Check if target voice already exists
    target_dir = f"data/voices/{voice_name}"
    if os.path.exists(target_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice '{voice_name}' already exists"
        )
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy all files from source voices
        file_index = 1
        copied_files = []
        
        for source_voice in source_voices:
            source_dir = f"data/voices/{source_voice}"
            
            if not os.path.exists(source_dir):
                logger.warning(f"Source voice '{source_voice}' not found, skipping")
                continue
            
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(('.wav', '.mp3')):
                    source_path = os.path.join(source_dir, filename)
                    file_extension = os.path.splitext(filename)[1]
                    target_filename = f"{voice_name}_{str(file_index).zfill(2)}{file_extension}"
                    target_path = os.path.join(target_dir, target_filename)
                    
                    import shutil
                    shutil.copy2(source_path, target_path)
                    copied_files.append(target_filename)
                    file_index += 1
        
        if len(copied_files) == 0:
            # Clean up empty directory
            os.rmdir(target_dir)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid audio files found in source voices"
            )
        
        # Reload voices
        tts_service.load_voices()
        
        # Log for audit
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent")
        audit_logger.log_voice_upload(
            voice_name, 
            f"combined from {', '.join(source_voices)}", 
            0, 
            client_ip, 
            user_agent
        )
        
        return {
            "message": f"Successfully combined {len(source_voices)} voices into '{voice_name}'",
            "voice_name": voice_name,
            "files_copied": len(copied_files),
            "source_voices": source_voices
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error combining voices: {str(e)}", exc_info=True)
        # Clean up on error
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to combine voices: {str(e)}"
        )

@app.post("/voices", status_code=status.HTTP_201_CREATED, response_model=VoiceUploadResponse)
async def add_voice(
    voice_name: str, 
    request: Request,
    file: UploadFile = File(...),
    transcription: Optional[str] = Form(None)
):
    """
    Add a new voice by uploading an audio file.
    
    - **voice_name**: Name for the voice (alphanumeric and underscores only)
    - **file**: Audio file (.wav or .mp3) containing voice sample
    - **transcription**: Optional transcription text of the audio. If provided, this text
      will be used as the voice prompt instead of running STT on the audio.
    
    The voice will be available for TTS generation after upload.
    """
    # Check rate limit
    check_rate_limit(request)
    
    # Validate voice name (alphanumeric and underscores only)
    if not voice_name.replace('_', '').isalnum():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice name must contain only letters, numbers, and underscores"
        )
    
    # Validate file type
    if not file.filename or not (file.filename.lower().endswith('.wav') or file.filename.lower().endswith('.mp3')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav and .mp3 files are supported"
        )

    voice_dir = f"data/voices/{voice_name}"
    
    # Create voice directory if it doesn't exist
    os.makedirs(voice_dir, exist_ok=True)
    
    # Find next available index
    voice_index = 1
    file_extension = os.path.splitext(file.filename)[1]
    voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"
    while os.path.exists(os.path.join(voice_dir, voice_file_name)):
        voice_index += 1
        voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"

    # Save file to disk
    local_save_path = os.path.join(voice_dir, voice_file_name)
    logger.info(f"Saving voice file to {local_save_path}")
    
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
            
        with open(local_save_path, "wb") as f:
            f.write(content)
        
        # Save transcription text file if provided
        if transcription and transcription.strip():
            transcription_path = os.path.splitext(local_save_path)[0] + ".txt"
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(transcription.strip())
            logger.info(f"Saved transcription to {transcription_path}")
            
        # Reload voices to include the new one
        tts_service.load_voices()
        
        # Log for audit
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent")
        audit_logger.log_voice_upload(voice_name, voice_file_name, len(content), client_ip, user_agent)
        
        return VoiceUploadResponse(
            message=f"Voice '{voice_name}' uploaded successfully",
            voice_name=voice_name,
            file_name=voice_file_name,
            file_size=len(content)
        )
        
    except Exception as e:
        logger.error(f"Error saving voice file: {str(e)}", exc_info=True)
        # Clean up partial file if it exists
        if os.path.exists(local_save_path):
            os.remove(local_save_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save voice file: {str(e)}"
        )

# TTS Generation
@app.post("/tts", response_class=Response)
async def generate_speech(
    request: TTSRequest,
    http_request: Request
):
    """
    Generate speech from text using the specified voice.
    
    - **text**: Text to convert to speech (1-2000 characters)
    - **voice_name**: Name of the voice to use
    - **language**: Two-letter language code (e.g., "en", "es", "fr")
    - **temperature**: Controls randomness (0.1-1.0)
    - **top_p**: Top-p sampling parameter (0.1-1.0)
    - **emotion**: Optional emotion tag to control speech style
    - **speed**: Speech speed multiplier (0.5-2.0)
    
    Returns audio data in WAV format.
    """
    # Check rate limit
    check_rate_limit(http_request)
    
    try:
        # Validate voice exists
        available_voices = tts_service.get_voices()
        if request.voice_name not in available_voices:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice '{request.voice_name}' not found. Available voices: {', '.join(available_voices)}"
            )

        # Prepare text with emotion tags if provided
        text = request.text
        if request.emotion:
            # Add emotion tag to the beginning of the text
            emotion_tag = request.emotion if request.emotion.startswith('(') else f"({request.emotion})"
            text = f"{emotion_tag} {text}"

        logger.debug(
            f"Generating speech: text='{text[:50]}...', "
            f"voice={request.voice_name}, language={request.language}"
        )

        # Build kwargs for optional GLM-TTS parameters
        glm_kwargs = {}
        if request.sampling is not None:
            glm_kwargs["sampling"] = request.sampling
        if request.min_token_text_ratio is not None:
            glm_kwargs["min_token_text_ratio"] = request.min_token_text_ratio
        if request.max_token_text_ratio is not None:
            glm_kwargs["max_token_text_ratio"] = request.max_token_text_ratio
        if request.beam_size is not None:
            glm_kwargs["beam_size"] = request.beam_size
        if request.temperature is not None:
            glm_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            glm_kwargs["top_p"] = request.top_p
        if request.repetition_penalty is not None:
            glm_kwargs["repetition_penalty"] = request.repetition_penalty
        if request.sample_method is not None:
            glm_kwargs["sample_method"] = request.sample_method
        
        async with tts_inference_semaphore:
            audio, sample_rate = await asyncio.to_thread(
                tts_service.generate_speech,
                text=text,
                voice_name=request.voice_name,
                language=request.language,
                **glm_kwargs
            )

        # Debug logging
        # logger.debug(f"generate_speech(): {type(audio) = }")    
        # if len(audio) > 100:
        #     random_start_index = random.randint(0, len(audio) - 100)
        #     logger.debug(f"generate_speech(): {audio[random_start_index:random_start_index + 100] = }")

        if audio is None or len(audio) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate audio"
            )
        
        # Encode audio based on requested format
        output_format = request.output_format.value if request.output_format else "raw"
        
        if output_format == "wav":
            # Encode as proper WAV file
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            media_type = "audio/wav"
        else:
            # Default: raw float32 PCM bytes (backward compatible)
            audio_bytes = np.array(audio, dtype=np.float32).tobytes()
            media_type = "audio/pcm"

        # logger.debug(f"Successfully generated audio ({output_format}): {len(audio_bytes)} bytes")
        
        # Log for audit
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent")
        audit_logger.log_tts_generation(
            request.voice_name, 
            len(request.text), 
            request.language, 
            client_ip, 
            user_agent
        )
        
        # Record detailed metrics with actual text content for word-based tracking
        metrics_collector.record_request(
            endpoint="/tts",
            method="POST",
            status_code=200,
            response_time=0,  # Will be set by middleware
            voice_name=request.voice_name,
            language=request.language,
            text_length=len(request.text),
            text=request.text  # Pass actual text for word counting
        )
        
        filename = "speech.wav" if output_format == "wav" else "speech.pcm"
        
        # Build response headers with profiling data
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(audio_bytes)),
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Duration": str(len(audio)/sample_rate),
            "X-Audio-Format": output_format,
            "X-Model": tts_service.model_name,
            "X-Voice": request.voice_name,
            "X-Language": request.language,
            "X-Text-Length": str(len(request.text))
        }
        
        # Add profiling headers if available (GLM-TTS backend)
        if hasattr(tts_service, 'get_last_timings'):
            timings = tts_service.get_last_timings()
            if timings:
                headers["X-Timing-Total-Ms"] = str(round(timings.get('total', 0) * 1000, 1))
                headers["X-Timing-LLM-Ms"] = str(round(timings.get('llm_inference', 0) * 1000, 1))
                headers["X-Timing-Flow-Ms"] = str(round(timings.get('flow_inference', 0) * 1000, 1))
                headers["X-Timing-Normalize-Ms"] = str(round(timings.get('text_normalize', 0) * 1000, 1))
                headers["X-Timing-Prompt-Ms"] = str(round(timings.get('prompt_extraction', 0) * 1000, 1))
                headers["X-Timing-Validation-Ms"] = str(round(timings.get('validation', 0) * 1000, 1))
                headers["X-Tokens-Generated"] = str(timings.get('tokens_generated', 0))
                headers["X-RTF"] = str(round(timings.get('rtf', 0), 2))
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

from fastapi.responses import StreamingResponse
import struct
import json

@app.post("/tts/stream")
async def generate_speech_stream(
    request: TTSRequest,
    http_request: Request
):
    """
    Stream TTS audio chunks as they're generated.
    
    This endpoint streams audio chunks in real-time, reducing time-to-first-audio
    by up to 1500ms compared to the regular /tts endpoint.
    
    - **text**: Text to convert to speech (1-2000 characters)
    - **voice_name**: Name of the voice to use
    - **language**: Two-letter language code
    
    Returns a chunked binary stream with framing:
    - Each chunk: 4-byte audio_len + 4-byte metadata_len + audio_bytes + metadata_json
    
    Response headers include:
    - X-Streaming: true
    - X-Sample-Rate: 24000
    """
    check_rate_limit(http_request)
    
    # Validate voice exists
    available_voices = tts_service.get_voices()
    if request.voice_name not in available_voices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{request.voice_name}' not found. Available voices: {', '.join(available_voices)}"
        )
    
    # Check if backend supports streaming
    if not hasattr(tts_service, 'generate_speech_streaming'):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="This TTS backend does not support streaming. Use /tts instead."
        )
    
    # Prepare text with emotion tags if provided
    text = request.text
    if request.emotion:
        emotion_tag = request.emotion if request.emotion.startswith('(') else f"({request.emotion})"
        text = f"{emotion_tag} {text}"
    
    logger.info(f"[STREAMING] Starting stream for: '{text[:50]}...' voice={request.voice_name}")
    
    # Build kwargs for optional GLM-TTS parameters
    glm_kwargs = {}
    if request.sampling is not None:
        glm_kwargs["sampling"] = request.sampling
    if request.min_token_text_ratio is not None:
        glm_kwargs["min_token_text_ratio"] = request.min_token_text_ratio
    if request.max_token_text_ratio is not None:
        glm_kwargs["max_token_text_ratio"] = request.max_token_text_ratio
    if request.beam_size is not None:
        glm_kwargs["beam_size"] = request.beam_size
    if request.temperature is not None:
        glm_kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        glm_kwargs["top_p"] = request.top_p
    if request.repetition_penalty is not None:
        glm_kwargs["repetition_penalty"] = request.repetition_penalty
    if request.sample_method is not None:
        glm_kwargs["sample_method"] = request.sample_method
    
    async def audio_generator():
        """Async generator that yields framed audio chunks.
        
        Runs the blocking TTS inference in a thread pool with a queue
        so chunks are yielded as they're generated (true streaming).
        """
        import queue
        import threading
        
        try:
            chunk_queue = queue.Queue()
            error_holder = [None]
            
            def _producer():
                try:
                    for chunk_audio, sample_rate, metadata in tts_service.generate_speech_streaming(
                        text=text,
                        voice_name=request.voice_name,
                        language=request.language.value if hasattr(request.language, 'value') else request.language,
                        **glm_kwargs
                    ):
                        chunk_queue.put((chunk_audio, sample_rate, metadata))
                except Exception as e:
                    error_holder[0] = e
                finally:
                    chunk_queue.put(None)  # sentinel
            
            async with tts_inference_semaphore:
                thread = threading.Thread(target=_producer, daemon=True)
                thread.start()
                
                while True:
                    item = await asyncio.to_thread(chunk_queue.get, True, 300.0)
                    if item is None:
                        break
                    
                    chunk_audio, sample_rate, metadata = item
                    
                    # Encode audio as WAV or raw float32
                    if request.output_format and request.output_format.value == "wav":
                        wav_buffer = io.BytesIO()
                        sf.write(wav_buffer, chunk_audio, sample_rate, format='WAV', subtype='PCM_16')
                        wav_buffer.seek(0)
                        audio_bytes = wav_buffer.read()
                    else:
                        audio_bytes = chunk_audio.astype(np.float32).tobytes()
                    
                    # Encode metadata as JSON
                    metadata_bytes = json.dumps(metadata).encode('utf-8')
                    
                    # Frame: 4-byte audio_len (little-endian) + 4-byte metadata_len + audio + metadata
                    header = struct.pack('<II', len(audio_bytes), len(metadata_bytes))
                    
                    yield header + audio_bytes + metadata_bytes
                    
                    logger.debug(
                        f"[STREAMING] Yielded chunk {metadata.get('chunk_index', 0)+1}: "
                        f"{len(audio_bytes)} bytes"
                    )
                
                thread.join()
            
            if error_holder[0]:
                raise error_holder[0]
                
        except Exception as e:
            logger.error(f"[STREAMING] Error during generation: {str(e)}", exc_info=True)
            # Yield error as final chunk
            error_metadata = {
                'error': str(e),
                'is_final': True
            }
            error_bytes = json.dumps(error_metadata).encode('utf-8')
            header = struct.pack('<II', 0, len(error_bytes))
            yield header + error_bytes
    
    # Log for audit
    client_ip = http_request.client.host if http_request.client else "unknown"
    user_agent = http_request.headers.get("user-agent")
    audit_logger.log_tts_generation(
        request.voice_name,
        len(request.text),
        request.language.value if hasattr(request.language, 'value') else request.language,
        client_ip,
        user_agent
    )
    
    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Streaming": "true",
            "X-Sample-Rate": str(tts_service.sample_rate),
            "X-Model": tts_service.model_name,
            "X-Voice": request.voice_name,
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked"
        }
    )


# OpenAI-compatible voice mapping
# Maps OpenAI standard voices to Speaker voices (fallback to first available)
OPENAI_VOICE_MAPPING = {
    "alloy": None,    # Will map to first available voice
    "echo": None,
    "fable": None,
    "onyx": None,
    "nova": None,
    "shimmer": None,
    "ash": None,
    "ballad": None,
    "coral": None,
    "sage": None,
}

def convert_audio_format(audio_data: np.ndarray, sample_rate: int, target_format: str) -> tuple[bytes, str]:
    """
    Convert audio to the specified format.
    
    Returns (audio_bytes, content_type)
    """
    import subprocess
    import tempfile
    
    # Format to content-type mapping
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    
    content_type = content_types.get(target_format, "audio/mpeg")
    
    # Native formats - no ffmpeg needed
    if target_format == "wav":
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return wav_buffer.read(), content_type
    
    if target_format == "flac":
        flac_buffer = io.BytesIO()
        sf.write(flac_buffer, audio_data, sample_rate, format='FLAC')
        flac_buffer.seek(0)
        return flac_buffer.read(), content_type
    
    if target_format == "pcm":
        # Return raw 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes(), content_type
    
    # For mp3, opus, aac - use ffmpeg
    try:
        # Write WAV to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            sf.write(wav_file.name, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            wav_path = wav_file.name
        
        # Output file
        output_path = wav_path.replace('.wav', f'.{target_format}')
        
        # Build ffmpeg command based on format
        if target_format == "mp3":
            cmd = ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-b:a', '192k', output_path]
        elif target_format == "opus":
            cmd = ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libopus', '-b:a', '128k', output_path]
        elif target_format == "aac":
            cmd = ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'aac', '-b:a', '192k', output_path]
        else:
            # Fallback to mp3
            cmd = ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-b:a', '192k', output_path]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            # Fallback to WAV on error
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            return wav_buffer.read(), "audio/wav"
        
        # Read output
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Cleanup
        os.unlink(wav_path)
        os.unlink(output_path)
        
        return audio_bytes, content_type
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        # Fallback to WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return wav_buffer.read(), "audio/wav"


@app.post("/v1/audio/speech", response_class=Response)
async def openai_create_speech(
    request: OpenAISpeechRequest,
    http_request: Request
):
    """
    OpenAI-compatible Text-to-Speech endpoint.
    
    Generates audio from the input text using the specified voice.
    This endpoint is compatible with the OpenAI /v1/audio/speech API.
    
    - **input**: The text to generate audio for (max 4096 characters)
    - **model**: TTS model (tts-1, tts-1-hd accepted for compatibility)
    - **voice**: Voice name - OpenAI voices (alloy, echo, etc.) or Speaker voice names
    - **response_format**: mp3, opus, aac, flac, wav, or pcm
    - **speed**: Speed of audio (0.25 to 4.0)
    
    Returns binary audio data in the requested format.
    """
    check_rate_limit(http_request)
    
    try:
        # Resolve voice name
        available_voices = tts_service.get_voices()
        
        if request.voice in OPENAI_VOICE_MAPPING:
            # OpenAI voice name - map to first available Speaker voice
            if len(available_voices) == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No voices available"
                )
            voice_name = available_voices[0]
            logger.debug(f"Mapped OpenAI voice '{request.voice}' to Speaker voice '{voice_name}'")
        elif request.voice in available_voices:
            # Direct Speaker voice name
            voice_name = request.voice
        else:
            # Unknown voice - try to use it anyway, let backend handle error
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{request.voice}' not found. Available: {', '.join(available_voices[:10])}"
            )
        
        logger.info(f"[OpenAI API] Generating speech: input='{request.input[:50]}...', voice={voice_name}, format={request.response_format.value}")
        
        # Generate audio using the TTS backend
        async with tts_inference_semaphore:
            audio, sample_rate = await asyncio.to_thread(
                tts_service.generate_speech,
                text=request.input,
                voice_name=voice_name,
                language="en"  # Default to English for OpenAI compatibility
            )
        
        if audio is None or len(audio) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate audio"
            )
        
        # Convert to requested format
        audio_bytes, content_type = convert_audio_format(
            audio, sample_rate, request.response_format.value
        )
        
        # Log for audit
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent")
        audit_logger.log_tts_generation(
            voice_name,
            len(request.input),
            "en",
            client_ip,
            user_agent
        )
        
        # Record metrics
        metrics_collector.record_request(
            endpoint="/v1/audio/speech",
            method="POST",
            status_code=200,
            response_time=0,
            voice_name=voice_name,
            language="en",
            text_length=len(request.input),
            text=request.input
        )
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Length": str(len(audio_bytes)),
                "X-Request-Id": str(time.time_ns()),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns service health information including model status and version.
    """
    try:
        # Basic health check
        voices = tts_service.get_voices()
        
        # Get system health
        health_data = get_health()
        
        return HealthResponse(
            status=health_data['status'],
            service="TTS",
            version=version.app_version,
            model=tts_service.model_name,
            available_voices=len(voices),
            timestamp=datetime.utcnow(),
            uptime=health_data['system'].get('uptime'),
            memory_usage=health_data['system'].get('memory_usage'),
            gpu_usage=health_data['gpu']
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "TTS", 
                "version": version.app_version,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/languages", response_model=LanguageListResponse)
async def get_supported_languages():
    """
    Get list of supported languages for TTS generation.
    
    Returns available language codes and their names.
    """
    languages = {
        "en": "English",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "sv": "Swedish",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish",
        "cs": "Czech",
        "hu": "Hungarian",
        "ro": "Romanian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian"
    }
    return LanguageListResponse(languages=languages, total_count=len(languages))

@app.delete("/voices/{voice_name}", response_model=VoiceDeleteResponse)
async def delete_voice(voice_name: str, request: Request):
    """
    Delete a voice and all its associated audio files.
    
    - **voice_name**: Name of the voice to delete
    """
    # Check rate limit
    check_rate_limit(request)
    
    voice_dir = f"data/voices/{voice_name}"
    
    if not os.path.exists(voice_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{voice_name}' not found"
        )
    
    try:
        import shutil
        shutil.rmtree(voice_dir)
        
        # Reload voices to reflect the deletion
        tts_service.load_voices()
        
        # Log for audit
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent")
        audit_logger.log_voice_delete(voice_name, client_ip, user_agent)
        
        return VoiceDeleteResponse(
            message=f"Voice '{voice_name}' deleted successfully",
            deleted_voice=voice_name
        )
    except Exception as e:
        logger.error(f"Error deleting voice: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_api_metrics():
    """
    Get API usage metrics and performance statistics.
    
    Returns detailed metrics about API usage, performance, and system health.
    """
    metrics = get_metrics()
    return MetricsResponse(**metrics)

@app.get("/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping by monitoring systems.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
