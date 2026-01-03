import os
import random
import logging
import numpy as np
import io
import soundfile as sf
import time
from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel
from pydantic import BaseModel, Field, constr
from fastapi import UploadFile, File, HTTPException, Depends, Request
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
    HealthResponse, APIInfo, MetricsResponse, ErrorResponse, ErrorDetail
)
import app.version as version

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
    file: UploadFile = File(...)
):
    """
    Add a new voice by uploading an audio file.
    
    - **voice_name**: Name for the voice (alphanumeric and underscores only)
    - **file**: Audio file (.wav or .mp3) containing voice sample
    
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

        logger.info(
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
        
        audio, sample_rate = tts_service.generate_speech(
            text=text,
            voice_name=request.voice_name,
            language=request.language,
            **glm_kwargs
        )

        # Debug logging
        logger.debug(f"generate_speech(): {type(audio) = }")    
        if len(audio) > 100:
            random_start_index = random.randint(0, len(audio) - 100)
            logger.debug(f"generate_speech(): {audio[random_start_index:random_start_index + 100] = }")

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

        logger.debug(f"Successfully generated audio ({output_format}): {len(audio_bytes)} bytes")
        
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
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
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
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
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
    uvicorn.run(app, host="0.0.0.0", port=8010)
