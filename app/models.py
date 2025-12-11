from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class LanguageCode(str, Enum):
    """Supported language codes for TTS"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    POLISH = "pl"
    TURKISH = "tr"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    CZECH = "cs"
    HUNGARIAN = "hu"
    ROMANIAN = "ro"
    BULGARIAN = "bg"
    CROATIAN = "hr"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ESTONIAN = "et"
    LATVIAN = "lv"
    LITHUANIAN = "lt"

class EmotionType(str, Enum):
    """Supported emotion types for TTS"""
    HAPPY = "(happy)"
    SAD = "(sad)"
    ANGRY = "(angry)"
    EXCITED = "(excited)"
    CALM = "(calm)"
    NERVOUS = "(nervous)"
    CONFIDENT = "(confident)"
    SHY = "(shy)"
    SERIOUS = "(serious)"
    PLAYFUL = "(playful)"

class OutputFormat(str, Enum):
    """Output audio format"""
    RAW = "raw"  # Raw float32 PCM bytes (default, backward compatible)
    WAV = "wav"  # Proper WAV file with headers

# Request Models
class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str = Field(
        description="Text to convert to speech",
        min_length=1, 
        max_length=2000,
        examples=["Hello, this is a test of the TTS system."]
    )
    voice_name: str = Field(
        description="Name of the voice to use for cloning",
        examples=["demo_1"]
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH, 
        description="Two-letter language code"
    )
    temperature: float = Field(
        default=0.8, 
        ge=0.1, 
        le=1.0, 
        description="Temperature parameter for randomness"
    )
    top_p: float = Field(
        default=0.9, 
        ge=0.1, 
        le=1.0, 
        description="Top-p sampling parameter"
    )
    emotion: Optional[EmotionType] = Field(
        default=None, 
        description="Emotion tag to control speech style"
    )
    speed: float = Field(
        default=1.0, 
        ge=0.5, 
        le=2.0, 
        description="Speech speed multiplier"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.RAW,
        description="Output audio format: 'raw' for float32 PCM bytes (default), 'wav' for proper WAV file"
    )
    
    # GLM-TTS specific parameters (optional, uses defaults if not provided)
    sampling: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description="GLM-TTS: Top-k sampling value (default: 25)"
    )
    min_token_text_ratio: Optional[float] = Field(
        default=None,
        ge=1.0, le=20.0,
        description="GLM-TTS: Min audio tokens per text token (default: 8)"
    )
    max_token_text_ratio: Optional[float] = Field(
        default=None,
        ge=10.0, le=100.0,
        description="GLM-TTS: Max audio tokens per text token (default: 30)"
    )
    beam_size: Optional[int] = Field(
        default=None,
        ge=1, le=5,
        description="GLM-TTS: Beam search width (default: 1)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.1, le=2.0,
        description="GLM-TTS: Sampling temperature (default: 1.0)"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.1, le=1.0,
        description="GLM-TTS: Nucleus sampling threshold (default: 0.8)"
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="GLM-TTS: Repetition penalty threshold (default: 0.1)"
    )
    sample_method: Optional[str] = Field(
        default=None,
        description="GLM-TTS: Sampling method - 'ras' (default) or 'topk'"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, this is a test of the TTS system.",
                "voice_name": "demo_1",
                "language": "en",
                "temperature": 0.8,
                "top_p": 0.9,
                "emotion": "(happy)",
                "speed": 1.0
            }
        }

class VoiceUploadRequest(BaseModel):
    """Request model for voice upload"""
    voice_name: str = Field(
        description="Name for the voice (alphanumeric and underscores only)",
        examples=["my_voice"]
    )

    @validator('voice_name')
    def validate_voice_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Voice name must contain only letters, numbers, and underscores')
        return v

# Response Models
class TTSResponse(BaseModel):
    """Response model for TTS generation"""
    audio_data: bytes = Field(description="Audio data in WAV format")
    sample_rate: int = Field(description="Sample rate of the audio")
    duration: float = Field(description="Duration of the audio in seconds")
    voice_name: str = Field(description="Voice used for generation")
    language: str = Field(description="Language used for generation")
    text_length: int = Field(description="Length of input text")

class VoiceInfo(BaseModel):
    """Model for voice information"""
    name: str = Field(description="Voice name")
    file_count: int = Field(description="Number of audio files for this voice")
    total_duration: Optional[float] = Field(default=None, description="Total duration of all samples")
    created_at: Optional[datetime] = Field(default=None, description="When the voice was created")
    last_used: Optional[datetime] = Field(default=None, description="When the voice was last used")

class VoiceListResponse(BaseModel):
    """Response model for voice listing"""
    voices: List[str] = Field(description="List of available voice names")
    total_count: int = Field(description="Total number of voices")
    voice_details: Optional[List[VoiceInfo]] = Field(default=None, description="Detailed voice information")

class VoiceUploadResponse(BaseModel):
    """Response model for voice upload"""
    message: str = Field(description="Success message")
    voice_name: str = Field(description="Name of the uploaded voice")
    file_name: str = Field(description="Name of the uploaded file")
    file_size: int = Field(description="Size of the uploaded file in bytes")

class VoiceDeleteResponse(BaseModel):
    """Response model for voice deletion"""
    message: str = Field(description="Success message")
    deleted_voice: str = Field(description="Name of the deleted voice")

class LanguageInfo(BaseModel):
    """Model for language information"""
    code: str = Field(description="Two-letter language code")
    name: str = Field(description="Full language name")
    native_name: Optional[str] = Field(default=None, description="Language name in native script")

class LanguageListResponse(BaseModel):
    """Response model for language listing"""
    languages: Dict[str, str] = Field(description="Mapping of language codes to names")
    total_count: int = Field(description="Total number of supported languages")

# Health and Status Models
class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: HealthStatus = Field(description="Service health status")
    service: str = Field(description="Service name")
    version: str = Field(description="API version")
    model: str = Field(description="TTS model name")
    available_voices: int = Field(description="Number of available voices")
    timestamp: datetime = Field(description="Health check timestamp")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")
    memory_usage: Optional[Dict[str, Any]] = Field(default=None, description="Memory usage information")
    gpu_usage: Optional[Dict[str, Any]] = Field(default=None, description="GPU usage information")

# Error Models
class ErrorDetail(BaseModel):
    """Model for error details"""
    code: int = Field(description="HTTP status code")
    message: str = Field(description="Error message")
    timestamp: datetime = Field(description="Error timestamp")
    path: str = Field(description="Request path")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: ErrorDetail = Field(description="Error information")

# API Information Models
class APIInfo(BaseModel):
    """Model for API information"""
    message: str = Field(description="Welcome message")
    version: str = Field(description="API version")
    model: str = Field(description="TTS model name")
    status: str = Field(description="Service status")
    documentation: str = Field(description="Documentation URL")
    health_check: str = Field(description="Health check endpoint")

# Monitoring Models
class MetricsResponse(BaseModel):
    """Response model for metrics"""
    total_requests: int = Field(description="Total number of requests")
    successful_requests: int = Field(description="Number of successful requests")
    failed_requests: int = Field(description="Number of failed requests")
    average_response_time: float = Field(description="Average response time in seconds")
    requests_per_minute: float = Field(description="Requests per minute")
    active_voices: int = Field(description="Number of active voices")
    model_load_time: Optional[float] = Field(default=None, description="Model load time in seconds")
    last_request_time: Optional[datetime] = Field(default=None, description="Timestamp of last request")

# Legacy models for backward compatibility
class ItemBase(BaseModel):
    name: str = Field(examples=["Laptop"])
    description: str = Field(examples=["A high-performance laptop"])

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int = Field(examples=[1])

    class Config:
        orm_mode = True

class ItemUpdate(ItemBase):
    name: Optional[str] = Field(default=None, examples=["Updated Laptop"])
    description: Optional[str] = Field(default=None, examples=["An updated high-performance laptop"])

class ItemList(BaseModel):
    items: List[Item]

class Message(BaseModel):
    message: str = Field(examples=["Operation successful"])