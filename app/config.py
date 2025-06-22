"""
Configuration management for the TTS API.
Handles environment-specific settings and configuration validation.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from pathlib import Path

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_title: str = Field(default="Speaker TTS API")
    api_version: str = Field(default="1.0.0")
    api_description: str = Field(
        default="High-quality Text-to-Speech API with XTTS v2 voice cloning"
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8010)
    debug: bool = Field(default=False)
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["*"], 
        description="List of allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["*"])
    cors_allow_headers: List[str] = Field(default=["*"])
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # TTS Model Configuration
    model_name: str = Field(
        default="tts_models--multilingual--multi-dataset--tts_models--multilingual--multi-dataset--xtts_v2"
    )
    use_deepspeed: bool = Field(default=True)
    tau: float = Field(default=0.3)
    gpt_cond_len: int = Field(default=3)
    top_k: int = Field(default=3)
    top_p: int = Field(default=5)
    
    # File Storage
    voice_data_dir: str = Field(default="data/voices")
    max_file_size_mb: int = Field(default=50)
    allowed_audio_extensions: List[str] = Field(
        default=[".wav", ".mp3", ".flac", ".m4a"]
    )
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    audit_log_file: str = Field(default="logs/audit.log")
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    enable_health_checks: bool = Field(default=True)
    metrics_history_size: int = Field(default=1000)
    
    # Security
    enable_audit_logging: bool = Field(default=True)
    api_key_required: bool = Field(default=False)
    api_keys: List[str] = Field(default=[])
    
    # Performance
    max_text_length: int = Field(default=2000)
    min_text_length: int = Field(default=1)
    max_concurrent_requests: int = Field(default=10)
    
    # Environment
    environment: str = Field(default="development")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'environment must be one of {valid_envs}')
        return v.lower()
    
    @validator('cors_origins', 'cors_allow_methods', 'cors_allow_headers')
    def validate_cors_lists(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    @validator('allowed_audio_extensions')
    def validate_audio_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v
    
    @validator('api_keys')
    def validate_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["*"]
    enable_audit_logging: bool = True
    api_key_required: bool = False

class StagingSettings(Settings):
    """Staging environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = ["https://staging.example.com"]
    enable_audit_logging: bool = True
    api_key_required: bool = True

class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "WARNING"
    cors_origins: List[str] = ["https://api.example.com"]
    enable_audit_logging: bool = True
    api_key_required: bool = True
    max_concurrent_requests: int = 50

def get_settings() -> Settings:
    """Get settings based on environment"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "staging":
        return StagingSettings()
    else:
        return DevelopmentSettings()

# Global settings instance
settings = get_settings()

def validate_configuration():
    """Validate configuration and create necessary directories"""
    # Create necessary directories
    Path(settings.voice_data_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Validate file paths
    if not os.path.exists(settings.voice_data_dir):
        raise ValueError(f"Voice data directory does not exist: {settings.voice_data_dir}")
    
    # Validate model configuration
    if settings.tau <= 0 or settings.tau > 1:
        raise ValueError("tau must be between 0 and 1")
    
    if settings.gpt_cond_len <= 0:
        raise ValueError("gpt_cond_len must be positive")
    
    if settings.top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if settings.top_p <= 0:
        raise ValueError("top_p must be positive")
    
    # Validate rate limiting
    if settings.rate_limit_requests <= 0:
        raise ValueError("rate_limit_requests must be positive")
    
    if settings.rate_limit_window <= 0:
        raise ValueError("rate_limit_window must be positive")
    
    return True

def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration"""
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers,
    }

def get_model_config() -> Dict[str, Any]:
    """Get TTS model configuration"""
    return {
        "model_name": settings.model_name,
        "use_deepspeed": settings.use_deepspeed,
        "tau": settings.tau,
        "gpt_cond_len": settings.gpt_cond_len,
        "top_k": settings.top_k,
        "top_p": settings.top_p,
    }

def is_production() -> bool:
    """Check if running in production environment"""
    return settings.environment == "production"

def is_development() -> bool:
    """Check if running in development environment"""
    return settings.environment == "development"

def is_staging() -> bool:
    """Check if running in staging environment"""
    return settings.environment == "staging" 