"""Configuration dataclasses for Qwen3-TTS backend."""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QwenTTSConfig:
    """Configuration for Qwen3-TTS backend.
    
    Attributes:
        model_size: Model variant - "1.7B" (full) or "0.6B" (lightweight)
        enable_custom_voice: Enable predefined speaker synthesis
        enable_voice_design: Enable voice creation from text descriptions
        enable_voice_clone: Enable zero-shot voice cloning
        enable_streaming: Enable streaming audio generation
        device: CUDA device (None for auto-detect)
        dtype: Model precision - "float16", "bfloat16", or "float32"
        use_flash_attention: Use FlashAttention 2 for memory efficiency
    """
    
    # Model selection
    model_size: str = "1.7B"  # "1.7B" or "0.6B"
    
    # Model paths (None = download from HuggingFace)
    custom_voice_model_path: Optional[str] = None
    voice_design_model_path: Optional[str] = None
    voice_clone_model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Feature flags
    enable_custom_voice: bool = True
    enable_voice_design: bool = True
    enable_voice_clone: bool = True
    enable_streaming: bool = True
    streaming_compile: bool = True
    streaming_compile_mode: str = "reduce-overhead"
    streaming_decode_window: int = 80
    streaming_emit_every: int = 16
    streaming_first_chunk_emit: int = 8
    streaming_first_chunk_window: int = 48
    streaming_first_chunk_frames: int = 48
    
    # Hardware configuration
    device: Optional[str] = None  # None = auto-detect
    dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    use_flash_attention: bool = True
    
    # Generation defaults
    default_language: str = "Auto"
    default_speaker: str = "Vivian"
    max_new_tokens: int = 2048
    
    # Performance tuning
    max_workers: int = 4
    cache_voice_prompts: bool = True
    max_cached_prompts: int = 100
    
    # vLLM-Omni settings (optional high-performance mode)
    use_vllm: bool = True
    vllm_tensor_parallel_size: int = 1
    
    @classmethod
    def from_env(cls) -> "QwenTTSConfig":
        """Create config from environment variables.
        
        Environment variables:
            QWEN_TTS_MODEL_SIZE: "0.6B" or "1.7B"
            QWEN_TTS_DEVICE: e.g., "cuda:0"
            QWEN_TTS_DTYPE: "float16", "bfloat16", "float32"
            QWEN_TTS_FLASH_ATTENTION: "true" or "false"
            QWEN_TTS_ENABLE_CUSTOM_VOICE: "true" or "false"
            QWEN_TTS_ENABLE_VOICE_DESIGN: "true" or "false"
            QWEN_TTS_ENABLE_VOICE_CLONE: "true" or "false"
        """
        def str_to_bool(s: str) -> bool:
            return s.lower() in ("true", "1", "yes")
        
        model_size = os.getenv("QWEN_TTS_MODEL_SIZE", "1.7B")
        
        # For 0.6B model, voice_design is not available
        enable_voice_design = str_to_bool(os.getenv("QWEN_TTS_ENABLE_VOICE_DESIGN", "true"))
        if model_size == "0.6B":
            enable_voice_design = False

        
        return cls(
            model_size=model_size,
            device=os.getenv("QWEN_TTS_DEVICE"),
            dtype=os.getenv("QWEN_TTS_DTYPE", "bfloat16"),
            use_flash_attention=str_to_bool(os.getenv("QWEN_TTS_FLASH_ATTENTION", "true")),
            enable_custom_voice=str_to_bool(os.getenv("QWEN_TTS_ENABLE_CUSTOM_VOICE", "true")),
            enable_voice_design=enable_voice_design,
            enable_voice_clone=str_to_bool(os.getenv("QWEN_TTS_ENABLE_VOICE_CLONE", "true")),
            enable_streaming=str_to_bool(os.getenv("QWEN_TTS_ENABLE_STREAMING", "true")),
            streaming_compile=str_to_bool(os.getenv("QWEN_TTS_STREAMING_COMPILE", "true")),
            streaming_compile_mode=os.getenv("QWEN_TTS_STREAMING_COMPILE_MODE", "reduce-overhead"),
            streaming_decode_window=int(os.getenv("QWEN_TTS_STREAMING_DECODE_WINDOW", "80")),
            streaming_emit_every=int(os.getenv("QWEN_TTS_STREAMING_EMIT_EVERY", "16")),
            streaming_first_chunk_emit=int(os.getenv("QWEN_TTS_STREAMING_FIRST_CHUNK_EMIT", "8")),
            streaming_first_chunk_window=int(os.getenv("QWEN_TTS_STREAMING_FIRST_CHUNK_WINDOW", "48")),
            streaming_first_chunk_frames=int(os.getenv("QWEN_TTS_STREAMING_FIRST_CHUNK_FRAMES", "48")),
        )
    
    @classmethod
    def from_dict(cls, config: dict) -> "QwenTTSConfig":
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config.items()
            if k in cls.__dataclass_fields__
        })
    
    def validate(self) -> None:
        """Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.model_size not in ("1.7B", "0.6B"):
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be '1.7B' or '0.6B'")
        
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be 'float16', 'bfloat16', or 'float32'")
        
        if self.model_size == "0.6B" and self.enable_voice_design:
            raise ValueError("VoiceDesign not available for 0.6B model. Set enable_voice_design=False")
