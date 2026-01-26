# Development Plan: Qwen3-TTS Backend for Speaker

## Executive Summary

This document outlines the implementation plan for adding Qwen3-TTS as a new backend engine to the Speaker platform. Qwen3-TTS is Alibaba's open-source TTS model family offering voice cloning, voice design, and high-quality multilingual speech synthesis with 3-second voice cloning capability.

---

## 1. Architecture Overview

### 1.1 Current Speaker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Speaker Platform                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)  │  FastAPI Backend  │  Monitoring Stack  │
├─────────────────────────────────────────────────────────────┤
│                     Backend Engines                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   GLM-TTS    │  │   vLLM       │  │  TensorRT    │       │
│  │  (Primary)   │  │  (LLM Stage) │  │  (Vocoder)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Proposed Architecture with Qwen3-TTS

```
┌─────────────────────────────────────────────────────────────┐
│                    Speaker Platform                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)  │  FastAPI Backend  │  Monitoring Stack  │
├─────────────────────────────────────────────────────────────┤
│                     Backend Engines                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   GLM-TTS    │  │  Qwen3-TTS   │  │   Future     │       │
│  │  (Legacy)    │  │   (New)      │  │  Backends    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                  │                                 │
│         └────────┬─────────┘                                 │
│                  ▼                                           │
│         ┌──────────────────┐                                 │
│         │  Backend Router  │                                 │
│         │  (Engine Select) │                                 │
│         └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Files to Create/Modify

### 2.1 New Files to Create

| File Path | Description |
|-----------|-------------|
| `app/backends/qwen_tts.py` | Main Qwen3-TTS backend implementation |
| `app/backends/qwen_tts_config.py` | Configuration dataclasses for Qwen3-TTS |
| `app/backends/qwen_tts_streaming.py` | Streaming audio generation support |
| `app/models/qwen_voices.py` | Voice definitions and presets |
| `tests/test_qwen_tts_backend.py` | Unit tests for Qwen3-TTS backend |
| `tests/test_qwen_tts_integration.py` | Integration tests |
| `docs/qwen_tts_backend.md` | Backend documentation |
| `scripts/download_qwen_models.py` | Model download utility script |

### 2.2 Existing Files to Modify

| File Path | Modification |
|-----------|--------------|
| `app/backends/__init__.py` | Register Qwen3-TTS backend |
| `app/api/routes.py` | Add Qwen3-TTS specific endpoints |
| `app/core/engine_factory.py` | Add Qwen3-TTS engine creation |
| `app/models/voice.py` | Extend voice model for Qwen speakers |
| `config.yaml` | Add Qwen3-TTS configuration section |
| `requirements-ml.txt` | Add `qwen-tts` dependency |
| `docker-compose.yml` | Add Qwen3-TTS service configuration |
| `Dockerfile` | Include Qwen3-TTS dependencies |
| `frontend/src/components/VoiceSelector.tsx` | Add Qwen voice options |

---

## 3. Implementation Phases

### Phase 1: Core Backend Implementation (Week 1-2)

#### 3.1.1 Create Base Backend Class

**File: `app/backends/qwen_tts.py`**

```python
"""
Qwen3-TTS Backend Implementation for Speaker Platform

This module implements the Qwen3-TTS backend, providing:
- Custom voice generation with built-in speakers
- Voice design from natural language descriptions
- Zero-shot voice cloning from 3-second audio
- Streaming audio generation
"""

import torch
import numpy as np
from typing import Optional, List, Union, Tuple, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from app.backends.base import BaseBackend, SynthesisResult
from app.backends.qwen_tts_config import QwenTTSConfig
from app.models.qwen_voices import QWEN_SPEAKERS, QwenVoice
from app.core.metrics import MetricsCollector


class QwenTTSBackend(BaseBackend):
    """
    Qwen3-TTS backend implementation supporting multiple generation modes:
    - CustomVoice: Use predefined speakers with instruction control
    - VoiceDesign: Create voices from natural language descriptions
    - VoiceClone: Clone voices from reference audio
    """
    
    SUPPORTED_MODELS = {
        "custom_voice": {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        },
        "voice_design": {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        },
        "voice_clone": {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }
    }
    
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean", 
        "German", "French", "Russian", "Portuguese", 
        "Spanish", "Italian", "Auto"
    ]

    def __init__(self, config: QwenTTSConfig):
        super().__init__()
        self.config = config
        self.models = {}
        self.tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._voice_clone_prompts = {}  # Cache for voice clone prompts
        self.metrics = MetricsCollector("qwen_tts")
        
    async def initialize(self) -> None:
        """Initialize Qwen3-TTS models based on configuration."""
        device = self.config.device or "cuda:0"
        dtype = getattr(torch, self.config.dtype)
        
        # Load models based on enabled modes
        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
        }
        
        if self.config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load CustomVoice model (primary)
        if self.config.enable_custom_voice:
            model_id = self.SUPPORTED_MODELS["custom_voice"][self.config.model_size]
            self.models["custom_voice"] = Qwen3TTSModel.from_pretrained(
                self.config.custom_voice_model_path or model_id,
                **load_kwargs
            )
            self._log(f"Loaded CustomVoice model: {model_id}")
        
        # Load VoiceDesign model
        if self.config.enable_voice_design:
            model_id = self.SUPPORTED_MODELS["voice_design"]["1.7B"]
            self.models["voice_design"] = Qwen3TTSModel.from_pretrained(
                self.config.voice_design_model_path or model_id,
                **load_kwargs
            )
            self._log(f"Loaded VoiceDesign model: {model_id}")
        
        # Load Base model for voice cloning
        if self.config.enable_voice_clone:
            model_id = self.SUPPORTED_MODELS["voice_clone"][self.config.model_size]
            self.models["voice_clone"] = Qwen3TTSModel.from_pretrained(
                self.config.voice_clone_model_path or model_id,
                **load_kwargs
            )
            self._log(f"Loaded VoiceClone model: {model_id}")
        
        # Load tokenizer for audio encoding/decoding
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self.config.tokenizer_path or "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map=device
        )
        
        self._initialized = True
        self._log("Qwen3-TTS backend initialized successfully")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "Auto",
        instruct: Optional[str] = None,
        mode: str = "custom_voice",
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text using specified mode.
        
        Args:
            text: Input text to synthesize
            voice: Speaker name (for custom_voice) or voice description
            language: Target language or "Auto" for detection
            instruct: Instruction for style control
            mode: Generation mode (custom_voice, voice_design, voice_clone)
            **kwargs: Additional generation parameters
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        self._ensure_initialized()
        
        with self.metrics.measure_latency(f"synthesize_{mode}"):
            if mode == "custom_voice":
                return await self._synthesize_custom_voice(
                    text, voice, language, instruct, **kwargs
                )
            elif mode == "voice_design":
                return await self._synthesize_voice_design(
                    text, language, instruct, **kwargs
                )
            elif mode == "voice_clone":
                return await self._synthesize_voice_clone(
                    text, language, **kwargs
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

    async def _synthesize_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str],
        **kwargs
    ) -> SynthesisResult:
        """Generate speech using predefined speaker with optional instruction."""
        model = self.models.get("custom_voice")
        if not model:
            raise RuntimeError("CustomVoice model not loaded")
        
        # Validate speaker
        if speaker not in QWEN_SPEAKERS:
            available = list(QWEN_SPEAKERS.keys())
            raise ValueError(f"Unknown speaker '{speaker}'. Available: {available}")
        
        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        wavs, sr = await loop.run_in_executor(
            self._executor,
            lambda: model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or "",
                **self._get_generation_kwargs(kwargs)
            )
        )
        
        return self._create_result(wavs[0], sr, {
            "mode": "custom_voice",
            "speaker": speaker,
            "language": language,
            "instruct": instruct
        })

    async def _synthesize_voice_design(
        self,
        text: str,
        language: str,
        instruct: str,
        **kwargs
    ) -> SynthesisResult:
        """Generate speech with voice designed from natural language description."""
        model = self.models.get("voice_design")
        if not model:
            raise RuntimeError("VoiceDesign model not loaded")
        
        if not instruct:
            raise ValueError("Voice design requires an 'instruct' description")
        
        loop = asyncio.get_event_loop()
        wavs, sr = await loop.run_in_executor(
            self._executor,
            lambda: model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **self._get_generation_kwargs(kwargs)
            )
        )
        
        return self._create_result(wavs[0], sr, {
            "mode": "voice_design",
            "language": language,
            "instruct": instruct
        })

    async def _synthesize_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, bytes, Tuple[np.ndarray, int]]] = None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[dict] = None,
        x_vector_only_mode: bool = False,
        **kwargs
    ) -> SynthesisResult:
        """Clone voice from reference audio and synthesize new content."""
        model = self.models.get("voice_clone")
        if not model:
            raise RuntimeError("VoiceClone model not loaded")
        
        # Use cached prompt or create new one
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either voice_clone_prompt or ref_audio required")
            
            loop = asyncio.get_event_loop()
            voice_clone_prompt = await loop.run_in_executor(
                self._executor,
                lambda: model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode
                )
            )
        
        loop = asyncio.get_event_loop()
        wavs, sr = await loop.run_in_executor(
            self._executor,
            lambda: model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **self._get_generation_kwargs(kwargs)
            )
        )
        
        return self._create_result(wavs[0], sr, {
            "mode": "voice_clone",
            "language": language,
        })

    async def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, bytes, Tuple[np.ndarray, int]],
        ref_text: Optional[str] = None,
        voice_id: Optional[str] = None,
        x_vector_only_mode: bool = False
    ) -> dict:
        """
        Pre-compute voice clone prompt for reuse across multiple generations.
        
        Args:
            ref_audio: Reference audio (path, URL, bytes, or numpy array)
            ref_text: Transcript of reference audio (optional but recommended)
            voice_id: Optional ID to cache the prompt
            x_vector_only_mode: Use only speaker embedding (lower quality)
            
        Returns:
            Voice clone prompt dictionary
        """
        model = self.models.get("voice_clone")
        if not model:
            raise RuntimeError("VoiceClone model not loaded")
        
        loop = asyncio.get_event_loop()
        prompt = await loop.run_in_executor(
            self._executor,
            lambda: model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode
            )
        )
        
        if voice_id:
            self._voice_clone_prompts[voice_id] = prompt
        
        return prompt

    def get_cached_voice_prompt(self, voice_id: str) -> Optional[dict]:
        """Retrieve cached voice clone prompt by ID."""
        return self._voice_clone_prompts.get(voice_id)

    def get_supported_speakers(self) -> List[QwenVoice]:
        """Return list of available predefined speakers."""
        return list(QWEN_SPEAKERS.values())

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()

    def _get_generation_kwargs(self, kwargs: dict) -> dict:
        """Extract and validate generation parameters."""
        valid_keys = {
            "max_new_tokens", "top_p", "top_k", 
            "temperature", "repetition_penalty"
        }
        return {
            k: v for k, v in kwargs.items() 
            if k in valid_keys and v is not None
        }

    def _create_result(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        metadata: dict
    ) -> SynthesisResult:
        """Create standardized synthesis result."""
        duration = len(audio) / sample_rate
        
        self.metrics.record_audio_generated(duration)
        
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            backend="qwen_tts",
            metadata=metadata
        )

    async def encode_audio(
        self,
        audio: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Encode audio to Qwen3-TTS tokens."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.tokenizer.encode(audio)
        )

    async def decode_audio(self, tokens: np.ndarray) -> Tuple[np.ndarray, int]:
        """Decode Qwen3-TTS tokens back to audio."""
        loop = asyncio.get_event_loop()
        wavs, sr = await loop.run_in_executor(
            self._executor,
            lambda: self.tokenizer.decode(tokens)
        )
        return wavs[0], sr

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        self.models.clear()
        self._voice_clone_prompts.clear()
        self.tokenizer = None
        self._initialized = False
        self._log("Qwen3-TTS backend shutdown complete")
```

#### 3.1.2 Configuration Class

**File: `app/backends/qwen_tts_config.py`**

```python
"""Configuration dataclasses for Qwen3-TTS backend."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QwenTTSConfig:
    """Configuration for Qwen3-TTS backend."""
    
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
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    
    @classmethod
    def from_dict(cls, config: dict) -> "QwenTTSConfig":
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config.items()
            if k in cls.__dataclass_fields__
        })
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.model_size not in ("1.7B", "0.6B"):
            raise ValueError(f"Invalid model_size: {self.model_size}")
        
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        
        if self.model_size == "0.6B" and self.enable_voice_design:
            raise ValueError("VoiceDesign not available for 0.6B model")
```

#### 3.1.3 Voice Definitions

**File: `app/models/qwen_voices.py`**

```python
"""Qwen3-TTS voice definitions and presets."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QwenVoice:
    """Represents a Qwen3-TTS predefined speaker."""
    
    id: str
    name: str
    description: str
    native_language: str
    supported_languages: List[str]
    gender: str
    age_range: str
    style_tags: List[str]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "native_language": self.native_language,
            "supported_languages": self.supported_languages,
            "gender": self.gender,
            "age_range": self.age_range,
            "style_tags": self.style_tags,
            "backend": "qwen_tts"
        }


# Predefined speakers from Qwen3-TTS-CustomVoice
QWEN_SPEAKERS = {
    "Vivian": QwenVoice(
        id="vivian",
        name="Vivian",
        description="Bright, slightly edgy young female voice",
        native_language="Chinese",
        supported_languages=["Chinese", "English", "Japanese", "Korean", 
                           "German", "French", "Russian", "Portuguese", 
                           "Spanish", "Italian"],
        gender="female",
        age_range="young_adult",
        style_tags=["bright", "edgy", "energetic"]
    ),
    "Serena": QwenVoice(
        id="serena",
        name="Serena",
        description="Warm, gentle young female voice",
        native_language="Chinese",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="female",
        age_range="young_adult",
        style_tags=["warm", "gentle", "soothing"]
    ),
    "Uncle_Fu": QwenVoice(
        id="uncle_fu",
        name="Uncle Fu",
        description="Seasoned male voice with a low, mellow timbre",
        native_language="Chinese",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="male",
        age_range="mature",
        style_tags=["deep", "mellow", "authoritative"]
    ),
    "Dylan": QwenVoice(
        id="dylan",
        name="Dylan",
        description="Youthful Beijing male voice with clear, natural timbre",
        native_language="Chinese",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="male",
        age_range="young_adult",
        style_tags=["youthful", "clear", "natural", "beijing_dialect"]
    ),
    "Eric": QwenVoice(
        id="eric",
        name="Eric",
        description="Lively Chengdu male voice with slightly husky brightness",
        native_language="Chinese",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="male",
        age_range="young_adult",
        style_tags=["lively", "husky", "sichuan_dialect"]
    ),
    "Ryan": QwenVoice(
        id="ryan",
        name="Ryan",
        description="Dynamic male voice with strong rhythmic drive",
        native_language="English",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="male",
        age_range="adult",
        style_tags=["dynamic", "rhythmic", "energetic"]
    ),
    "Aiden": QwenVoice(
        id="aiden",
        name="Aiden",
        description="Sunny American male voice with a clear midrange",
        native_language="English",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="male",
        age_range="young_adult",
        style_tags=["sunny", "american", "clear"]
    ),
    "Ono_Anna": QwenVoice(
        id="ono_anna",
        name="Ono Anna",
        description="Playful Japanese female voice with light, nimble timbre",
        native_language="Japanese",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="female",
        age_range="young_adult",
        style_tags=["playful", "light", "nimble"]
    ),
    "Sohee": QwenVoice(
        id="sohee",
        name="Sohee",
        description="Warm Korean female voice with rich emotion",
        native_language="Korean",
        supported_languages=["Chinese", "English", "Japanese", "Korean",
                           "German", "French", "Russian", "Portuguese",
                           "Spanish", "Italian"],
        gender="female",
        age_range="young_adult",
        style_tags=["warm", "emotional", "expressive"]
    ),
}


def get_speaker_by_id(speaker_id: str) -> Optional[QwenVoice]:
    """Get speaker by ID (case-insensitive)."""
    for name, voice in QWEN_SPEAKERS.items():
        if voice.id == speaker_id.lower() or name.lower() == speaker_id.lower():
            return voice
    return None


def get_speakers_by_language(language: str) -> List[QwenVoice]:
    """Get speakers native to a specific language."""
    return [
        voice for voice in QWEN_SPEAKERS.values()
        if voice.native_language.lower() == language.lower()
    ]


def get_speakers_by_gender(gender: str) -> List[QwenVoice]:
    """Get speakers of a specific gender."""
    return [
        voice for voice in QWEN_SPEAKERS.values()
        if voice.gender.lower() == gender.lower()
    ]
```

---

### Phase 2: API Integration (Week 2-3)

#### 3.2.1 Update Backend Registry

**File: `app/backends/__init__.py`** (modification)

```python
# Add to existing imports
from app.backends.qwen_tts import QwenTTSBackend
from app.backends.qwen_tts_config import QwenTTSConfig

# Add to BACKEND_REGISTRY
BACKEND_REGISTRY = {
    "glm_tts": GLMTTSBackend,
    "qwen_tts": QwenTTSBackend,  # NEW
    # ... other backends
}

CONFIG_REGISTRY = {
    "glm_tts": GLMTTSConfig,
    "qwen_tts": QwenTTSConfig,  # NEW
}
```

#### 3.2.2 API Routes

**File: `app/api/routes.py`** (additions)

```python
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.backends.qwen_tts import QwenTTSBackend
from app.models.qwen_voices import QWEN_SPEAKERS

router = APIRouter()

# Add new Qwen3-TTS specific endpoints

@router.post("/api/v1/qwen/synthesize")
async def qwen_synthesize(
    text: str = Form(...),
    mode: str = Form("custom_voice"),
    speaker: Optional[str] = Form(None),
    language: str = Form("Auto"),
    instruct: Optional[str] = Form(None),
):
    """
    Synthesize speech using Qwen3-TTS.
    
    Modes:
    - custom_voice: Use predefined speakers with optional instruction
    - voice_design: Create voice from natural language description
    """
    backend = get_qwen_backend()
    
    result = await backend.synthesize(
        text=text,
        mode=mode,
        voice=speaker,
        language=language,
        instruct=instruct
    )
    
    return StreamingResponse(
        io.BytesIO(result.to_wav_bytes()),
        media_type="audio/wav",
        headers={"X-Audio-Duration": str(result.duration)}
    )


@router.post("/api/v1/qwen/clone")
async def qwen_voice_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
):
    """
    Clone voice from reference audio and synthesize new content.
    
    - ref_audio: 3+ seconds of clear reference audio
    - ref_text: Transcript of reference (recommended for better quality)
    """
    backend = get_qwen_backend()
    
    audio_bytes = await ref_audio.read()
    
    result = await backend.synthesize(
        text=text,
        language=language,
        mode="voice_clone",
        ref_audio=audio_bytes,
        ref_text=ref_text
    )
    
    return StreamingResponse(
        io.BytesIO(result.to_wav_bytes()),
        media_type="audio/wav"
    )


@router.post("/api/v1/qwen/voices/create-prompt")
async def create_voice_prompt(
    voice_id: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
):
    """
    Pre-compute and cache a voice clone prompt for faster repeated use.
    """
    backend = get_qwen_backend()
    
    audio_bytes = await ref_audio.read()
    
    prompt = await backend.create_voice_clone_prompt(
        ref_audio=audio_bytes,
        ref_text=ref_text,
        voice_id=voice_id
    )
    
    return {"voice_id": voice_id, "status": "cached"}


@router.get("/api/v1/qwen/speakers")
async def list_qwen_speakers():
    """List available predefined Qwen3-TTS speakers."""
    return {
        "speakers": [v.to_dict() for v in QWEN_SPEAKERS.values()]
    }


@router.get("/api/v1/qwen/languages")
async def list_qwen_languages():
    """List supported languages."""
    backend = get_qwen_backend()
    return {"languages": backend.get_supported_languages()}
```

---

### Phase 3: Streaming Support (Week 3)

#### 3.3.1 Streaming Implementation

**File: `app/backends/qwen_tts_streaming.py`**

```python
"""
Streaming audio generation for Qwen3-TTS.

Qwen3-TTS supports dual-track hybrid streaming with ~97ms first-packet latency.
"""

import asyncio
from typing import AsyncIterator, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """A chunk of streaming audio."""
    data: bytes
    sample_rate: int
    is_final: bool
    chunk_index: int
    timestamp_ms: float


class QwenTTSStreamingHandler:
    """
    Handles streaming audio generation from Qwen3-TTS.
    
    Qwen3-TTS's dual-track architecture enables:
    - First audio packet after single character input
    - End-to-end latency as low as 97ms
    """
    
    def __init__(self, backend: "QwenTTSBackend"):
        self.backend = backend
        self.chunk_size = 4096  # samples per chunk
        
    async def stream_synthesis(
        self,
        text: str,
        mode: str = "custom_voice",
        **kwargs
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio generation chunk by chunk.
        
        Note: Current qwen-tts package may not expose true streaming.
        This implementation uses chunked output from full generation.
        For true streaming, use vLLM-Omni when online serving is supported.
        """
        # Generate full audio (replace with true streaming when available)
        result = await self.backend.synthesize(text, mode=mode, **kwargs)
        
        audio = result.audio
        sr = result.sample_rate
        
        chunk_index = 0
        for i in range(0, len(audio), self.chunk_size):
            chunk_data = audio[i:i + self.chunk_size]
            is_final = (i + self.chunk_size >= len(audio))
            
            yield AudioChunk(
                data=self._to_bytes(chunk_data),
                sample_rate=sr,
                is_final=is_final,
                chunk_index=chunk_index,
                timestamp_ms=(i / sr) * 1000
            )
            
            chunk_index += 1
            await asyncio.sleep(0)  # Yield to event loop
    
    def _to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio to bytes."""
        return (audio * 32767).astype(np.int16).tobytes()
```

---

### Phase 4: Configuration & Docker (Week 3-4)

#### 3.4.1 Config Updates

**File: `config.yaml`** (additions)

```yaml
# Qwen3-TTS Backend Configuration
qwen_tts:
  enabled: true
  model_size: "1.7B"  # "1.7B" or "0.6B"
  
  # Feature toggles
  enable_custom_voice: true
  enable_voice_design: true
  enable_voice_clone: true
  enable_streaming: true
  
  # Hardware settings
  device: "cuda:0"
  dtype: "bfloat16"
  use_flash_attention: true
  
  # Local model paths (optional - leave empty to download)
  model_paths:
    custom_voice: ""
    voice_design: ""
    voice_clone: ""
    tokenizer: ""
  
  # Generation defaults
  defaults:
    language: "Auto"
    speaker: "Vivian"
    max_new_tokens: 2048
  
  # Performance
  max_workers: 4
  cache_voice_prompts: true
  max_cached_prompts: 100
  
  # vLLM high-performance mode (optional)
  vllm:
    enabled: false
    tensor_parallel_size: 1
```

#### 3.4.2 Requirements Update

**File: `requirements-ml.txt`** (additions)

```
# Qwen3-TTS
qwen-tts>=0.1.0
flash-attn>=2.0.0
```

#### 3.4.3 Docker Updates

**File: `docker-compose.yml`** (additions to services)

```yaml
services:
  speaker-api:
    # ... existing config
    environment:
      - QWEN_TTS_ENABLED=true
      - QWEN_TTS_MODEL_SIZE=1.7B
      - QWEN_TTS_DEVICE=cuda:0
      - HF_HOME=/models/huggingface
    volumes:
      - qwen-models:/models/qwen-tts
      - huggingface-cache:/models/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  qwen-models:
  huggingface-cache:
```

**File: `Dockerfile`** (additions)

```dockerfile
# Add Qwen3-TTS dependencies
RUN pip install --no-cache-dir qwen-tts

# Install FlashAttention (requires CUDA)
RUN pip install flash-attn --no-build-isolation

# Pre-download models (optional, for faster startup)
ARG PRELOAD_QWEN_MODELS=false
RUN if [ "$PRELOAD_QWEN_MODELS" = "true" ]; then \
    python -c "from qwen_tts import Qwen3TTSModel; \
               Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"; \
    fi
```

---

### Phase 5: Frontend Integration (Week 4)

#### 3.5.1 Voice Selector Component

**File: `frontend/src/components/QwenVoiceSelector.tsx`**

```typescript
import React, { useState, useEffect } from 'react';

interface QwenVoice {
  id: string;
  name: string;
  description: string;
  native_language: string;
  gender: string;
  style_tags: string[];
}

interface QwenVoiceSelectorProps {
  onVoiceSelect: (voice: QwenVoice) => void;
  onModeChange: (mode: 'custom_voice' | 'voice_design' | 'voice_clone') => void;
  selectedMode: string;
}

export const QwenVoiceSelector: React.FC<QwenVoiceSelectorProps> = ({
  onVoiceSelect,
  onModeChange,
  selectedMode
}) => {
  const [voices, setVoices] = useState<QwenVoice[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/v1/qwen/speakers')
      .then(res => res.json())
      .then(data => {
        setVoices(data.speakers);
        setLoading(false);
      });
  }, []);

  return (
    <div className="qwen-voice-selector">
      {/* Mode Selection */}
      <div className="mode-tabs">
        <button 
          className={selectedMode === 'custom_voice' ? 'active' : ''}
          onClick={() => onModeChange('custom_voice')}
        >
          Preset Voices
        </button>
        <button 
          className={selectedMode === 'voice_design' ? 'active' : ''}
          onClick={() => onModeChange('voice_design')}
        >
          Design Voice
        </button>
        <button 
          className={selectedMode === 'voice_clone' ? 'active' : ''}
          onClick={() => onModeChange('voice_clone')}
        >
          Clone Voice
        </button>
      </div>

      {/* Voice Grid for Custom Voice Mode */}
      {selectedMode === 'custom_voice' && (
        <div className="voice-grid">
          {voices.map(voice => (
            <div 
              key={voice.id} 
              className="voice-card"
              onClick={() => onVoiceSelect(voice)}
            >
              <h4>{voice.name}</h4>
              <p>{voice.description}</p>
              <div className="tags">
                {voice.style_tags.map(tag => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Voice Design Input */}
      {selectedMode === 'voice_design' && (
        <div className="voice-design-input">
          <label>Describe your desired voice:</label>
          <textarea 
            placeholder="e.g., A warm, friendly female voice with a slight British accent, speaking at a moderate pace with gentle intonation..."
          />
        </div>
      )}

      {/* Voice Clone Upload */}
      {selectedMode === 'voice_clone' && (
        <div className="voice-clone-upload">
          <label>Upload reference audio (3+ seconds):</label>
          <input type="file" accept="audio/*" />
          <label>Reference text (optional but recommended):</label>
          <textarea placeholder="Transcript of the reference audio..." />
        </div>
      )}
    </div>
  );
};
```

---

### Phase 6: Testing (Week 4-5)

#### 3.6.1 Unit Tests

**File: `tests/test_qwen_tts_backend.py`**

```python
"""Unit tests for Qwen3-TTS backend."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from app.backends.qwen_tts import QwenTTSBackend
from app.backends.qwen_tts_config import QwenTTSConfig
from app.models.qwen_voices import QWEN_SPEAKERS, get_speaker_by_id


class TestQwenTTSConfig:
    def test_default_config(self):
        config = QwenTTSConfig()
        assert config.model_size == "1.7B"
        assert config.enable_custom_voice is True
        
    def test_config_validation(self):
        config = QwenTTSConfig(model_size="0.6B", enable_voice_design=True)
        with pytest.raises(ValueError):
            config.validate()
    
    def test_from_dict(self):
        config = QwenTTSConfig.from_dict({
            "model_size": "0.6B",
            "enable_voice_design": False
        })
        assert config.model_size == "0.6B"


class TestQwenVoices:
    def test_all_speakers_defined(self):
        assert len(QWEN_SPEAKERS) == 9
        
    def test_get_speaker_by_id(self):
        voice = get_speaker_by_id("vivian")
        assert voice is not None
        assert voice.name == "Vivian"
        
    def test_speaker_languages(self):
        for voice in QWEN_SPEAKERS.values():
            assert "Chinese" in voice.supported_languages
            assert "English" in voice.supported_languages


class TestQwenTTSBackend:
    @pytest.fixture
    def config(self):
        return QwenTTSConfig(
            enable_custom_voice=True,
            enable_voice_design=False,
            enable_voice_clone=False
        )
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.generate_custom_voice.return_value = (
            [np.zeros(16000, dtype=np.float32)], 
            16000
        )
        return model
    
    @pytest.mark.asyncio
    async def test_synthesize_custom_voice(self, config, mock_model):
        with patch('qwen_tts.Qwen3TTSModel.from_pretrained', return_value=mock_model):
            backend = QwenTTSBackend(config)
            await backend.initialize()
            
            result = await backend.synthesize(
                text="Hello world",
                voice="Vivian",
                language="English",
                mode="custom_voice"
            )
            
            assert result.sample_rate == 16000
            assert result.backend == "qwen_tts"
    
    @pytest.mark.asyncio
    async def test_invalid_speaker(self, config, mock_model):
        with patch('qwen_tts.Qwen3TTSModel.from_pretrained', return_value=mock_model):
            backend = QwenTTSBackend(config)
            await backend.initialize()
            
            with pytest.raises(ValueError, match="Unknown speaker"):
                await backend.synthesize(
                    text="Test",
                    voice="NonexistentSpeaker",
                    mode="custom_voice"
                )
```

#### 3.6.2 Integration Tests

**File: `tests/test_qwen_tts_integration.py`**

```python
"""Integration tests for Qwen3-TTS backend."""

import pytest
from fastapi.testclient import TestClient
import io

from app.main import app


@pytest.mark.integration
class TestQwenTTSAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_list_speakers(self, client):
        response = client.get("/api/v1/qwen/speakers")
        assert response.status_code == 200
        data = response.json()
        assert "speakers" in data
        assert len(data["speakers"]) > 0
    
    def test_list_languages(self, client):
        response = client.get("/api/v1/qwen/languages")
        assert response.status_code == 200
        data = response.json()
        assert "Chinese" in data["languages"]
        assert "English" in data["languages"]
    
    @pytest.mark.slow
    def test_synthesize_custom_voice(self, client):
        response = client.post(
            "/api/v1/qwen/synthesize",
            data={
                "text": "Hello, this is a test.",
                "mode": "custom_voice",
                "speaker": "Ryan",
                "language": "English"
            }
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
    
    @pytest.mark.slow
    def test_voice_design(self, client):
        response = client.post(
            "/api/v1/qwen/synthesize",
            data={
                "text": "Testing voice design feature.",
                "mode": "voice_design",
                "language": "English",
                "instruct": "A cheerful young female voice with enthusiasm"
            }
        )
        assert response.status_code == 200
```

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

The existing GLM-TTS backend remains fully functional. Qwen3-TTS is added as an alternative backend that can be selected via:

1. **API**: Use `/api/v1/qwen/*` endpoints
2. **Config**: Set `default_backend: qwen_tts` in config.yaml
3. **Per-request**: Include `backend: qwen_tts` in request body

### 4.2 Gradual Rollout

1. **Week 1-2**: Deploy Qwen3-TTS alongside GLM-TTS in staging
2. **Week 3**: Enable for internal testing with feature flag
3. **Week 4**: Public beta with opt-in toggle
4. **Week 5+**: Monitor metrics, gather feedback, iterate

---

## 5. Performance Considerations

### 5.1 Resource Requirements

| Model | VRAM | RAM | First Token Latency |
|-------|------|-----|---------------------|
| 1.7B CustomVoice | ~8GB | 16GB | ~200ms |
| 1.7B VoiceDesign | ~8GB | 16GB | ~200ms |
| 1.7B Base (Clone) | ~8GB | 16GB | ~200ms |
| 0.6B CustomVoice | ~4GB | 8GB | ~150ms |
| 0.6B Base (Clone) | ~4GB | 8GB | ~150ms |

### 5.2 Optimization Recommendations

1. **FlashAttention 2**: Enabled by default, reduces VRAM by ~40%
2. **BFloat16**: Use on Ampere+ GPUs for best speed/quality tradeoff
3. **Voice Prompt Caching**: Pre-compute and cache voice clone prompts
4. **Batch Processing**: Group requests when possible
5. **vLLM-Omni**: Consider for production high-throughput scenarios

---

## 6. Monitoring & Metrics

### 6.1 New Metrics to Track

```python
# Add to Prometheus metrics
qwen_tts_synthesis_duration_seconds = Histogram(
    'qwen_tts_synthesis_duration_seconds',
    'Time spent on synthesis',
    ['mode', 'language']
)

qwen_tts_audio_duration_seconds = Counter(
    'qwen_tts_audio_duration_seconds_total',
    'Total audio duration generated',
    ['mode']
)

qwen_tts_model_load_time_seconds = Gauge(
    'qwen_tts_model_load_time_seconds',
    'Time to load model',
    ['model_type']
)
```

### 6.2 Grafana Dashboard Additions

- Qwen3-TTS RTF (Real-Time Factor) panel
- Voice clone prompt cache hit rate
- Per-speaker usage distribution
- Language distribution pie chart

---

## 7. Documentation

### 7.1 API Documentation Updates

Add OpenAPI specs for new endpoints in `app/api/openapi.py`.

### 7.2 User Documentation

**File: `docs/qwen_tts_backend.md`**

Document:
- Available speakers and their characteristics
- Voice design prompt best practices
- Voice cloning requirements (audio quality, duration)
- Language support matrix
- Performance tuning guide

---

## 8. Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Core Backend | QwenTTSBackend, config, voice definitions |
| 2-3 | API Integration | Routes, engine factory updates |
| 3 | Streaming | Streaming handler, WebSocket support |
| 3-4 | Infrastructure | Config, Docker, requirements |
| 4 | Frontend | Voice selector component |
| 4-5 | Testing | Unit tests, integration tests |
| 5 | Documentation | API docs, user guides |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model download failures | Pre-download models in Docker build |
| VRAM constraints | Implement model offloading, provide 0.6B option |
| API breaking changes | Pin qwen-tts version, add integration tests |
| Performance regression | Benchmark against GLM-TTS, set alerts |

---

## 10. Future Enhancements

1. **vLLM-Omni Integration**: True streaming when online serving launches
2. **Fine-tuning Pipeline**: Custom voice training workflow
3. **Multi-GPU Support**: Tensor parallelism for larger batches
4. **Voice Library**: User-uploaded voice management system
5. **Emotion Presets**: Pre-configured emotion instructions

---

## 11. API Quick Start (curl Examples)

### Start the Qwen TTS Container
```bash
docker compose up qwen-tts  # Runs on port 8013
```

### Voice Clone with Reference Audio
Clone any voice from a 3+ second audio sample:
```bash
# Basic voice clone (x-vector mode - just captures voice timbre)
curl -X POST "http://localhost:8013/api/v1/qwen/clone?text=Your%20text%20here&language=English&use_xvector_only=true" \
  -F "ref_audio=@/path/to/reference.wav" \
  -o output.wav

# Example: Clone Batman voice
curl -X POST "http://localhost:8013/api/v1/qwen/clone?text=I%20am%20the%20night&language=English&use_xvector_only=true" \
  -F "ref_audio=@data/voices/batman/batman_01.wav" \
  -o batman_output.wav
```

### Built-in Speakers with Style Instructions
Use predefined speakers with emotional/style control:
```bash
curl -X POST "http://localhost:8013/api/v1/qwen/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I cannot believe you would do that!",
    "mode": "custom_voice",
    "speaker": "Vivian",
    "language": "English",
    "instruct": "Speak with an angry and frustrated tone"
  }' \
  -o angry_speech.wav
```

**Available Speakers:** `Chelsie`, `Ethan`, `Laura`, `Ryan`, `Aspen`, `Vivian`, `Harmony`, `Maverick`, `Echo`

### Voice Design (Natural Language Description)
Create a voice by describing it:
```bash
curl -X POST "http://localhost:8013/api/v1/qwen/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to our podcast.",
    "mode": "voice_design",
    "instruct": "A deep, authoritative male voice with a warm radio host quality",
    "language": "English"
  }' \
  -o podcast_host.wav
```

### Simple Synthesis (Default Speaker)
```bash
curl -X POST "http://localhost:8013/api/v1/qwen/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "speaker": "Ryan", "language": "English"}' \
  -o hello.wav
```

### Streaming Synthesis
```bash
curl -X POST "http://localhost:8013/api/v1/qwen/synthesize/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is streaming audio.", "speaker": "Vivian", "language": "English"}' \
  -o streaming.wav
```