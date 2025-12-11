# Implementation Plan: GLM-TTS Backend Integration

## Current Goal
Implement a new backend for GLM-TTS model while keeping the existing XTTS backend, with configurable backend selection.

## Current Architecture Analysis

### Existing XTTS Backend
- **Service File**: `app/xtts_service_v2.py`
- **Main Class**: `TTSService` - handles XTTS v2 model loading, voice management, and speech generation
- **Dependencies**: Uses `coqui-tts` library with `TTS.tts.models.xtts.Xtts`
- **API Integration**: `app/main.py` imports and uses `TTSService` directly

### Key Components to Abstract
1. Model initialization
2. Voice loading/management  
3. Speech generation (text -> audio)
4. Configuration handling

## GLM-TTS Model Analysis

### Model Source
- HuggingFace: https://huggingface.co/zai-org/GLM-TTS
- GitHub: https://github.com/zai-org/GLM-TTS

### Architecture
- **Two-stage design**:
  1. **Stage 1 (LLM)**: Llama-based model converts text to speech token sequences
  2. **Stage 2 (Flow Matching)**: Converts tokens to mel-spectrograms, then vocoder generates waveforms

### Key Features
- Zero-shot voice cloning (3-10 seconds of prompt audio)
- Streaming inference support
- Multi-language support (primarily Chinese, with English)
- RL-enhanced emotion control

### GLM-TTS Dependencies
- `transformers` - for LlamaForCausalLM, AutoTokenizer
- `torch`, `torchaudio`
- Custom modules: `cosyvoice`, `llm/glmtts.py`, `flow/`, `frontend/`

### GLM-TTS Inference Flow
1. Load frontends (TTSFrontEnd, TextFrontEnd, SpeechTokenizer)
2. Load LLM model (GLMTTS based on Llama architecture)
3. Load Flow model for token-to-wav conversion
4. Process text through frontend
5. Generate speech tokens via LLM
6. Convert tokens to audio via Flow + Vocoder

---

## Proposed Implementation Plan

### Phase 1: Create Abstract TTS Backend Interface

Create a base class that defines the common interface for all TTS backends:

```python
# app/tts_backend_base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

class TTSBackendBase(ABC):
    """Abstract base class for TTS backends"""
    
    @abstractmethod
    def initialize_model(self) -> None:
        """Initialize the TTS model"""
        pass
    
    @abstractmethod
    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """Load a voice from file(s)"""
        pass
    
    @abstractmethod
    def load_voices(self) -> None:
        """Load all voices from the voices directory"""
        pass
    
    @abstractmethod
    def get_voices(self) -> List[str]:
        """Get list of available voice names"""
        pass
    
    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text. Returns (audio_data, sample_rate)"""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier"""
        pass
```

### Phase 2: Refactor XTTS Backend

Modify `xtts_service_v2.py` to implement the abstract interface:

```python
# app/backends/xtts_backend.py
from app.tts_backend_base import TTSBackendBase

class XTTSBackend(TTSBackendBase):
    """XTTS v2 TTS Backend"""
    
    @property
    def backend_name(self) -> str:
        return "xtts"
    
    # ... implement all abstract methods
```

### Phase 3: Implement GLM-TTS Backend

Create a new backend for GLM-TTS:

```python
# app/backends/glm_tts_backend.py
from app.tts_backend_base import TTSBackendBase

class GLMTTSBackend(TTSBackendBase):
    """GLM-TTS Backend"""
    
    @property
    def backend_name(self) -> str:
        return "glm-tts"
    
    # ... implement all abstract methods using GLM-TTS inference code
```

### Phase 4: Create Backend Factory

```python
# app/tts_factory.py
from typing import Optional
from app.tts_backend_base import TTSBackendBase

class TTSBackendFactory:
    """Factory for creating TTS backend instances"""
    
    _backends = {}
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        cls._backends[name] = backend_class
    
    @classmethod
    def create_backend(cls, backend_name: str, **kwargs) -> TTSBackendBase:
        if backend_name not in cls._backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        return cls._backends[backend_name](**kwargs)
    
    @classmethod
    def available_backends(cls) -> list:
        return list(cls._backends.keys())
```

### Phase 5: Update Configuration

Add backend configuration to `app/config.py` and `config.yaml`:

```yaml
# config.yaml
tts_backend: "xtts"  # or "glm-tts"

# XTTS-specific config
xtts:
  model_name: tts_models--multilingual--multi-dataset--xtts_v2
  use_deepspeed: true
  tau: 0.3
  gpt_cond_len: 3
  top_k: 3
  top_p: 5

# GLM-TTS-specific config
glm_tts:
  model_path: ckpt/  # path to GLM-TTS checkpoint
  sample_rate: 24000
  use_phoneme: false
  beam_size: 1
  sampling: 25
```

### Phase 6: Update Docker Configuration

Create separate Dockerfiles or multi-stage builds for different backends:

```yaml
# docker-compose.yml - add environment variable for backend selection
services:
  tts-api:
    environment:
      - TTS_BACKEND=xtts  # or glm-tts
```

Consider:
- GLM-TTS has different dependencies (cosyvoice, etc.)
- May need larger GPU memory
- Different model download/caching requirements

### Phase 7: Update API Endpoints

Modify `app/main.py` to use the factory pattern:

```python
# In main.py
from app.tts_factory import TTSBackendFactory
from app.config import settings

# Initialize backend based on configuration
tts_service = TTSBackendFactory.create_backend(
    settings.tts_backend,
    logger=logger
)
```

---

## File Structure After Implementation

```
app/
├── __init__.py
├── config.py              # Updated with backend config
├── config.yaml            # Updated with backend options
├── main.py                # Updated to use factory
├── tts_backend_base.py    # NEW: Abstract base class
├── tts_factory.py         # NEW: Backend factory
├── backends/              # NEW: Backend implementations
│   ├── __init__.py
│   ├── xtts_backend.py    # Refactored from xtts_service_v2.py
│   └── glm_tts_backend.py # NEW: GLM-TTS implementation
├── glm_tts/               # NEW: GLM-TTS specific modules
│   ├── __init__.py
│   ├── frontend.py        # Adapted from GLM-TTS cosyvoice/cli/frontend.py
│   ├── flow.py            # Adapted from GLM-TTS flow/
│   └── llm.py             # Adapted from GLM-TTS llm/glmtts.py
├── models.py
├── monitoring.py
├── version.py
└── log_util.py
```

---

## GLM-TTS Specific Considerations

### Dependencies to Add
```
# requirements.txt additions for GLM-TTS
onnxruntime>=1.15.0  # For speaker embedding model (campplus.onnx)
vocos>=0.1.0         # For vocoder (optional, HiFi-GAN also supported)
```

### Model Files Required
GLM-TTS needs these model files in a checkpoint directory:
- `vq32k-phoneme-tokenizer/` - Tokenizer
- LLM weights (Llama-based)
- Flow model weights
- Vocoder weights (Vocos or HiFi-GAN)
- `frontend/campplus.onnx` - Speaker embedding model
- `frontend/spk2info.pt` - Speaker info

### Voice Cloning Differences
- **XTTS**: Uses reference audio directly for voice cloning
- **GLM-TTS**: Extracts speaker embeddings from prompt audio (3-10 seconds)

Both support zero-shot cloning, but the internal mechanism differs.

---

## Implementation Order

1. [x] Create `app/tts_backend_base.py` with abstract interface
2. [x] Create `app/backends/` directory structure
3. [x] Refactor `xtts_service_v2.py` -> `app/backends/xtts_backend.py`
4. [x] Create `app/tts_factory.py`
5. [x] Update `app/config.py` with backend configuration
6. [x] Update `app/main.py` to use factory pattern
7. [x] Test XTTS backend still works after refactoring - PASSED
8. [x] Implement GLM-TTS backend:
   - [x] Create `glm_tts_backend.py` skeleton
   - [x] Add GLM-TTS dependencies to requirements.txt
   - [ ] Port necessary GLM-TTS modules (cosyvoice, flow, llm)
9. [ ] Update Dockerfile for GLM-TTS support
10. [x] Update docker-compose.yml with backend configuration
11. [ ] Test GLM-TTS backend (requires model download)
12. [ ] Documentation update

## Testing Results (2025-12-10)

### XTTS Backend - WORKING
- Health endpoint: OK
- Voices endpoint: OK (7 voices loaded)
- TTS generation: OK (generated 441KB audio file)
- GPU acceleration: OK (RTX 5090 detected, 1.8GB VRAM used)

### GLM-TTS Backend - WORKING
- Health endpoint: OK
- Voices endpoint: OK (10 voices loaded)
- English TTS: OK (generated 230KB audio file)
- Chinese TTS: OK (generated 1.2MB audio file)
- GPU acceleration: OK (RTX 5090, 8.4GB VRAM used)
- Model download: `python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-TTS', local_dir='GLM-TTS/ckpt')"`

---

## Potential Issues

1. **Memory Requirements**: GLM-TTS may need more GPU memory than XTTS
2. **Dependency Conflicts**: GLM-TTS uses specific versions of transformers/torch
3. **Model Size**: GLM-TTS checkpoint is likely larger than XTTS
4. **Language Support**: GLM-TTS primarily supports Chinese with English mixing; XTTS has broader multi-language support
5. **Inference Speed**: May differ significantly between backends

---

## Questions to Resolve

1. Should we support running both backends simultaneously (for A/B testing)?
2. Should backend switching require container restart or be dynamic?
3. How to handle voice format differences between backends?
4. Should we maintain separate voice directories per backend?

