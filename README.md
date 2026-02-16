# Speaker: Multi-Engine Generative Speech Platform

Speaker is a production-grade platform for ultra-realistic Text-to-Speech synthesis, zero-shot voice cloning, and AI voice design. It supports multiple TTS backends that can be switched with a single command, and ships with a polished React frontend for voice creation and management.

<!-- TODO: Screenshot of the main TTS Workspace page -->
![TTS Workspace](docs/screenshots/tts_workspace.png)

---

## Features

### 🎙️ Text-to-Speech Generation
Generate speech from text using any of the supported backends. Supports long-form synthesis, configurable parameters, and streaming output.

<!-- TODO: GIF showing text-to-speech generation workflow (type text → select voice → generate → play audio) -->
![TTS Generation Demo](docs/screenshots/tts_generation.gif)

### 🎨 Voice Studio — Design Voices from Text
Create entirely new voices from natural language descriptions — no reference audio needed. Powered by MOSS-VoiceGenerator.

> *"A warm, friendly female voice with a moderate pace"*
> *"A deep, authoritative male voice like a news anchor"*

<!-- TODO: GIF showing Voice Studio Design tab (type voice description → type text → generate → play result) -->
![Voice Design Demo](docs/screenshots/voice_design.gif)

### 🧬 Zero-Shot Voice Cloning
Clone any voice with as little as 3 seconds of reference audio. Upload a sample, and the system captures the voice's identity for synthesis.

<!-- TODO: Screenshot of the Clone tab with audio upload and generation -->
![Voice Cloning](docs/screenshots/voice_cloning.png)

### 📚 Voice Library
Manage saved voices in a visual library. Upload reference audio, test voices, and organize your collection.

<!-- TODO: Screenshot of Voice Library page showing saved voices grid -->
![Voice Library](docs/screenshots/voice_library.png)

---

## Architecture

Speaker supports three TTS backends, selectable via a single config change:

| Backend | Model | Key Strength | Port |
|---------|-------|-------------|------|
| **MOSS-TTS** | `OpenMOSS-Team/MOSS-TTS` | Voice design from text, high quality | 8013 |
| **GLM-TTS** | GLM-4-Voice | vLLM + TensorRT acceleration | 8012 |
| **Qwen3-TTS** | `Qwen/Qwen3-TTS` | Multilingual, style control | 8016 |

<!-- TODO: Screenshot/diagram of the architecture or the Settings/sidebar showing "MOSS-TTS Active" -->
![Architecture](docs/screenshots/architecture.png)

### MOSS-TTS Backend
The recommended backend, featuring:
- **MOSS-TTS** (8B) — High-quality TTS with voice cloning
- **MOSS-VoiceGenerator** (8B) — Voice design from text descriptions (no reference audio)
- 4-bit NF4 quantization, Flash Attention 2
- Both models run simultaneously on separate GPUs (~14 GB + ~9 GB VRAM)

### GLM-TTS Backend
The original backend, optimized for Blackwell GPUs:
- vLLM engine with PagedAttention for the LLM stage
- TensorRT-accelerated HiFT vocoder (3x speedup)
- Configurable flow-matching steps for speed/quality tradeoff

### Qwen3-TTS Backend
Multilingual backend with style control:
- Built-in speaker library with instruction-following
- Cross-lingual voice cloning
- X-vector voice extraction

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU with 16+ GB VRAM (24+ GB recommended for dual-model MOSS setup)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

### Deploy with MOSS-TTS (Recommended)

```bash
# Clone the repo
git clone https://github.com/aleckarfonta/speaker.git
cd speaker

# Start MOSS-TTS backend + frontend
docker compose --profile moss up -d moss-tts frontend

# Wait for models to load (~2-3 minutes), then open:
# Frontend:  http://localhost:3012
# API:       http://localhost:8013
```

### Deploy with GLM-TTS

```bash
# GLM-TTS requires model weights in GLM-TTS/ckpt/
docker compose up -d tts-api frontend
```

---

## Switching Backends

Switch between backends with a single command:

```bash
# Switch to MOSS-TTS
./scripts/switch-backend.sh moss

# Switch to GLM-TTS
./scripts/switch-backend.sh glm

# Switch to Qwen3-TTS
./scripts/switch-backend.sh qwen

# Check current backend
./scripts/switch-backend.sh status
```

This updates the frontend proxy target and restarts the container. No rebuild required.

You can also switch manually via `docker-compose.yml`:
```yaml
frontend:
  environment:
    - TTS_BACKEND_HOST=moss-tts:8000   # or tts-api:8000, qwen-tts:8000
```

---

## API Reference

All backends expose a unified REST API:

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + GPU info |
| `GET` | `/voices` | List available voices |
| `POST` | `/tts` | Generate speech (auto-clones if `voice_name` provided) |
| `POST` | `/tts/stream` | Streaming TTS generation |
| `POST` | `/tts/clone` | Clone voice from uploaded audio |
| `POST` | `/tts/design` | Voice design from text description (MOSS only) |
| `POST` | `/voices/{name}` | Upload reference audio for a voice |

### Voice Studio Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/qwen/synthesize` | Unified synth (routes by `mode`) |
| `POST` | `/api/v1/qwen/clone` | Clone from uploaded audio |
| `GET` | `/api/v1/qwen/speakers` | List available speakers |

### Example: Generate Speech
```bash
curl -X POST http://localhost:8013/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice_name": "trump"}' \
  --output speech.wav
```

### Example: Design a Voice
```bash
curl -X POST http://localhost:8013/tts/design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to our tavern!",
    "instruction": "Hearty, jovial tavern owner voice, loud and welcoming"
  }' --output designed_voice.wav
```

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3012 | React UI |
| MOSS-TTS API | 8013 | MOSS backend |
| GLM-TTS API | 8012 | GLM backend |
| Qwen3-TTS API | 8016 | Qwen backend |
| Grafana | 3333 | Performance monitoring |
| Prometheus | 9199 | Metrics collection |

---

## Performance Telemetry

Speaker includes a Prometheus + Grafana monitoring stack for production observability:

- **Real-Time Factor (RTF)** — ratio of audio duration to inference time
- **Per-Stage Latency** — LLM, Flow Matching, Vocoder breakdown
- **VRAM Tracking** — real-time GPU memory usage

<!-- TODO: Screenshot of Grafana dashboard showing performance metrics -->
![Grafana Dashboard](docs/screenshots/grafana_dashboard.png)

---

## Validation

Automated quality validation via round-trip STT transcription analysis:

```bash
python3 scripts/run_tts_validation.py --url http://localhost:8013
```

The MOSS-TTS test suite covers all endpoints:

```bash
python3 scripts/test_moss_api.py --url http://localhost:8013
# Tests: health, voices, tts, stream, clone, voice_design, tts_voice_name (9/9)
```

---

## Project Structure

```
speaker/
├── app/                    # Backend API implementations
│   ├── moss_api.py         #   MOSS-TTS + VoiceGenerator service
│   ├── api/                #   GLM-TTS backend
│   └── qwen/               #   Qwen3-TTS backend
├── frontend/               # React frontend (Vite + TypeScript)
│   ├── src/components/
│   │   ├── tts/            #   TTS Workspace (main generation page)
│   │   ├── studio/         #   Voice Studio (design/clone/speakers)
│   │   ├── voices/         #   Voice Library (management)
│   │   └── layout/         #   Sidebar + Layout
│   ├── nginx.conf.template #   Templated proxy config (envsubst)
│   └── Dockerfile          #   Multi-stage build
├── scripts/
│   ├── switch-backend.sh   #   One-command backend switching
│   ├── test_moss_api.py    #   MOSS-TTS test suite (9 tests)
│   └── run_tts_validation.py # Quality validation
├── k8s/                    # Kubernetes + monitoring configs
├── docker-compose.yml      # Full stack orchestration
└── Dockerfile.moss         # MOSS-TTS container
```

---

## Environment Variables

### MOSS-TTS Backend
| Variable | Default | Description |
|----------|---------|-------------|
| `MOSS_MODEL_ID` | `OpenMOSS-Team/MOSS-TTS` | TTS model to load |
| `MOSS_VOICE_GEN_MODEL` | `OpenMOSS-Team/MOSS-VoiceGenerator` | VoiceGenerator model |
| `MOSS_ENABLE_VOICE_GEN` | `true` | Enable voice design |
| `MOSS_QUANTIZE` | `4bit` | Quantization: `4bit`, `8bit`, `none` |
| `VOICES_DIR` | `/app/data/voices` | Voice reference storage |

### GLM-TTS Backend
| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_BACKEND` | `glm-tts` | Backend engine |
| `GLM_TTS_ENGINE` | `vllm` | LLM engine: `vllm` or `transformers` |
| `GLM_TTS_QUANTIZATION` | `4bit` | LLM quantization |
| `GLM_TTS_FLOW_STEPS` | `13` | Flow matching steps (lower = faster) |

### Frontend
| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_BACKEND_HOST` | `moss-tts:8000` | TTS backend for nginx proxy |

---

## License

This project is licensed under the MIT License.
