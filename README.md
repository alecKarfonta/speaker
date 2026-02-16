# Speaker

A multi-engine platform for text-to-speech, voice cloning, and voice design. Supports three TTS backends that can be swapped with a single command. Ships with a React frontend.

<!-- TODO: Screenshot of the main TTS generation page -->
![TTS Workspace](docs/screenshots/tts_workspace.png)

## What It Does

**Text-to-Speech** — Generate speech from text with configurable parameters, streaming output, and voice selection.

<!-- TODO: GIF — type text, pick a voice, generate, play the result -->
![TTS Demo](docs/screenshots/tts_generation.gif)

**Voice Design** — Describe a voice in plain English and the system creates it. No reference audio needed. Powered by [MOSS-VoiceGenerator](https://github.com/OpenMOSS/MOSS-TTS).

> *"A deep, authoritative male voice like a news anchor"*

<!-- TODO: GIF — Voice Studio Design tab, type a description, generate, listen -->
![Voice Design](docs/screenshots/voice_design.gif)

**Voice Cloning** — Clone any voice from as little as 3 seconds of reference audio. Upload a sample, and the system captures the identity for synthesis.

<!-- TODO: Screenshot of the Clone tab with an uploaded audio file -->
![Voice Cloning](docs/screenshots/voice_cloning.png)

**Voice Library** — Save, organize, and preview cloned voices.

<!-- TODO: Screenshot of the Voice Library page -->
![Voice Library](docs/screenshots/voice_library.png)

---

## Backends

| Backend | Model | Strength | Port |
|---------|-------|----------|------|
| **MOSS-TTS** | OpenMOSS-Team/MOSS-TTS | Voice design from text, high quality | 8013 |
| **GLM-TTS** | GLM-4-Voice | vLLM + TensorRT acceleration | 8012 |
| **Qwen3-TTS** | Qwen/Qwen3-TTS | Multilingual, style control | 8016 |

**MOSS-TTS** is the recommended default. It runs two models — MOSS-TTS for synthesis/cloning and MOSS-VoiceGenerator for voice design — on separate GPUs with 4-bit quantization. Total VRAM: ~23 GB across two cards.

**GLM-TTS** is the original backend, optimized for NVIDIA Blackwell hardware. Uses vLLM with PagedAttention for the LLM stage and a TensorRT HiFT vocoder.

**Qwen3-TTS** adds multilingual support and instruction-following style control with a built-in speaker library.

---

## Getting Started

```bash
git clone https://github.com/alecKarfonta/speaker.git
cd speaker

# Start MOSS-TTS + frontend
docker compose --profile moss up -d moss-tts frontend
```

Models download on first launch (~2-3 min). Then open:

- **Frontend**: http://localhost:3012
- **API**: http://localhost:8013/health

For GLM-TTS instead, place weights in `GLM-TTS/ckpt/` and run `docker compose up -d tts-api frontend`.

### Switching Backends

```bash
./scripts/switch-backend.sh moss     # MOSS-TTS
./scripts/switch-backend.sh glm      # GLM-TTS
./scripts/switch-backend.sh qwen     # Qwen3-TTS
./scripts/switch-backend.sh status   # Show current config
```

Or set `TTS_BACKEND_HOST` in `docker-compose.yml` under the `frontend` service:

```yaml
- TTS_BACKEND_HOST=moss-tts:8000   # or tts-api:8000, qwen-tts:8000
```

---

## API

All backends share a common REST API. The frontend hits these through an nginx reverse proxy.

```bash
# Generate speech
curl -X POST http://localhost:8013/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice_name": "trump"}' \
  -o speech.wav

# Design a voice from a description (MOSS only)
curl -X POST http://localhost:8013/tts/design \
  -H "Content-Type: application/json" \
  -d '{"text": "Welcome!", "instruction": "Warm female narrator"}' \
  -o designed.wav

# Clone a voice from audio
curl -X POST http://localhost:8013/tts/clone \
  -F "reference=@sample.wav" \
  -F "text=Testing voice clone" \
  -o cloned.wav
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check, GPU info |
| `GET /voices` | List saved voices |
| `POST /tts` | Generate speech (auto-clones if `voice_name` set) |
| `POST /tts/stream` | Streaming generation |
| `POST /tts/clone` | Clone from uploaded audio |
| `POST /tts/design` | Voice design from text prompt |
| `POST /voices/{name}` | Save reference audio |

---

## Monitoring

Prometheus + Grafana stack for tracking inference performance:

- Real-Time Factor (audio duration / inference time)
- Per-stage latency (LLM, flow matching, vocoder)
- GPU memory usage

<!-- TODO: Screenshot of Grafana dashboard -->
![Grafana](docs/screenshots/grafana_dashboard.png)

Grafana runs at http://localhost:3333.

---

## Project Layout

```
app/                     Backend APIs (MOSS, GLM, Qwen)
frontend/                React + TypeScript UI
  src/components/
    tts/                 TTS Workspace (main page)
    studio/              Voice Studio (design, clone, speakers)
    voices/              Voice Library
scripts/
  switch-backend.sh      Backend switching
  test_moss_api.py       API test suite (9 tests)
  run_tts_validation.py  Round-trip STT quality checks
docker-compose.yml       Full stack orchestration
```

## Configuration

Key environment variables (set in `docker-compose.yml`):

| Variable | Default | What it does |
|----------|---------|--------------|
| `TTS_BACKEND_HOST` | `moss-tts:8000` | Frontend proxy target |
| `MOSS_MODEL_ID` | `OpenMOSS-Team/MOSS-TTS` | TTS model |
| `MOSS_VOICE_GEN_MODEL` | `OpenMOSS-Team/MOSS-VoiceGenerator` | Voice design model |
| `MOSS_ENABLE_VOICE_GEN` | `true` | Enable voice design endpoint |
| `MOSS_QUANTIZE` | `4bit` | Quantization: `4bit`, `8bit`, `none` |
| `GLM_TTS_ENGINE` | `vllm` | GLM LLM engine |
| `GLM_TTS_QUANTIZATION` | `4bit` | GLM quantization |

---

## Service Ports

| Service | Port |
|---------|------|
| Frontend | 3012 |
| MOSS-TTS | 8013 |
| GLM-TTS | 8012 |
| Qwen3-TTS | 8016 |
| Grafana | 3333 |
| Prometheus | 9199 |

---

## License

MIT
