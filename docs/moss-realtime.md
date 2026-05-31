# MOSS-TTS-Realtime (standalone)

Low-latency token-level streaming via [MOSS-TTS-Realtime](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_realtime) (~1.7B Qwen3 backbone + 200M depth transformer). Separate from **openmoss** (GGML v1.5 clone on port 8012).

## When to use which backend

| Backend | Port | Best for |
|---------|------|----------|
| **openmoss** | 8012 | Fast voice clone quality (MOSS-TTS v1.5 Q8 GGML) |
| **moss-realtime** | 8016 | Low TTFA, token streaming, conversational TTS |
| **moss-tts** | 8013 | Voice design + long-form v1.5 PyTorch |

## Quick start (Docker)

```bash
docker compose --env-file .env.moss-realtime up -d moss-realtime
curl -s http://localhost:8016/health | jq .
```

First launch downloads `OpenMOSS-Team/MOSS-TTS-Realtime` and `OpenMOSS-Team/MOSS-Audio-Tokenizer` (~3–5 min). Startup runs a warmup pass; **the first request after a cold container start** may still hit CUDA JIT once.

## Bare metal

```bash
MOSS_RT_GPU=3 ./scripts/start-moss-realtime.sh
```

Requires PyTorch + MOSS-TTS installed (see `Dockerfile.moss`). Docker is the supported path.

## GPU sizing

Default: **GPU 3** (`NVIDIA_VISIBLE_DEVICES=3` in compose) so it can run alongside openmoss on GPU 1.

| Component | VRAM (approx) |
|-----------|---------------|
| MOSS-TTS-Realtime backbone (bf16) | ~3.4 GB |
| MOSS-Audio-Tokenizer codec | ~3 GB |
| **Total** | ~5–8 GB |

Uses bf16 (`MOSS_QUANTIZE=none`) — no 4-bit quant on the realtime path.

## API

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | `model_id`: `moss-tts-realtime`, `streaming_mode`: `realtime` |
| GET | `/voices` | Reference voices in `data/voices/{name}/` |
| POST | `/tts` | Full utterance (realtime model, non-streaming) |
| POST | `/tts/stream` | Token-level streaming (binary framing) |

### Voice clone

Place a reference WAV under `data/voices/{voice_name}/`, then:

```bash
curl -s -N -X POST http://localhost:8016/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from MOSS realtime.","voice_name":"YOUR_VOICE","language":"en"}' \
  --output /tmp/rt_stream.bin
```

Second request with the same voice hits the in-memory prompt cache (faster encode).

### Streaming binary framing

Same as GLM-TTS / moss-tts:

```
[4B audio_len LE][4B meta_len LE][wav_bytes][metadata_json] ...
```

Final chunk has `audio_len=0` and `"is_final": true` in metadata.

## Environment variables

| Variable | Default (moss-realtime) | Purpose |
|----------|-------------------------|---------|
| `MOSS_ENABLE_MAIN_MODEL` | `false` | Skip MOSS-TTS v1.5 8B |
| `MOSS_ENABLE_VOICE_GEN` | `false` | Skip VoiceGenerator |
| `MOSS_ENABLE_REALTIME` | `true` | Load MOSS-TTS-Realtime |
| `MOSS_ENABLE_STREAMING` | `true` | Enable `/tts/stream` |
| `MOSS_RT_MODEL_ID` | `OpenMOSS-Team/MOSS-TTS-Realtime` | Realtime weights (HF hub or merged local path) |
| `MOSS_RT_NATIVE_VOICE` | `false` | Skip reference WAV (set `true` for merged native-voice checkpoints) |
| `MOSS_RT_STREAM_CODEC_BACKEND` | `torch` | Stateful codec for `/tts/stream` quality |
| `MOSS_RT_STREAM_DECODER_OVERLAP_FRAMES` | `0` | Crossfade off (required with torch stream) |
| `MOSS_RT_CODEC_ID` | `OpenMOSS-Team/MOSS-Audio-Tokenizer` | PyTorch codec (if backend=torch) |
| `MOSS_RT_CODEC_BACKEND` | `auto` | `auto` → ONNX if weights on disk, else torch |
| `MOSS_RT_ONNX_CODEC_DIR` | `training/weights/MOSS-Audio-Tokenizer-ONNX` | ONNX encoder/decoder |
| `MOSS_RT_DOWNLOAD_ONNX` | unset | Set `1` to auto-download ONNX codec on start if missing |
| `MOSS_RT_ONNX_GPU` | `true` | ONNX Runtime on GPU (needs cuDNN 9 for ORT CUDA EP) |
| `MOSS_RT_DEVICES` | `0` | CUDA index inside container |
| `MOSS_QUANTIZE` | `none` | bf16 full precision |
| `CUDA_CACHE_MAXSIZE` | `4294967296` | 4 GiB PTX JIT cache |

## Smoke test

```bash
python3 scripts/test_moss_stream.py --api-url http://localhost:8016
```

Target after warm (merged weights, RTX 3090 Ti): stream sustained **~2×**, TTFA **~1.15s**, batch `/tts` **~1.7×**.

## Compose profile

```bash
# .env.moss-realtime
COMPOSE_PROFILES=moss-realtime
```

Does not start moss-tts, frontend, or other profiles — only `moss-realtime`.

## Native-voice finetuned weights (optional)

Point `MOSS_RT_MODEL_ID` at a local merged checkpoint and set `MOSS_RT_NATIVE_VOICE=true`. Finetune/merge scripts live under `training/` locally (gitignored), not in this repo.

For faster batch `/tts`, enable ONNX codec:

```bash
MOSS_RT_DOWNLOAD_ONNX=1 MOSS_RT_CODEC_BACKEND=auto ./scripts/start-moss-realtime.sh
```
