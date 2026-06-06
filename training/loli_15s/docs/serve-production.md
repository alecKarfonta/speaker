# loli15s epoch-7 production serve (TTFA-first)

Native-voice MOSS-TTS-Realtime finetuned on **8,902** QC-clean single-turn rows (~18.9h, 2,101 emotion).  
**Do not use openmoss GGUF for inference** — that stack is teacher-gen only.

## Recommended production config

```bash
export MOSS_RT_GPU=0                    # dedicated GPU; avoid sharing
export MOSS_RT_PORT=8016
export MOSS_RT_MODEL_ID=/path/to/speaker/training/loli_15s/exports/loli15s-epoch7-merged
export MOSS_RT_NATIVE_VOICE=true
# Decode defaults (warm_092_072 — omit on /tts/stream to use these)
export MOSS_RT_AUDIO_TEMPERATURE=0.92
export MOSS_RT_AUDIO_TOP_P=0.72
export MOSS_RT_AUDIO_TOP_K=40
export MOSS_RT_AUDIO_REPETITION_PENALTY=1.05
./scripts/start-moss-realtime.sh
```

Production v2 voice: `exports/loli15s-v2-merged` (8-epoch full SFT on merged 8.9k corpus).

## Audio sampling (warm_092)

When the client does not pass `audio_temperature` / `audio_top_p` / `audio_top_k`, the server uses:

| Param | Value |
|-------|------:|
| `audio_temperature` | 0.92 |
| `audio_top_p` | 0.72 |
| `audio_top_k` | 40 |
| `audio_repetition_penalty` | 1.05 |

These are set in `scripts/start-moss-realtime.sh` and [`app/moss_api.py`](../../app/moss_api.py). Lower temps (0.8/0.6) sound flatter on the LoRA; avoid unless A/B testing.

Merged weights are exported from `checkpoint-epoch-7`:

```bash
python training/loli_15s/scripts/legacy/finetune/merge_moss_rt_lora.py \
  --checkpoint training/loli_15s/output/sft_ddp_single/checkpoint-epoch-7 \
  --output training/loli_15s/exports/loli15s-epoch7-merged
```

## TTFA tuning (baked into start-moss-realtime.sh)

| Env | Value | Notes |
|-----|-------|-------|
| `MOSS_RT_INITIAL_TEXT_CHUNK` | 1 | Prime delay tokens |
| `MOSS_RT_STEADY_TEXT_CHUNK` | 4 | Smaller = faster first audio |
| `MOSS_RT_MIN_SAMPLES_FIRST_MS` | 40 | First buffer |
| `MOSS_RT_MIN_SAMPLES_STEADY_MS` | 120 | Steady buffer |
| `MOSS_RT_DECODER_CHUNK_FRAMES` | 6 | Decoder batch size |

Measured TTFA on `/tts/stream` (warm server, RTX 3090 Ti class GPU):

| Config | Short prompt | Medium prompt |
|--------|--------------|---------------|
| Old defaults (steady=24, dec=24) | ~1214 ms | ~1229 ms |
| **ttfa_fast (current)** | **~368–380 ms** | **~366–380 ms** |

Benchmark artifacts: `training/loli_15s/eval/bench/ttfa_baseline_epoch7.json`, `ttfa_merged_epoch7.json`.

Per-request overrides (same fields as env) are supported on `POST /tts/stream` JSON body.

## Warmup

First boot loads ~4.6 GB merged weights + codec (~4–5 min cold). Production should:

1. Start server and poll until ready:
   ```bash
   curl -sf http://127.0.0.1:8016/health | jq .
   ```
2. Optional smoke request:
   ```bash
   python3 scripts/test_moss_stream.py --api-url http://127.0.0.1:8016
   ```
3. Keep process running — do not scale to zero between conversational turns.

## API

- **Streaming (conversational):** `POST /tts/stream` — use this for TTFA.
- **Batch (full WAV):** `POST /tts` — faster sustained RTF (~1.7×) but no audio until complete; not for TTFA.

Streaming requires PyTorch codec (`MOSS_RT_STREAM_CODEC_BACKEND=torch`). ONNX codec is batch-only.

## Quality check

Listen: `training/loli_15s/eval/listen/epoch7_variety/`  
A/B merged vs LoRA (same text): `ab_merged_02_question.wav` vs `ab_finetuned_02_question.wav`.

## Not recommended

| Option | Why |
|--------|-----|
| LoRA at serve (`checkpoint-epoch-7` adapter dir) | Works but slower boot + forward vs merged |
| `MOSS_RT_EXPERIMENTAL_COMPILE_BACKBONE=true` | **Fails** on Realtime streaming (torch.compile + StaticCache); see `eval/bench/compile_attempt_epoch7.json` |
| openmoss GGUF | Different model; not your finetune |
| ONNX codec on `/tts/stream` | Breaks streaming quality |

## Re-benchmark

```bash
python training/moss-realtime/scripts/legacy/bench/sweep_rt_chunk_sizes.py \
  --api-url http://127.0.0.1:8016

MOSS_RT_GPU=0 SKIP_LORA=1 SKIP_COMPILE=1 \
  training/moss-realtime/scripts/legacy/bench/benchmark_rt_merged_compile.sh
```

Generate varied eval clips:

```bash
MOSS_RT_API=http://127.0.0.1:8016 \
  python training/loli_15s/scripts/generate_eval_samples.py
```

## v2 streaming cutoff audit

After serving epoch-7 or v2 merged weights with warm_092:

```bash
MOSS_RT_API=http://127.0.0.1:8016 \
  python training/loli_15s/scripts/verify_streaming_stack.py

# A/B epoch-7 vs v2 (second server on another port)
COMPARE_API=http://127.0.0.1:8017 \
  MOSS_RT_MODEL_ID=training/loli_15s/exports/loli15s-v2-merged \
  training/loli_15s/scripts/run_eval_v2.sh
```

Reports land in `training/loli_15s/eval/bench/verify_streaming_*.json` with `last_word_match_rate` and `likely_cutoff` flags.
