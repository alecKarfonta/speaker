# Cutting MOSS-TTS-Realtime TTFA from 2.5 seconds to under 500ms

Your **2.5-second time-to-first-audio is driven by at least four compounding bottlenecks**, most of which are fixable without changing the model itself. The biggest culprits are torch.compile cold-start overhead on an SM 120 GPU that lacks native SASS kernels, an oversized codec decode chunk, a conservative 300ms audio buffer, and likely PTX JIT compilation happening silently on every process restart. Implementing the six priority fixes below should bring TTFA into the 300–500ms range on your RTX 5070 Ti.

One critical correction first: **the RTX 5070 Ti is SM 120 (compute capability 12.0), not SM 100.** SM 100 refers to data-center Blackwell (B100/B200). This distinction matters because PyTorch's stable wheels ship native SASS only through SM 90 — your GPU is falling back to PTX JIT compilation for every CUDA kernel, adding massive first-call overhead.

## Where 2.5 seconds actually goes

The TTFA decomposes across a pipeline of sequential steps, each contributing latency before the client hears anything. Based on the MOSS-TTS-Realtime architecture (1.7B backbone + 200M local transformer, 12.5Hz audio token rate, 24kHz output via a 1.6B-parameter causal transformer codec), the breakdown looks approximately like this:

| Pipeline stage | Estimated cost | Root cause |
|---|---|---|
| PTX JIT kernel compilation (first request after restart) | 500–2,000ms | No native SM 120 SASS in PyTorch wheels |
| torch.compile tracing + Inductor codegen (first request) | 500–2,000ms | Cold compilation of backbone + local transformer graphs |
| Prefill (reference audio + text tokens through backbone) | 200–400ms | KV cache computation for prompt context |
| Autoregressive decode until enough tokens for first chunk | 150–300ms | Generating ~4 audio tokens at 12.5Hz before buffer threshold |
| MIN_SAMPLES buffer accumulation (0.3s) | ~300ms | Hardcoded 7,200-sample buffer before first yield |
| Codec decode of first chunk | 100–300ms | 1.6B-param MOSS-Audio-Tokenizer decode pass |
| WAV encoding + HTTP chunked transfer | 20–50ms | In-memory WAV + FastAPI StreamingResponse |

The first two rows — PTX JIT and torch.compile cold-start — only hit the very first request after process startup but account for **1–4 seconds** of your 2.5s TTFA. Subsequent requests likely already run faster. The remaining ~500–1,000ms comes from the prefill/decode/buffer/codec pipeline, which hits every request.

## Priority 1: Eliminate torch.compile and PTX cold-start overhead

The official MOSS-TTS-Realtime code **does not use torch.compile or StaticCache at all**. The official examples use standard HuggingFace inference with `MossTTSRealtimeInference` and the `push_text()` streaming API. If you added torch.compile yourself, it is likely the single largest contributor to your 2.5s TTFA on first request.

**Option A — Remove torch.compile entirely.** The official implementation achieves real-time streaming without it. Try running without compilation first to establish a baseline TTFA. The 30–50% decode speedup from `reduce-overhead` mode is valuable for throughput but catastrophic for cold-start latency.

**Option B — Keep torch.compile but pre-warm aggressively at startup.** If you need the throughput benefit, you must warm up before serving any requests:

```python
# At server startup, before accepting traffic:
with torch.inference_mode():
    dummy_session = create_streaming_session(model, reference_audio)
    for chunk in dummy_session.push_text("Warmup sentence for compilation."):
        pass  # Triggers all compilation paths
    dummy_session.end_text()
    dummy_session.drain()
torch.cuda.synchronize()
```

Additionally, enable persistent compilation caching so subsequent process restarts are fast:

```bash
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="/persistent/ssd/torch_compile_cache"
export TRITON_CACHE_DIR="/persistent/ssd/triton_cache"
```

For PTX JIT caching specifically, increase the CUDA compute cache size and point it at fast storage:

```bash
export CUDA_CACHE_PATH="/local/ssd/.nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=4294967296  # 4 GiB (default is only 256 MiB)
```

After the first successful warm-up, the PTX JIT cache persists across restarts (until you upgrade the NVIDIA driver). The torch.compile cache persists across restarts with the same PyTorch version and model.

## Priority 2: Fix the codec chunk size — the silent 300ms+ bottleneck

The official non-streaming examples use `chunk_duration=8` (8 seconds), which is designed for batch decode, not streaming. For streaming decode, the MOSS-Audio-Tokenizer supports chunks as small as **0.08 seconds** (1,920 samples at 24kHz — the minimum, constrained by the codec's downsample rate of 1,920). The user's `chunk_duration=0.24` is better than 8s but still conservative.

The critical architectural detail: MOSS-Audio-Tokenizer is a **1.6-billion-parameter causal transformer**. Each decode call has non-trivial overhead regardless of chunk size. Two changes matter here:

First, use the `codec.streaming(batch_size=1)` context manager that the official streaming examples use. This enables incremental stateful decoding rather than independent chunk decodes:

```python
with codec.streaming(batch_size=1):
    # Each decode call now carries forward internal state
    for audio_tokens in token_stream:
        decoded = codec.decode(audio_tokens, chunk_duration=0.08)
        yield decoded["audio"][0]
```

Second, consider switching to the **ONNX tokenizer** (`OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX`) for the decode path. The ONNX runtime eliminates Python overhead and enables graph-level optimizations. With TensorRT acceleration, small-chunk decode latency can drop from hundreds of milliseconds to single-digit milliseconds. The MOSS team provides ready-made configs including a `trt.yaml` for TensorRT-accelerated audio decode.

## Priority 3: Slash the audio buffer from 300ms to 80ms

Your `MIN_SAMPLES = int(0.3 * 24000)` adds **300ms of pure waiting** to every request's TTFA. This is far more conservative than necessary. At 24kHz with the codec's 12.5Hz frame rate, one complete codec frame produces **1,920 samples (80ms of audio)**. That is the atomic minimum unit the codec can decode.

```python
# Replace:
MIN_SAMPLES = int(0.3 * 24000)   # 7,200 samples = 300ms

# With adaptive buffering:
MIN_SAMPLES_FIRST = int(0.08 * 24000)   # 1,920 samples = 80ms (1 codec frame)
MIN_SAMPLES_SUBSEQUENT = int(0.24 * 24000)  # 5,760 samples = 240ms
```

Yield the very first audio chunk as soon as one complete codec frame is decoded (80ms). Use larger subsequent buffers (160–240ms) to absorb generation jitter without glitches. This single change saves **~220ms** from TTFA with no quality impact.

## FlashAttention 2 does not work on your GPU — use SDPA instead

**FlashAttention 2 has a hardcoded architecture whitelist that excludes SM 120.** Multiple GitHub issues (flash-attn #1665, #1987; NVIDIA Isaac-GR00T #309) confirm it fails at runtime on RTX 5070 Ti with `RuntimeError: FlashAttention only supports Ampere GPUs or newer` — despite SM 120 being newer than Ampere, the library's check doesn't recognize it. The MOSS-TTS code's `resolve_attn_implementation()` checks `major >= 8` from `get_device_capability()`, so it will try FlashAttention 2 if the package is installed and **fail silently or crash**.

**Uninstall flash-attn** and let the code fall back to PyTorch's native SDPA kernels, which should work on SM 120 (PyTorch PR #145602 added Blackwell SDPA support). The official MOSS code also explicitly configures SDPA backends:

```python
torch.backends.cuda.enable_cudnn_sdp(False)   # Disable broken cuDNN backend
torch.backends.cuda.enable_flash_sdp(True)     # PyTorch's built-in flash kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

The SDPA `flash_sdp` backend in PyTorch provides most of FlashAttention 2's performance without the external package dependency or architecture compatibility issues.

## Quantization trades modest TTFA gains for significant complexity

The 1.7B backbone in full bfloat16 occupies roughly **3.4 GB of VRAM** — well within the RTX 5070 Ti's 16 GB. Quantization would help TTFA primarily by speeding up the prefill phase (which is memory-bandwidth-bound for moderate prompt lengths). GPTQ 4-bit with Marlin kernels can deliver **~50% prefill speedup**, saving perhaps 100–200ms. However, GPTQ/AWQ quantized models have limited torch.compile compatibility (custom CUDA kernels conflict with Inductor).

A more promising path: the MOSS team publishes **GGUF-quantized weights** at `OpenMOSS-Team/MOSS-TTS-GGUF` with a llama.cpp backend. This torch-free inference path eliminates Python overhead entirely, supports native KV cache quantization (`q8_0`, `q4_0`), and can be combined with the ONNX audio tokenizer for a fully non-PyTorch pipeline. The llama.cpp backend's configs support `low_memory: true` for staged GPU loading and `flash_attn: auto` for attention optimization. If you're willing to rewrite your FastAPI wrapper to use this backend, it could be the lowest-latency option overall.

For the current PyTorch path, **bfloat16 without quantization is correct** for your hardware. Blackwell's 5th-gen Tensor Cores handle bfloat16 at the same throughput as float16, with better numerical range. Don't change this.

## Streaming architecture adjustments to the FastAPI wrapper

Several aspects of the described inference pipeline introduce unnecessary latency:

**Text chunking of 12 tokens is reasonable but can be front-loaded.** At 12.5Hz audio frame rate, the model needs relatively few text tokens to begin generating speech. Push a smaller first chunk (6–8 tokens) to start audio generation sooner, then increase subsequent chunks to 16–24 tokens. The official API uses `push_text(delta)` where `delta` is a text string from an LLM stream — the model handles tokenization internally. If you're pre-tokenizing and using `push_text_tokens()`, note that the official API method is actually `push_text()`, not `push_text_tokens()`.

**The `prefill_text_len` parameter controls how many text tokens are processed before audio generation begins.** Setting it to `rt_processor.delay_tokens_len` is the intended default, but if this value is large (e.g., 50+ tokens), it forces the model to process all delay tokens before emitting any audio. Check this value — if it's more than ~20, it may be adding hundreds of milliseconds.

**The inference semaphore serializes GPU access correctly**, but ensure it's an `asyncio.Semaphore` (not `threading.Semaphore`) if your producer thread is running in the event loop. If the producer runs in a `ThreadPoolExecutor`, a `threading.Semaphore` is correct but the thread-to-async queue handoff can add latency. Use `asyncio.Queue` with `loop.call_soon_threadsafe` for the lowest-latency bridge.

**Set the `X-Accel-Buffering: no` response header** to prevent reverse proxies (Nginx, Cloudflare) from buffering your chunked audio stream:

```python
return StreamingResponse(
    audio_generator(),
    media_type="audio/wav",
    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
)
```

## Complete optimization checklist ranked by TTFA impact

| Priority | Action | Expected TTFA savings | Effort |
|---|---|---|---|
| **1** | Remove torch.compile OR pre-warm + enable persistent cache | 500–2,000ms | Low |
| **2** | Uninstall flash-attn; ensure SDPA fallback is active | Prevents crashes; modest speedup | Low |
| **3** | Increase `CUDA_CACHE_MAXSIZE` to 4 GiB | 500–2,000ms (first restart only) | Trivial |
| **4** | Reduce MIN_SAMPLES to 80ms (1 codec frame) for first chunk | ~220ms | Low |
| **5** | Use `codec.streaming(batch_size=1)` + reduce first chunk_duration to 0.08s | 100–300ms | Medium |
| **6** | Switch codec decoder to ONNX (`MOSS-Audio-Tokenizer-ONNX`) | 100–250ms | Medium |
| **7** | Front-load smaller first text chunk (6–8 tokens) | 50–100ms | Low |
| **8** | Cache reference audio KV states across same-speaker requests | 100–300ms (repeat requests) | Medium |
| **9** | Consider llama.cpp + ONNX backend for torch-free inference | Eliminates Python/torch overhead entirely | High |

Priorities 1–4 alone should bring your TTFA from ~2.5s to **400–700ms** with minimal code changes. Adding priorities 5–7 pushes toward **250–400ms**. The llama.cpp + ONNX path (priority 9) is the nuclear option that eliminates PyTorch from the inference pipeline entirely, which is how the MOSS team's own production configs (`configs/llama_cpp/trt.yaml`) achieve the lowest latency.