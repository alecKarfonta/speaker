# TTS Pipeline Profiling

The Qwen TTS backend includes built-in profiling to measure performance at each stage of speech generation.

## Example Output

```
┌─ Generation Profile ─────────────────────────────────
│ text_validation               0.01ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
│ mode_detection                0.16ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
│ model_lookup                  0.00ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
│ param_preparation             0.01ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
│ prompt_lookup                 0.00ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
│ model_inference            7819.66ms  ███████████████████░ 100.0%
│ audio_extraction              0.00ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
├───────────────────────────────────────────────────────
│ Total generation:         7821.36ms
│ Audio duration:              3.82s
│ Real-time factor:            2.05x
└───────────────────────────────────────────────────────
```

## Profiled Stages

| Stage | Description |
|-------|-------------|
| `text_validation` | Input text normalization and validation |
| `mode_detection` | Determining generation mode (custom_voice, voice_design, voice_clone) |
| `model_lookup` | Retrieving the model from the loaded models dict |
| `speaker_validation` | Validating speaker name exists (custom_voice only) |
| `input_validation` | Validating instruct text (voice_design only) |
| `param_preparation` | Filtering and preparing generation kwargs |
| `audio_resampling` | Resampling reference audio to 24kHz (voice_clone with direct audio) |
| `prompt_lookup` | Looking up cached voice clone prompt |
| `model_inference` | **The actual model generation** - typically 99%+ of time |
| `audio_extraction` | Extracting first audio segment from model output |

## Implementation

### Core Classes

Located in `app/backends/qwen_tts.py`:

```python
@dataclass
class ProfileStage:
    """A single profiled stage with timing info."""
    name: str
    duration_ms: float
    start_time: float
    end_time: float


@dataclass
class GenerationProfile:
    """Complete profile of a generation request."""
    stages: List[ProfileStage]
    total_duration_ms: float
    audio_duration_s: float
    rtf: float  # Real-time factor
    
    def summary(self) -> str:
        """Generate formatted summary with visual bars."""
        ...
```

### ProfileTimer Context Manager

```python
class ProfileTimer:
    """Context manager for profiling generation stages."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.profile = GenerationProfile()
    
    def start(self) -> None:
        """Start overall timing."""
        self._total_start = time.perf_counter()
    
    @contextmanager
    def stage(self, name: str):
        """Time a specific stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            self.profile.add_stage(name, duration_ms, start, end)
    
    def finish(self, audio_samples: int, sample_rate: int) -> GenerationProfile:
        """Finalize profile with audio info and calculate RTF."""
        ...
```

### Usage Pattern

Each generation method wraps stages with the profiler:

```python
def _generate_custom_voice(self, text, speaker, language, instruct, profiler=None, **kwargs):
    # Model lookup stage
    if profiler:
        with profiler.stage("model_lookup"):
            model = self._models.get("custom_voice")
    else:
        model = self._models.get("custom_voice")
    
    # ... validation stages ...
    
    # Model inference stage (the main bottleneck)
    if profiler:
        with profiler.stage("model_inference"):
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or "",
                **gen_kwargs
            )
    else:
        wavs, sr = model.generate_custom_voice(...)
```

## Metrics

### Real-Time Factor (RTF)

```
RTF = generation_time / audio_duration
```

- **RTF < 1.0** = Faster than real-time (good)
- **RTF = 1.0** = Real-time
- **RTF > 1.0** = Slower than real-time

### Visual Progress Bar

Each stage shows a 20-character bar based on percentage of total time:

```python
pct = (stage.duration_ms / total_duration_ms) * 100
bar_len = int(pct / 5)  # 20 chars max
bar = "█" * bar_len + "░" * (20 - bar_len)
```

## Disabling Profiling

Pass `enable_profiling=False` to `generate_speech()`:

```python
audio, sr = backend.generate_speech(
    text="Hello world",
    voice_name="Aiden",
    enable_profiling=False  # Skip profiling overhead
)
```

## Logs

Profiling output appears at `INFO` level with the `[QwenTTSBackend._log]` prefix.

Individual stage times also logged at `DEBUG` level:
```
DEBUG | [PROFILE] model_inference: 7819.66ms
```
