"""
Qwen3-TTS Backend Implementation for Speaker Platform.

This module implements the Qwen3-TTS backend, providing:
- Custom voice generation with built-in speakers
- Voice design from natural language descriptions
- Zero-shot voice cloning from 3-second audio
- Streaming audio generation
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from app.tts_backend_base import TTSBackendBase, Voice
from app.backends.qwen_tts_config import QwenTTSConfig
from app.backends.qwen_voices import QWEN_SPEAKERS, QwenVoice, get_speaker_by_id
from app.log_util import get_class_logger, Colors


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
    stages: List[ProfileStage] = field(default_factory=list)
    total_duration_ms: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0  # Real-time factor
    
    def add_stage(self, name: str, duration_ms: float, start_time: float, end_time: float) -> None:
        self.stages.append(ProfileStage(name, duration_ms, start_time, end_time))
    
    def summary(self) -> str:
        """Generate a formatted summary of the profile."""
        lines = ["┌─ Generation Profile ─────────────────────────────────"]
        
        # Stage breakdown
        for stage in self.stages:
            pct = (stage.duration_ms / self.total_duration_ms * 100) if self.total_duration_ms > 0 else 0
            bar_len = int(pct / 5)  # 20 chars max
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"│ {stage.name:<25} {stage.duration_ms:>8.2f}ms  {bar} {pct:>5.1f}%")
        
        lines.append("├───────────────────────────────────────────────────────")
        lines.append(f"│ Total generation:        {self.total_duration_ms:>8.2f}ms")
        lines.append(f"│ Audio duration:          {self.audio_duration_s:>8.2f}s")
        lines.append(f"│ Real-time factor:        {self.rtf:>8.2f}x")
        lines.append("└───────────────────────────────────────────────────────")
        
        return "\n".join(lines)


class ProfileTimer:
    """Context manager for profiling generation stages."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.profile = GenerationProfile()
        self._current_stage: Optional[str] = None
        self._stage_start: float = 0.0
        self._total_start: float = 0.0
    
    def start(self) -> None:
        """Start overall timing."""
        self._total_start = time.perf_counter()
    
    @contextmanager
    def stage(self, name: str):
        """Time a specific stage."""
        start = time.perf_counter()
        self._current_stage = name
        self._stage_start = start
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            self.profile.add_stage(name, duration_ms, start, end)
            self.logger.debug(f"[PROFILE] {name}: {duration_ms:.2f}ms")
    
    def finish(self, audio_samples: int, sample_rate: int) -> GenerationProfile:
        """Finalize profile with audio info."""
        total_end = time.perf_counter()
        self.profile.total_duration_ms = (total_end - self._total_start) * 1000
        self.profile.audio_duration_s = audio_samples / sample_rate if sample_rate > 0 else 0
        self.profile.rtf = (self.profile.total_duration_ms / 1000) / self.profile.audio_duration_s if self.profile.audio_duration_s > 0 else 0
        return self.profile

# Lazy imports for qwen_tts - may not be installed
_qwen_tts_available = False
Qwen3TTSModel = None
Qwen3TTSTokenizer = None

try:
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    _qwen_tts_available = True
except ImportError:
    pass


class QwenTTSBackend(TTSBackendBase):
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
    
    # Map ISO 639-1 codes to the full language names expected by Qwen models
    LANGUAGE_CODE_MAP = {
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "auto": "auto"
    }
    
    _SAMPLE_RATE = 24000  # Qwen3-TTS outputs 24kHz audio

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        qwen_config: Optional[QwenTTSConfig] = None
    ):
        """Initialize Qwen3-TTS backend.
        
        Args:
            logger: Logger instance
            config: Generic config dict (for TTSBackendBase compatibility)
            qwen_config: Typed configuration object
        """
        super().__init__(logger=logger, config=config)
        
        # Set up class logger for detailed output
        self.log = get_class_logger(self, self.logger)
        
        # Use typed config if provided, else create from env vars
        if qwen_config is not None:
            self.qwen_config = qwen_config
        elif config:
            self.qwen_config = QwenTTSConfig.from_dict(config)
        else:
            # Read QWEN_TTS_* environment variables
            self.qwen_config = QwenTTSConfig.from_env()
        
        self._models: Dict[str, Any] = {}
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=self.qwen_config.max_workers)
        self._voice_clone_prompts: Dict[str, dict] = {}  # Cache for voice clone prompts
        self._initialized = False
        self._streaming_enabled = False
        
        self.log.info(f"Backend created with model_size={self.qwen_config.model_size}, modes: custom={self.qwen_config.enable_custom_voice}, design={self.qwen_config.enable_voice_design}, clone={self.qwen_config.enable_voice_clone}")
        
        # Auto-initialize model on construction (like GLM backend)
        self.initialize_model()
        self.load_voices()
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language name/code for the Qwen model.
        
        Translates 'en' -> 'english', 'ZH' -> 'chinese', etc.
        """
        if not language:
            return "auto"
        
        # Check explicit mapping first (lowercase)
        lang_lower = language.lower()
        if lang_lower in self.LANGUAGE_CODE_MAP:
            return self.LANGUAGE_CODE_MAP[lang_lower]
        
        # If it's already one of the full names, return it
        if lang_lower in [l.lower() for l in self.SUPPORTED_LANGUAGES]:
            return lang_lower
            
        return lang_lower

    @property
    def backend_name(self) -> str:
        return "qwen-tts"
    
    @property
    def model_name(self) -> str:
        return f"Qwen3-TTS-{self.qwen_config.model_size}"
    
    @property
    def sample_rate(self) -> int:
        return self._SAMPLE_RATE
    
    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()
    
    def initialize_model(self) -> None:
        """Initialize Qwen3-TTS models based on configuration."""
        if not _qwen_tts_available:
            raise ImportError(
                "qwen-tts package not installed. Install with: pip install qwen-tts"
            )
        
        self.qwen_config.validate()
        
        import torch
        device = self.qwen_config.device or "cuda:0"
        dtype = getattr(torch, self.qwen_config.dtype)
        
        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
        }
        
        if self.qwen_config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load CustomVoice model (primary)
        if self.qwen_config.enable_custom_voice:
            model_id = self.SUPPORTED_MODELS["custom_voice"][self.qwen_config.model_size]
            self.log.info(f"Loading CustomVoice model: '{model_id}'...")
            start = time.time()
            self._models["custom_voice"] = Qwen3TTSModel.from_pretrained(
                self.qwen_config.custom_voice_model_path or model_id,
                **load_kwargs
            )
            self.log.info(f"CustomVoice model loaded in {time.time() - start:.2f}s")
        
        # Load VoiceDesign model
        if self.qwen_config.enable_voice_design:
            model_id = self.SUPPORTED_MODELS["voice_design"]["1.7B"]
            self.log.info(f"Loading VoiceDesign model: '{model_id}'...")
            start = time.time()
            self._models["voice_design"] = Qwen3TTSModel.from_pretrained(
                self.qwen_config.voice_design_model_path or model_id,
                **load_kwargs
            )
            self.log.info(f"VoiceDesign model loaded in {time.time() - start:.2f}s")
        
        # Load Base model for voice cloning
        if self.qwen_config.enable_voice_clone:
            model_id = self.SUPPORTED_MODELS["voice_clone"][self.qwen_config.model_size]
            self.log.info(f"Loading VoiceClone model: '{model_id}'...")
            start = time.time()
            self._models["voice_clone"] = Qwen3TTSModel.from_pretrained(
                self.qwen_config.voice_clone_model_path or model_id,
                **load_kwargs
            )
            self.log.info(f"VoiceClone model loaded in {time.time() - start:.2f}s")
        
        # Load tokenizer for audio encoding/decoding
        self.log.debug("Loading tokenizer...")
        self._tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self.qwen_config.tokenizer_path or "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map=device
        )
        
        # Populate voices dict with predefined speakers
        for name, qwen_voice in QWEN_SPEAKERS.items():
            self.voices[name] = Voice(
                name=name,
                file_paths=[],  # Predefined voices don't need files
                metadata=qwen_voice.to_dict()
            )
        
        self._initialized = True
        self.log.info(f"Backend initialized with {len(self._models)} models and {len(self.voices)} voices")
        
        # Enable streaming optimizations (torch.compile + CUDA graphs)
        if self.qwen_config.enable_streaming:
            clone_model = self._models.get("voice_clone")
            if clone_model and hasattr(clone_model, 'enable_streaming_optimizations'):
                self.log.info("Enabling streaming optimizations (torch.compile + CUDA graphs)...")
                try:
                    clone_model.enable_streaming_optimizations(
                        decode_window_frames=self.qwen_config.streaming_decode_window,
                        use_compile=self.qwen_config.streaming_compile,
                        compile_mode=self.qwen_config.streaming_compile_mode,
                    )
                    self._streaming_enabled = True
                    self.log.info("Streaming optimizations enabled")
                except Exception as e:
                    self._streaming_enabled = False
                    self.log.warning(f"Streaming optimizations failed (will use batch fallback): {e}")
            else:
                self._streaming_enabled = False
                self.log.warning("Voice clone model missing stream_generate_voice_clone — streaming disabled")
        else:
            self._streaming_enabled = False
    
    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """Load a voice from audio file for voice cloning.
        
        Args:
            voice_name: Identifier for the cloned voice
            voice_path: Path to reference audio file (3+ seconds)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize_model() first.")
        
        path = Path(voice_path)
        if not path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        # Load audio and create voice clone prompt
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            
            # Use internal method to create the voice
            self._create_voice_prompt(voice_name, audio, sr, ref_text=None, file_path=voice_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load voice '{voice_name}': {e}")
    
    def create_voice_from_audio(
        self,
        voice_name: str,
        audio_data: np.ndarray,
        sample_rate: int,
        ref_text: Optional[str] = None,
        x_vector_only: bool = False
    ) -> None:
        """Create a voice clone prompt from audio data (for API use).
        
        Args:
            voice_name: Identifier for the cloned voice
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            ref_text: Transcript of the reference audio (improves quality)
            x_vector_only: Use x-vector only mode (no ref_text needed, lower quality)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize_model() first.")
        
        self._create_voice_prompt(
            voice_name, audio_data, sample_rate, 
            ref_text=ref_text, 
            x_vector_only=x_vector_only
        )
    
    def _create_voice_prompt(
        self,
        voice_name: str,
        audio: np.ndarray,
        sr: int,
        ref_text: Optional[str] = None,
        file_path: Optional[str] = None,
        x_vector_only: bool = False
    ) -> None:
        """Internal method to create voice clone prompt."""
        # Convert stereo to mono if needed (Qwen model requires mono audio)
        if audio.ndim > 1:
            self.log.debug(f"Converting stereo ({audio.shape[1]} channels) to mono")
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != self._SAMPLE_RATE:
            import librosa
            self.log.debug(f"Resampling from {sr} to {self._SAMPLE_RATE}")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._SAMPLE_RATE)
            sr = self._SAMPLE_RATE
        
        # Ensure audio is a contiguous, writable float32 array (required by Qwen model)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        
        ref_duration = len(audio) / sr
        self.log.info(f"Creating voice prompt '{voice_name}': duration={ref_duration:.2f}s, ref_text={'yes' if ref_text else 'no'}, x_vector_only={x_vector_only}")
        
        # Create voice clone prompt
        model = self._models.get("voice_clone")
        if not model:
            raise RuntimeError("VoiceClone model not loaded. Enable it in config.")
        
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=(audio, sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )
            self._voice_clone_prompts[voice_name] = prompt
            
            self.voices[voice_name] = Voice(
                name=voice_name,
                file_paths=[file_path] if file_path else [],
                metadata={
                    "type": "cloned", 
                    "sample_rate": self._SAMPLE_RATE,
                    "has_ref_text": bool(ref_text),
                    "x_vector_only": x_vector_only
                }
            )
            self.log.info(f"Voice prompt '{voice_name}' created and cached")
            
        except Exception as e:
            self.log.error(f"Failed to create voice prompt: {e}")
            raise RuntimeError(f"Failed to create voice prompt '{voice_name}': {e}")
    
    def load_voices(self, voices_dir: Optional[str] = None) -> None:
        """Load all voices from the voices directory on startup.
        
        For each voice directory, loads the first audio file and attempts
        to transcribe it via STT for best voice cloning quality.
        
        Args:
            voices_dir: Path to voices directory (default: from VOICES_DIR env or data/voices)
        """
        import os
        import requests
        import soundfile as sf
        
        if voices_dir is None:
            voices_dir = os.environ.get("VOICES_DIR", "data/voices")
        
        voices_path = Path(voices_dir)
        if not voices_path.exists():
            self.log.warning(f"Voices directory not found: {voices_dir}")
            return
        
        # Get STT config from environment
        stt_url = os.environ.get("STT_API_URL", "")
        stt_key = os.environ.get("STT_API_KEY", "")
        stt_model = os.environ.get("STT_MODEL", "distil-large-v3.5-ct2")
        
        voice_dirs = [d for d in voices_path.iterdir() if d.is_dir()]
        self.log.info(f"Loading {len(voice_dirs)} voices from {voices_dir}...")
        
        loaded = 0
        for voice_dir in sorted(voice_dirs):
            voice_name = voice_dir.name
            
            # Find audio files
            audio_files = list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.mp3"))
            if not audio_files:
                self.log.debug(f"Skipping {voice_name}: no audio files found")
                continue
            
            # Use first audio file
            audio_path = sorted(audio_files)[0]
            
            try:
                # Load audio
                audio, sr = sf.read(str(audio_path))
                ref_duration = len(audio) / sr
                self.log.debug(f"Loading voice '{voice_name}' from {audio_path.name} ({ref_duration:.1f}s)")
                
                # Try to transcribe via STT for best quality
                ref_text = None
                if stt_url:
                    try:
                        with open(audio_path, 'rb') as f:
                            response = requests.post(
                                stt_url,
                                headers={"Authorization": f"Bearer {stt_key}"} if stt_key else {},
                                files={"file": (audio_path.name, f, "audio/wav")},
                                data={"model": stt_model, "language": "en"},
                                timeout=60
                            )
                        if response.status_code == 200:
                            result = response.json()
                            ref_text = result.get("text", "").strip()
                            if ref_text:
                                self.log.debug(f"STT for '{voice_name}': '{ref_text[:40]}...'")
                    except requests.exceptions.ConnectionError:
                        self.log.debug("STT service not available for voice loading")
                        stt_url = ""  # Don't retry for remaining voices
                    except Exception as e:
                        self.log.debug(f"STT error for '{voice_name}': {e}")
                
                # Create voice prompt
                self._create_voice_prompt(
                    voice_name=voice_name,
                    audio=audio,
                    sr=sr,
                    ref_text=ref_text,
                    file_path=str(audio_path),
                    x_vector_only=(ref_text is None)
                )
                loaded += 1
                
            except Exception as e:
                self.log.warning(f"Failed to load voice '{voice_name}': {e}")
        
        self.log.info(f"Loaded {loaded}/{len(voice_dirs)} voices from {voices_dir}")
    
    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "Auto",
        mode: str = "auto",
        instruct: Optional[str] = None,
        ref_audio: Optional[Tuple[np.ndarray, int]] = None,
        ref_text: Optional[str] = None,
        enable_profiling: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: Input text to synthesize
            voice_name: Speaker name or cloned voice ID
            language: Target language or "Auto" for detection
            mode: Generation mode - "auto", "custom_voice", "voice_design", "voice_clone"
            instruct: Instruction for style control (optional)
            ref_audio: Reference audio tuple (waveform, sample_rate) for voice cloning
            ref_text: Reference text transcript for voice cloning
            enable_profiling: Enable detailed stage profiling (default: True)
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        if not self._initialized:
            self.log.error("Backend not initialized")
            raise RuntimeError("Backend not initialized. Call initialize_model() first.")
        
        # Create profiler
        profiler = ProfileTimer(self.logger) if enable_profiling else None
        if profiler:
            profiler.start()
        
        # Preprocessing stage
        if profiler:
            with profiler.stage("text_validation"):
                text = self.validate_text(text)
        else:
            text = self.validate_text(text)
            
        # Normalize language
        language = self._normalize_language(language)
        
        self.log.info(f"Request: text='{text_preview}' ({len(text)} chars), voice='{voice_name}', lang={language}, mode={mode}")
        
        # Mode detection stage
        if profiler:
            with profiler.stage("mode_detection"):
                mode, voice_name, instruct = self._detect_mode(mode, voice_name, instruct)
        else:
            mode, voice_name, instruct = self._detect_mode(mode, voice_name, instruct)
        
        # Generate with profiler
        if mode == "custom_voice":
            result = self._generate_custom_voice(text, voice_name, language, instruct, profiler=profiler, **kwargs)
        elif mode == "voice_design":
            result = self._generate_voice_design(text, language, instruct or voice_name, profiler=profiler, **kwargs)
        elif mode == "voice_clone":
            result = self._generate_voice_clone(text, voice_name, language, instruct=instruct, ref_audio=ref_audio, ref_text=ref_text, profiler=profiler, **kwargs)
        else:
            self.log.error(f"Unsupported mode: {mode}")
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Log result with profile
        audio, sr = result
        
        if profiler:
            profile = profiler.finish(len(audio), sr)
            self.log.info(f"\n{profile.summary()}")
        else:
            duration = len(audio) / sr
            gen_time = time.time() - start_time
            rtf = gen_time / duration if duration > 0 else 0
            self.log.info(f"Generated: duration={duration:.2f}s, time={gen_time:.2f}s, rtf={rtf:.2f}x, samples={len(audio)}")
        
        return result
    
    def _detect_mode(
        self,
        mode: str,
        voice_name: str,
        instruct: Optional[str]
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Detect and resolve generation mode from inputs."""
        # Auto-detect mode based on voice_name
        if mode == "auto":
            if voice_name in QWEN_SPEAKERS:
                mode = "custom_voice"
                self.log.debug(f"Auto-detected mode=custom_voice for speaker '{voice_name}'")
            elif voice_name in self._voice_clone_prompts:
                mode = "voice_clone"
                self.log.debug(f"Auto-detected mode=voice_clone for cached voice '{voice_name}'")
            else:
                # Treat voice_name as a design instruction
                mode = "voice_design"
                instruct = voice_name
                voice_name = None
                self.log.debug(f"Auto-detected mode=voice_design with instruct='{instruct}'")
        
        # If custom_voice mode is requested but voice is actually a cloned voice,
        # automatically switch to voice_clone mode for seamless user experience
        elif mode == "custom_voice" and voice_name not in QWEN_SPEAKERS:
            if voice_name in self._voice_clone_prompts:
                mode = "voice_clone"
                self.log.info(f"Redirecting cloned voice '{voice_name}' from custom_voice to voice_clone mode")
        
        return mode, voice_name, instruct
    
    def _generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str],
        profiler: Optional[ProfileTimer] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using predefined speaker with optional instruction."""
        self.log.debug(f"CustomVoice: speaker='{speaker}', instruct='{instruct}', lang={language}")
        
        # Model lookup stage
        if profiler:
            with profiler.stage("model_lookup"):
                model = self._models.get("custom_voice")
        else:
            model = self._models.get("custom_voice")
            
        if not model:
            self.log.error("CustomVoice model not loaded")
            raise RuntimeError("CustomVoice model not loaded. Enable it in config.")
        
        # Speaker validation stage
        if profiler:
            with profiler.stage("speaker_validation"):
                if speaker not in QWEN_SPEAKERS:
                    available = list(QWEN_SPEAKERS.keys())
                    self.log.error(f"Unknown speaker '{speaker}'. Available: {available}")
                    raise ValueError(f"Unknown speaker '{speaker}'. Available: {available}")
        else:
            if speaker not in QWEN_SPEAKERS:
                available = list(QWEN_SPEAKERS.keys())
                self.log.error(f"Unknown speaker '{speaker}'. Available: {available}")
                raise ValueError(f"Unknown speaker '{speaker}'. Available: {available}")
        
        # Parameter preparation stage
        if profiler:
            with profiler.stage("param_preparation"):
                gen_kwargs = self._filter_gen_kwargs(kwargs)
        else:
            gen_kwargs = self._filter_gen_kwargs(kwargs)
        self.log.debug(f"Generation params: {gen_kwargs}")
        
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
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or "",
                **gen_kwargs
            )
        
        # Post-processing stage
        if profiler:
            with profiler.stage("audio_extraction"):
                audio = wavs[0]
        else:
            audio = wavs[0]
        
        self.log.debug(f"CustomVoice generated {len(wavs)} segments, sr={sr}")
        return audio, sr
    
    def _generate_voice_design(
        self,
        text: str,
        language: str,
        instruct: str,
        profiler: Optional[ProfileTimer] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with voice designed from natural language description."""
        instruct_preview = instruct[:60] + "..." if instruct and len(instruct) > 60 else instruct
        self.log.debug(f"VoiceDesign: instruct='{instruct_preview}', lang={language}")
        
        # Model lookup stage
        if profiler:
            with profiler.stage("model_lookup"):
                model = self._models.get("voice_design")
        else:
            model = self._models.get("voice_design")
            
        if not model:
            self.log.error("VoiceDesign model not loaded")
            raise RuntimeError("VoiceDesign model not loaded. Enable it in config.")
        
        # Input validation stage
        if profiler:
            with profiler.stage("input_validation"):
                if not instruct:
                    self.log.error("Voice design requires instruct description")
                    raise ValueError("Voice design requires an 'instruct' description")
        else:
            if not instruct:
                self.log.error("Voice design requires instruct description")
                raise ValueError("Voice design requires an 'instruct' description")
        
        # Parameter preparation stage
        if profiler:
            with profiler.stage("param_preparation"):
                gen_kwargs = self._filter_gen_kwargs(kwargs)
        else:
            gen_kwargs = self._filter_gen_kwargs(kwargs)
        self.log.debug(f"Generation params: {gen_kwargs}")
        
        # Model inference stage (the main bottleneck)
        if profiler:
            with profiler.stage("model_inference"):
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                    **gen_kwargs
                )
        else:
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **gen_kwargs
            )
        
        # Post-processing stage
        if profiler:
            with profiler.stage("audio_extraction"):
                audio = wavs[0]
        else:
            audio = wavs[0]
        
        self.log.debug(f"VoiceDesign generated {len(wavs)} segments, sr={sr}")
        return audio, sr
    
    def _generate_voice_clone(
        self,
        text: str,
        voice_name: str,
        language: str,
        instruct: Optional[str] = None,
        ref_audio: Optional[Tuple[np.ndarray, int]] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        profiler: Optional[ProfileTimer] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Clone voice from cached prompt or direct audio and synthesize new content.
        
        Args:
            text: Text to synthesize
            voice_name: Name of cached voice prompt
            language: Target language
            instruct: Style instruction for controlling speech (e.g., "speak with excitement")
            ref_audio: Direct reference audio tuple (waveform, sample_rate)
            ref_text: Transcript of reference audio
            x_vector_only_mode: Use x-vector only (no ref_text needed)
            profiler: Optional profiler for timing
            **kwargs: Additional generation parameters
        """
        instruct_preview = instruct[:40] + "..." if instruct and len(instruct) > 40 else instruct
        self.log.debug(f"VoiceClone: voice='{voice_name}', instruct='{instruct_preview}', has_ref_audio={ref_audio is not None}, x_vector_only={x_vector_only_mode}")
        
        # Model lookup stage
        if profiler:
            with profiler.stage("model_lookup"):
                model = self._models.get("voice_clone")
        else:
            model = self._models.get("voice_clone")
            
        if not model:
            self.log.error("VoiceClone model not loaded")
            raise RuntimeError("VoiceClone model not loaded. Enable it in config.")
        
        # Parameter preparation stage
        if profiler:
            with profiler.stage("param_preparation"):
                gen_kwargs = self._filter_gen_kwargs(kwargs)
        else:
            gen_kwargs = self._filter_gen_kwargs(kwargs)
        
        # If ref_audio is provided, generate directly without caching
        if ref_audio is not None:
            audio_data, audio_sr = ref_audio
            ref_duration = len(audio_data) / audio_sr
            self.log.info(f"Cloning from ref_audio: duration={ref_duration:.2f}s, sr={audio_sr}")
            
            # Audio preprocessing stage (resampling)
            if profiler:
                with profiler.stage("audio_resampling"):
                    if audio_sr != self._SAMPLE_RATE:
                        import librosa
                        self.log.debug(f"Resampling from {audio_sr} to {self._SAMPLE_RATE}")
                        audio_data = librosa.resample(audio_data, orig_sr=audio_sr, target_sr=self._SAMPLE_RATE)
                        audio_sr = self._SAMPLE_RATE
            else:
                if audio_sr != self._SAMPLE_RATE:
                    import librosa
                    self.log.debug(f"Resampling from {audio_sr} to {self._SAMPLE_RATE}")
                    audio_data = librosa.resample(audio_data, orig_sr=audio_sr, target_sr=self._SAMPLE_RATE)
                    audio_sr = self._SAMPLE_RATE
            
            # Model inference stage with direct audio
            if profiler:
                with profiler.stage("model_inference"):
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=(audio_data, audio_sr),
                        ref_text=ref_text,
                        instruct=instruct or "",
                        x_vector_only_mode=x_vector_only_mode,
                        **gen_kwargs
                    )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=(audio_data, audio_sr),
                    ref_text=ref_text,
                    instruct=instruct or "",
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs
                )
            
            # Post-processing stage
            if profiler:
                with profiler.stage("audio_extraction"):
                    audio = wavs[0]
            else:
                audio = wavs[0]
                
            self.log.debug(f"VoiceClone (direct) generated {len(wavs)} segments")
            return audio, sr
        
        # Otherwise use cached prompt
        if profiler:
            with profiler.stage("prompt_lookup"):
                voice_clone_prompt = self._voice_clone_prompts.get(voice_name)
        else:
            voice_clone_prompt = self._voice_clone_prompts.get(voice_name)
            
        if not voice_clone_prompt:
            self.log.error(f"Cached voice '{voice_name}' not found")
            raise ValueError(
                f"Voice '{voice_name}' not found. Load it first with load_voice()."
            )
        
        self.log.debug(f"Using cached prompt for '{voice_name}'")
        
        # Model inference stage with cached prompt
        if profiler:
            with profiler.stage("model_inference"):
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    instruct=instruct or "",
                    **gen_kwargs
                )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                instruct=instruct or "",
                **gen_kwargs
            )
        
        # Post-processing stage
        if profiler:
            with profiler.stage("audio_extraction"):
                audio = wavs[0]
        else:
            audio = wavs[0]
        
        self.log.debug(f"VoiceClone (cached) generated {len(wavs)} segments")
        return audio, sr
    
    def generate_speech_streaming(
        self,
        text: str,
        voice_name: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        **kwargs
    ):
        """True streaming generation using Qwen3-TTS-streaming fork.
        
        Yields (audio_chunk_np, sample_rate) tuples as audio is generated
        incrementally. Uses two-phase streaming: aggressive first chunk
        for low TTFA, then stable settings for quality.
        
        Args:
            text: Text to synthesize
            voice_name: Name of cached voice or predefined speaker
            language: Target language or "Auto"
            instruct: Optional style instruction
            **kwargs: Additional generation parameters
            
        Yields:
            Tuple of (np.ndarray audio chunk, int sample_rate)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize_model() first.")
            
        # Normalize language
        language = self._normalize_language(language)
        
        if not getattr(self, '_streaming_enabled', False):
            raise RuntimeError(
                "Streaming not enabled. Ensure qwen-tts-streaming fork is installed "
                "and enable_streaming=True in config."
            )
        
        model = self._models.get("voice_clone")
        if not model:
            raise RuntimeError("VoiceClone model not loaded. Enable it in config.")
        
        # Get voice clone prompt (required for streaming)
        voice_clone_prompt = self._voice_clone_prompts.get(voice_name)
        if not voice_clone_prompt:
            # Try predefined speakers — they need custom_voice model, not streaming
            if voice_name in QWEN_SPEAKERS:
                raise ValueError(
                    f"Streaming requires voice cloning. Speaker '{voice_name}' is a "
                    f"predefined speaker — use /tts (non-streaming) for predefined speakers, "
                    f"or clone the voice first via load_voice()."
                )
            raise ValueError(
                f"Voice '{voice_name}' not found. Load it first with load_voice()."
            )
        
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self.log.info(
            f"Streaming: text='{text_preview}' ({len(text)} chars), "
            f"voice='{voice_name}', lang={language}"
        )
        
        cfg = self.qwen_config
        stream_kwargs = {
            "text": text,
            "language": language,
            "voice_clone_prompt": voice_clone_prompt,
            "emit_every_frames": cfg.streaming_emit_every,
            "decode_window_frames": cfg.streaming_decode_window,
            "first_chunk_emit_every": cfg.streaming_first_chunk_emit,
            "first_chunk_decode_window": cfg.streaming_first_chunk_window,
            "first_chunk_frames": cfg.streaming_first_chunk_frames,
        }
        
        import time
        t0 = time.perf_counter()
        first_chunk = True
        chunk_count = 0
        
        for chunk, sr in model.stream_generate_voice_clone(**stream_kwargs):
            if first_chunk:
                ttfa = (time.perf_counter() - t0) * 1000
                self.log.info(f"TTFA: {ttfa:.0f}ms")
                first_chunk = False
            chunk_count += 1
            yield chunk, sr, {"sample_rate": sr, "streaming": True}
        total = (time.perf_counter() - t0) * 1000
        self.log.info(f"Streaming complete: {chunk_count} chunks in {total:.0f}ms")
    
    def _filter_gen_kwargs(self, kwargs: dict) -> dict:
        """Extract and validate generation parameters."""
        valid_keys = {
            "max_new_tokens", "top_p", "top_k", 
            "temperature", "repetition_penalty"
        }
        return {
            k: v for k, v in kwargs.items() 
            if k in valid_keys and v is not None
        }
    
    def get_predefined_speakers(self) -> List[QwenVoice]:
        """Return list of available predefined speakers."""
        return list(QWEN_SPEAKERS.values())
    
    def get_speaker_info(self, speaker_name: str) -> Optional[QwenVoice]:
        """Get detailed info about a predefined speaker."""
        return get_speaker_by_id(speaker_name)
    
    def get_cloned_voices(self) -> List[dict]:
        """Return list of cloned voices that have been cached.
        
        Returns:
            List of dicts with voice info: name, type, etc.
        """
        cloned = []
        for name, voice in self.voices.items():
            if voice.metadata and voice.metadata.get("type") == "cloned":
                cloned.append({
                    "id": name,
                    "name": name,
                    "type": "cloned",
                    "description": "User-cloned voice",
                    "gender": "unknown",
                    "native_language": "unknown"
                })
        return cloned
    
    def has_cloned_voice(self, voice_name: str) -> bool:
        """Check if a cloned voice exists."""
        return voice_name in self._voice_clone_prompts
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self.log.info("Shutting down backend...")
        self._executor.shutdown(wait=False)
        self._models.clear()
        self._voice_clone_prompts.clear()
        self._tokenizer = None
        self._initialized = False
        self.log.info("Backend shutdown complete")
