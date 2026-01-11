"""
GLM-TTS Backend implementation.
Wraps the GLM-TTS model for text-to-speech synthesis with zero-shot voice cloning.

GLM-TTS uses a two-stage architecture:
1. LLM (Llama-based) converts text to speech token sequences
2. Flow Matching model converts tokens to mel-spectrograms, then vocoder generates audio

Reference: https://github.com/zai-org/GLM-TTS
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from functools import partial
import numpy as np
import torch
import torchaudio

from app.tts_backend_base import TTSBackendBase, Voice

# Add GLM-TTS to path
GLM_TTS_DIR = Path(__file__).parent.parent.parent / "GLM-TTS"
if GLM_TTS_DIR.exists() and str(GLM_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(GLM_TTS_DIR))

# GLM-TTS specific imports
#try:
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from llm.glmtts import GLMTTS
from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import tts_model_util, yaml_util
from utils.audio import mel_spectrogram
GLM_DEPS_AVAILABLE = True
#except ImportError as e:
#    GLM_DEPS_AVAILABLE = False
#    _import_error = str(e)

# Environment variable configuration for LLM optimization:
# GLM_TTS_DTYPE: fp32, fp16, bf16 (default: fp16)
# GLM_TTS_QUANTIZATION: none, 4bit, 8bit (default: none)
# GLM_TTS_ATTENTION: eager, sdpa, flash_attention_2, flash_attention_3 (default: eager)
#
# Flow/DiT optimization:
# GLM_TTS_FLOW_DTYPE: fp32, fp16, bf16 (default: fp16) 
# GLM_TTS_CFG_RATE: 0.0-1.0 (default: 0.7, 0=disabled for 2x speedup)
# GLM_TTS_COMPILE_FLOW: true/false (default: false) - torch.compile for DiT
# GLM_TTS_COMPILE_VOCODER: true/false (default: false) - torch.compile for HiFT


class GLMTTSBackend(TTSBackendBase):
    """
    GLM-TTS Backend for high-quality TTS with zero-shot voice cloning.
    
    Features:
    - Zero-shot voice cloning (3-10 seconds of prompt audio)
    - Streaming inference support
    - Multi-language support (primarily Chinese + English)
    - RL-enhanced emotion control
    """
    
    SUPPORTED_LANGUAGES = ["zh", "en"]
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        if not GLM_DEPS_AVAILABLE:
            raise ImportError(
                f"GLM-TTS dependencies not available: {_import_error}. "
                "Please ensure GLM-TTS is properly set up."
            )
        
        super().__init__(logger, config)
        
        # Model paths - default to GLM-TTS/ckpt
        self._model_path = model_path or self.config.get("model_path") or str(GLM_TTS_DIR / "ckpt")
        self._sample_rate = self.config.get("sample_rate", 24000)
        
        # Model components
        self.llm = None
        self.flow = None  # Token2Wav wrapper
        self.frontend = None
        self.text_frontend = None
        self.speech_tokenizer = None
        self.special_token_ids = None
        
        # Inference parameters
        self.beam_size = self.config.get("beam_size", 1)
        self.sampling = int(os.environ.get("GLM_TTS_SAMPLING", self.config.get("sampling", 25)))
        self.use_phoneme = self.config.get("use_phoneme", False)
        self.use_cache = self.config.get("use_cache", True)
        
        # Flow model timesteps (lower = faster, quality tradeoff) - default 10, try 5-8 for speed
        self.flow_steps = int(os.environ.get("GLM_TTS_FLOW_STEPS", self.config.get("flow_steps", 10)))
        
        # Inference engine selection: 'transformers' (default) or 'vllm' (high-performance)
        self.llm_engine = os.environ.get("GLM_TTS_ENGINE", "transformers").lower()
        
        # vLLM-specific quantization: 'fp8' (Ada/Hopper/Blackwell), 'awq', 'gptq', or 'none'
        # This is separate from GLM_TTS_QUANTIZATION which is for BitsAndBytes (transformers engine)
        self.vllm_quantization = os.environ.get("GLM_TTS_VLLM_QUANTIZATION", "none").lower()
        if self.vllm_quantization == "none":
            self.vllm_quantization = None
        
        # torch.compile for JIT optimization (requires PyTorch 2.0+)
        self.use_torch_compile = os.environ.get("GLM_TTS_TORCH_COMPILE", "").lower() in ("true", "1", "yes")
        self.compile_flow = os.environ.get("GLM_TTS_COMPILE_FLOW", "").lower() in ("true", "1", "yes")
        self.compile_vocoder = os.environ.get("GLM_TTS_COMPILE_VOCODER", "").lower() in ("true", "1", "yes")
        
        # Vocoder engine selection: 'pytorch' (default) or 'tensorrt' (2-3x faster)
        self.vocoder_engine = os.environ.get("GLM_TTS_VOCODER_ENGINE", "pytorch").lower()
        self.tensorrt_engine_path = os.environ.get("GLM_TTS_TENSORRT_ENGINE_PATH", "/app/data/hift.engine")
        
        # Flow model optimization settings
        self.flow_dtype = self._get_flow_dtype()
        self.cfg_rate = float(os.environ.get("GLM_TTS_CFG_RATE", "0.7"))
        
        # Token generation limits (affects how much audio is generated per text)
        self.max_token_text_ratio = float(os.environ.get("GLM_TTS_MAX_TOKEN_RATIO", self.config.get("max_token_text_ratio", 20.0)))
        self.min_token_text_ratio = float(os.environ.get("GLM_TTS_MIN_TOKEN_RATIO", self.config.get("min_token_text_ratio", 2.0)))
        
        # Audio validation thresholds (post-processing to trim garbage audio)
        # MAX_SEC_PER_WORD: Expected max seconds per word for slow speech (default: 0.65 = ~92 WPM)
        # GARBAGE_THRESHOLD: Seconds per word above which audio is considered garbage (default: 1.0)
        # SILENCE_THRESHOLD_DB: dB threshold for trailing silence detection (default: -40)
        self.max_sec_per_word = float(os.environ.get("GLM_TTS_MAX_SEC_PER_WORD", self.config.get("max_sec_per_word", 0.65)))
        self.garbage_threshold = float(os.environ.get("GLM_TTS_GARBAGE_THRESHOLD", self.config.get("garbage_threshold", 1.0)))
        self.silence_threshold_db = float(os.environ.get("GLM_TTS_SILENCE_THRESHOLD_DB", self.config.get("silence_threshold_db", -40)))
        
        # Text chunking parameters
        self.max_text_len = self.config.get("max_text_len", 200)  # Increased from 60
        self.min_text_len = self.config.get("min_text_len", 50)   # Increased from 30
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LLM optimization settings (from env vars or config)
        self.llm_dtype = self._get_llm_dtype()
        self.llm_quantization = self._get_llm_quantization()
        self.llm_attention = self._get_llm_attention()
        
        self.logger.info(f"Initializing GLMTTSBackend with model_path: {self._model_path}")
        self.logger.info(f"LLM settings: dtype={self.llm_dtype}, quantization={self.llm_quantization}, attention={self.llm_attention}, engine={self.llm_engine}")
        self.logger.info(f"Flow settings: dtype={self.flow_dtype}, cfg_rate={self.cfg_rate}, compile_flow={self.compile_flow}")
        self.logger.info(f"Vocoder settings: engine={self.vocoder_engine}, compile_vocoder={self.compile_vocoder}")
        self.logger.info(f"Speed settings: flow_steps={self.flow_steps}, sampling={self.sampling}, torch_compile={self.use_torch_compile}")
        self.logger.info(f"Validation settings: max_sec_per_word={self.max_sec_per_word}, garbage_threshold={self.garbage_threshold}, silence_db={self.silence_threshold_db}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}, Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            # Enable TensorFloat32 for better GPU performance on Ampere+ GPUs
            torch.set_float32_matmul_precision('high')
            self.logger.info("Enabled TensorFloat32 matmul precision for GPU acceleration")
        self.initialize_model()
        self.load_voices()
    
    @property
    def backend_name(self) -> str:
        return "glm-tts"
    
    @property
    def model_name(self) -> str:
        return f"GLM-TTS"
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def _get_llm_dtype(self) -> str:
        """Get LLM dtype from env var GLM_TTS_DTYPE or config. Options: fp32, fp16, bf16"""
        dtype = os.environ.get("GLM_TTS_DTYPE", "").lower()
        if not dtype:
            dtype = self.config.get("llm_dtype", "fp16").lower()
        
        valid_dtypes = ["fp32", "fp16", "bf16"]
        if dtype not in valid_dtypes:
            self.logger.warning(f"Invalid GLM_TTS_DTYPE '{dtype}', defaulting to fp16. Valid: {valid_dtypes}")
            dtype = "fp16"
        return dtype
    
    def _get_llm_quantization(self) -> str:
        """Get LLM quantization from env var GLM_TTS_QUANTIZATION or config. Options: none, 4bit, 8bit"""
        quant = os.environ.get("GLM_TTS_QUANTIZATION", "").lower()
        if not quant:
            quant = self.config.get("llm_quantization", "none").lower()
        
        valid_quants = ["none", "4bit", "8bit"]
        if quant not in valid_quants:
            self.logger.warning(f"Invalid GLM_TTS_QUANTIZATION '{quant}', defaulting to none. Valid: {valid_quants}")
            quant = "none"
        return quant
    
    def _get_llm_attention(self) -> str:
        """Get LLM attention impl from env var GLM_TTS_ATTENTION or config. Options: eager, sdpa, flash_attention_2"""
        attn = os.environ.get("GLM_TTS_ATTENTION", "").lower()
        if not attn:
            attn = self.config.get("llm_attention", "eager").lower()
        
        valid_attns = ["eager", "sdpa", "flash_attention_2", "flash_attention_3"]
        if attn not in valid_attns:
            self.logger.warning(f"Invalid GLM_TTS_ATTENTION '{attn}', defaulting to eager. Valid: {valid_attns}")
            attn = "eager"
        return attn
    
    def _get_flow_dtype(self) -> str:
        """Get Flow model dtype from env var GLM_TTS_FLOW_DTYPE. Options: fp32, fp16, bf16
        
        Note: GLM-TTS source is patched (via patch_glm_tts.py) to handle dtype
        consistency, so FP16 is safe and recommended for performance.
        """
        dtype = os.environ.get("GLM_TTS_FLOW_DTYPE", "").lower()
        if not dtype:
            dtype = self.config.get("flow_dtype", "fp16").lower()  # FP16 default (patched source)
        
        valid_dtypes = ["fp32", "fp16", "bf16"]
        if dtype not in valid_dtypes:
            self.logger.warning(f"Invalid GLM_TTS_FLOW_DTYPE '{dtype}', defaulting to fp16. Valid: {valid_dtypes}")
            dtype = "fp16"
        return dtype
    
    def _get_flow_torch_dtype(self):
        """Convert flow dtype string to torch dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map[self.flow_dtype]
    
    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map[self.llm_dtype]
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create BitsAndBytesConfig based on quantization setting"""
        if self.llm_quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.llm_quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    
    def _get_special_token_ids(self, tokenize_fn):
        """Get special token IDs for GLM-TTS"""
        _special_token_ids = {
            "ats": "<|audio_0|>",
            "ate": "<|audio_32767|>",
            "boa": "<|begin_of_audio|>",
            "eoa": "<|user|>",
            "pad": "<|endoftext|>",
        }
        
        special_token_ids = {}
        endoftext_id = tokenize_fn("<|endoftext|>")[0]
        
        for k, v in _special_token_ids.items():
            ids = tokenize_fn(v)
            if len(ids) != 1:
                raise AssertionError(f"Token '{k}' ({v}) encoded to multiple tokens: {ids}")
            special_token_ids[k] = ids[0]
        
        return special_token_ids
    
    def _create_mel_extractor(self):
        """Create mel spectrogram extractor based on sample rate"""
        if self._sample_rate == 32000:
            return partial(
                mel_spectrogram,
                sampling_rate=self._sample_rate,
                hop_size=640,
                n_fft=2560,
                num_mels=80,
                win_size=2560,
                fmin=0,
                fmax=8000,
                center=False
            )
        elif self._sample_rate == 24000:
            return partial(
                mel_spectrogram,
                sampling_rate=self._sample_rate,
                hop_size=480,
                n_fft=1920,
                num_mels=80,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False
            )
        else:
            raise ValueError(f"Unsupported sample rate: {self._sample_rate}")
    
    def initialize_model(self) -> None:
        """Initialize GLM-TTS model components"""
        self.logger.info("Initializing GLM-TTS model...")
        
        model_path = Path(self._model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"GLM-TTS model path not found: {self._model_path}. "
                f"Please download models: huggingface-cli download zai-org/GLM-TTS --local-dir {self._model_path}"
            )
        
        # Check for required subdirectories
        required_dirs = ["speech_tokenizer", "llm", "flow"]
        missing = [d for d in required_dirs if not (model_path / d).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model directories in {self._model_path}: {missing}. "
                "Please ensure models are fully downloaded."
            )
        
        # 1. Load Speech Tokenizer
        self.logger.debug("Loading speech tokenizer...")
        _model, _feature_extractor = yaml_util.load_speech_tokenizer(
            str(model_path / "speech_tokenizer")
        )
        self.speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)
        
        # 2. Load Tokenizer and Frontends
        self.logger.debug("Loading tokenizer and frontends...")
        tokenizer_path = model_path / "vq32k-phoneme-tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        glm_tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True
        )
        tokenize_fn = lambda text: glm_tokenizer.encode(text)
        
        # Create mel extractor
        feat_extractor = self._create_mel_extractor()
        
        # Frontend directory
        frontend_dir = GLM_TTS_DIR / "frontend"
        
        self.frontend = TTSFrontEnd(
            tokenize_fn,
            self.speech_tokenizer,
            feat_extractor,
            str(frontend_dir / "campplus.onnx"),
            "",  # spk2info path - optional
            self.device
        )
        
        self.text_frontend = TextFrontEnd(self.use_phoneme)
        
        # 3. Load LLM (with optional vLLM engine for high-performance inference)
        self.logger.debug("Loading LLM...")
        llama_path = str(model_path / "llm")
        
        # Get configs path
        configs_dir = GLM_TTS_DIR / "configs"
        
        # Get special token IDs first (needed for both engines)
        self.special_token_ids = self._get_special_token_ids(tokenize_fn)
        
        # Choose inference engine based on GLM_TTS_ENGINE env var
        if self.llm_engine == "vllm":
            self._load_llm_vllm(llama_path)
        else:
            self._load_llm_transformers(llama_path, configs_dir)
        
        # 4. Load Flow model (Token2Wav)
        self.logger.debug("Loading Flow model...")
        flow_model = yaml_util.load_flow_model(
            str(model_path / "flow" / "flow.pt"),
            str(model_path / "flow" / "config.yaml"),
            self.device
        )
        
        # Always convert Flow model to the target dtype (defaults to FP32)
        # This is necessary because the checkpoint may be saved in FP16, but GLM-TTS
        # internally creates timestep tensors as FP32, causing dtype mismatch errors.
        flow_torch_dtype = self._get_flow_torch_dtype()
        self.logger.info(f"Setting Flow model to {self.flow_dtype} ({flow_torch_dtype})")
        flow_model = flow_model.to(flow_torch_dtype)
        
        # Set CFG rate on the flow model (controls classifier-free guidance, 0=disabled for 2x speedup)
        if hasattr(flow_model, 'inference_cfg_rate'):
            flow_model.inference_cfg_rate = self.cfg_rate
            self.logger.info(f"Set Flow CFG rate to {self.cfg_rate}" + (" (CFG disabled for 2x speedup)" if self.cfg_rate == 0 else ""))
        
        # Apply torch.compile to Flow model for additional speedup
        if self.compile_flow:
            self.logger.info("Applying torch.compile to Flow model (first inference will be slow)...")
            try:
                flow_model = torch.compile(flow_model, mode="max-autotune")
                self.logger.info("torch.compile applied to Flow model successfully (mode=max-autotune)")
            except Exception as e:
                self.logger.warning(f"torch.compile for Flow failed: {e}. Continuing without compilation.")
        
        # Load vocoder with absolute paths
        self.logger.debug("Loading vocoder...")
        self.flow = self._create_token2wav(flow_model, model_path)
        
        self.logger.info("GLM-TTS model initialization complete")
    
    def _load_llm_transformers(self, llama_path: str, configs_dir: Path) -> None:
        """Load LLM using HuggingFace transformers (default engine)."""
        self.logger.info("Using HuggingFace transformers engine")
        
        self.llm = GLMTTS(
            llama_cfg_path=os.path.join(llama_path, "config.json"),
            mode="PRETRAIN",
            lora_adapter_config=str(configs_dir / "lora_adapter_configV3.1.json"),
            spk_prompt_dict_path=str(configs_dir / "spk_prompt_dict.yaml")
        )
        
        # Build LLM loading kwargs based on configuration
        llm_kwargs = {
            "torch_dtype": self._get_torch_dtype(),
        }
        
        # Add attention implementation
        if self.llm_attention != "eager":
            llm_kwargs["attn_implementation"] = self.llm_attention
            self.logger.debug(f"Using attention implementation: {self.llm_attention}")
        
        # Add quantization config if enabled
        quant_config = self._get_quantization_config()
        if quant_config:
            llm_kwargs["quantization_config"] = quant_config
            llm_kwargs["device_map"] = "auto"  # Required for quantization
            self.logger.debug(f"Using quantization: {self.llm_quantization}")
        
        # Load the model
        self.llm.llama = LlamaForCausalLM.from_pretrained(llama_path, **llm_kwargs)
        
        # Only manually move to device if not using quantization (which uses device_map)
        if not quant_config:
            self.llm.llama = self.llm.llama.to(self.device)
        
        self.llm.llama_embedding = self.llm.llama.model.embed_tokens
        
        # Set special token IDs
        self.llm.set_runtime_vars(special_token_ids=self.special_token_ids)
        
        # Apply torch.compile if enabled (PyTorch 2.0+ JIT optimization)
        # Note: Use "default" mode, not "reduce-overhead" which uses CUDA graphs
        # that don't work well with autoregressive generation
        if self.use_torch_compile:
            self.logger.info("Applying torch.compile to LLM (first inference will be slow)...")
            try:
                # Use default mode - reduce-overhead causes CUDA graph issues with autoregressive gen
                self.llm.llama = torch.compile(self.llm.llama, mode="default")
                self.logger.info("torch.compile applied successfully (mode=default)")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    def _load_llm_vllm(self, llama_path: str) -> None:
        """Load LLM using vLLM engine for high-performance inference."""
        from app.backends.vllm_glmtts import VLLMGLMTTSWrapper, is_vllm_available
        
        if not is_vllm_available():
            self.logger.warning("vLLM not available, falling back to transformers engine")
            self.llm_engine = "transformers"
            configs_dir = Path(self._model_path).parent.parent / "GLM-TTS" / "configs"
            return self._load_llm_transformers(llama_path, configs_dir)
        
        self.logger.info("Using vLLM engine for high-performance inference")
        
        # vLLM wrapper acts as drop-in replacement for GLMTTS
        self.llm = VLLMGLMTTSWrapper(
            model_path=llama_path,
            special_token_ids=self.special_token_ids,
            dtype=self.llm_dtype,
            gpu_memory_utilization=0.25,  # Low - shared with Flow model on same GPU
            max_model_len=4096,
            quantization=self.vllm_quantization,  # fp8, awq, gptq, or None
            logger=self.logger,
        )
        
        self.logger.info("vLLM engine loaded successfully")
    
    def _create_token2wav(self, flow_model, model_path: Path):
        """Create Token2Wav with proper absolute paths for vocoder"""
        from utils.hift_util import HiFTInference
        
        # Capture outer scope vars for inner class
        compile_vocoder = self.compile_vocoder
        logger = self.logger
        flow_dtype = self._get_flow_torch_dtype()  # Capture dtype for inner class
        vocoder_engine = self.vocoder_engine
        tensorrt_engine_path = self.tensorrt_engine_path
        
        # Load TensorRT vocoder if configured
        trt_vocoder = None
        if vocoder_engine == "tensorrt":
            try:
                from app.backends.hift_tensorrt import HiFTTensorRT, is_tensorrt_available
                if not is_tensorrt_available():
                    logger.warning("TensorRT not available, falling back to PyTorch vocoder")
                elif not os.path.exists(tensorrt_engine_path):
                    logger.warning(f"TensorRT engine not found at {tensorrt_engine_path}, falling back to PyTorch vocoder")
                    logger.info("Build engine with: python scripts/export_hift_tensorrt.py")
                else:
                    logger.info(f"Loading TensorRT vocoder from {tensorrt_engine_path}...")
                    trt_vocoder = HiFTTensorRT(tensorrt_engine_path, device=str(self.device))
                    logger.info("TensorRT vocoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load TensorRT vocoder: {e}, falling back to PyTorch")
        
        class Token2WavWrapper:
            def __init__(self, flow, sample_rate: int, device: str, hift_path: str, dtype, trt_vocoder=None):
                self.device = device
                self.dtype = dtype  # Store dtype for casting float tensors
                self.flow = flow
                self.input_frame_rate = flow.input_frame_rate
                self.sample_rate = sample_rate
                self.hop_size = 480 if sample_rate == 24000 else 640
                self.trt_vocoder = trt_vocoder
                
                # Load HiFT vocoder with absolute path (skip if using TensorRT)
                if trt_vocoder is not None:
                    self.vocoder = trt_vocoder
                    logger.info("Using TensorRT accelerated vocoder")
                else:
                    self.vocoder = HiFTInference(hift_path, device=device)
                    # Apply torch.compile to vocoder for speedup
                    # Vocoder is non-autoregressive so benefits from reduce-overhead mode
                    if compile_vocoder:
                        logger.info("Applying torch.compile to HiFT vocoder...")
                        try:
                            self.vocoder.model = torch.compile(self.vocoder.model, mode="reduce-overhead")
                            logger.info("torch.compile applied to vocoder successfully (mode=reduce-overhead)")
                        except Exception as e:
                            logger.warning(f"torch.compile for vocoder failed: {e}")
            
            def token2wav_with_cache(self, token_bt, n_timesteps=10, 
                                     prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                                     prompt_feat=torch.zeros(1, 0, 80),
                                     embedding=torch.zeros(1, 192)):
                import numpy as np
                if isinstance(token_bt, (list, np.ndarray)):
                    token_bt = torch.tensor(token_bt, dtype=torch.long)[None]
                
                assert prompt_token.shape[1] != 0 and prompt_feat.shape[1] != 0
                
                # GLM-TTS source is patched to handle dtype consistently
                # Cast inputs to model dtype for safety
                mel, _ = self.flow.inference_with_cache(
                    token=token_bt.to(self.device),
                    prompt_token=prompt_token.to(self.device),
                    prompt_feat=prompt_feat.to(device=self.device, dtype=self.dtype),
                    embedding=embedding.to(device=self.device, dtype=self.dtype),
                    n_timesteps=n_timesteps,
                )
                
                # Cast mel to FP32 for vocoder (HiFT is always FP32)
                mel = mel.float()
                wav = self.vocoder(mel)
                return wav, mel
        
        hift_path = str(model_path / "hift" / "hift.pt")
        return Token2WavWrapper(flow_model, self._sample_rate, self.device, hift_path, flow_dtype, trt_vocoder)
    
    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """
        Load a voice from audio file(s).
        GLM-TTS uses 3-10 seconds of prompt audio for voice cloning.
        Also transcribes the audio to get prompt_text for better quality.
        """
        self.logger.debug(f"Loading voice: {voice_name} from {voice_path}")
        
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        if os.path.isdir(voice_path):
            paths = []
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                paths.extend([str(p) for p in Path(voice_path).glob(ext)])
            paths.sort()
            
            if not paths:
                raise ValueError(f"No audio files found in {voice_path}")
            
            voice = Voice(voice_name, paths)
        else:
            voice = Voice(voice_name, [voice_path])
        
        # Auto-transcribe to get prompt_text (crucial for GLM-TTS quality!)
        prompt_text = self._transcribe_voice(voice.file_paths[0])
        voice.metadata["prompt_text"] = prompt_text
        self.logger.debug(f"Voice {voice_name} prompt_text: '{prompt_text[:50]}...'")
        
        # Pre-extract and cache all prompt features to avoid re-extraction on each generation
        # This runs the ONNX model once at load time, not on every request
        prompt_speech_path = voice.file_paths[0]
        self.logger.debug(f"Pre-extracting prompt features for {voice_name}...")
        
        voice.metadata["prompt_speech_token"] = self.frontend._extract_speech_token([prompt_speech_path])
        voice.metadata["speech_feat"] = self.frontend._extract_speech_feat(prompt_speech_path, sample_rate=self._sample_rate)
        voice.metadata["embedding"] = self.frontend._extract_spk_embedding(prompt_speech_path)
        
        self.logger.debug(f"Cached prompt features for {voice_name}")
        
        self.voices[voice_name] = voice
        self.logger.debug(f"Loaded voice: {self.voices[voice_name]}")
    
    def _transcribe_voice(self, audio_path: str) -> str:
        """Transcribe voice sample to get prompt_text for GLM-TTS."""
        import requests
        
        # Try external STT API first (faster, better quality)
        stt_api_url = self.config.get("stt_api_url", "http://localhost:8603/v1/audio/transcriptions")
        stt_api_key = self.config.get("stt_api_key", "stt-api-key")
        
        try:
            self.logger.debug(f"Transcribing {audio_path} via STT API...")
            with open(audio_path, 'rb') as f:
                response = requests.post(
                    stt_api_url,
                    headers={"Authorization": f"Bearer {stt_api_key}"},
                    files={"file": f},
                    data={"model": "base", "language": "en"},
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    return text
                self.logger.warning(f"STT API returned empty text for {audio_path}")
            else:
                self.logger.warning(f"STT API error {response.status_code}: {response.text[:100]}")
        except requests.exceptions.ConnectionError:
            self.logger.debug("STT API not available, falling back to local Whisper")
        except Exception as e:
            self.logger.warning(f"STT API error: {e}, falling back to local Whisper")
        
    
    def _llm_forward(
        self,
        prompt_text_token,
        tts_text_token,
        prompt_speech_token,
        sampling: int = None,
        min_token_text_ratio: float = None,
        max_token_text_ratio: float = None,
        beam_size: int = None,
        temperature: float = None,
        top_p: float = None,
        repetition_penalty: float = None,
        sample_method: str = None,
    ):
        """Run LLM forward pass to generate speech tokens"""
        # Use provided params or fall back to instance defaults
        sampling = sampling if sampling is not None else self.sampling
        min_token_text_ratio = min_token_text_ratio if min_token_text_ratio is not None else self.min_token_text_ratio
        max_token_text_ratio = max_token_text_ratio if max_token_text_ratio is not None else self.max_token_text_ratio
        beam_size = beam_size if beam_size is not None else self.beam_size
        temperature = temperature if temperature is not None else 1.0
        top_p = top_p if top_p is not None else 0.8
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 0.1
        sample_method = sample_method if sample_method is not None else "ras"
        
        def _get_len(token):
            return torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
        
        prompt_text_token_len = _get_len(prompt_text_token)
        tts_text_token_len = _get_len(tts_text_token)
        prompt_speech_token_len = _get_len(prompt_speech_token)
        
        self.logger.debug(
            f"LLM input: text_token_len={tts_text_token_len.item()}, "
            f"prompt_text_len={prompt_text_token_len.item()}, "
            f"prompt_speech_len={prompt_speech_token_len.item()}, "
            f"max_tokens={int(tts_text_token_len.item() * max_token_text_ratio)}"
        )
        
        tts_speech_token = self.llm.inference(
            text=tts_text_token,
            text_len=tts_text_token_len,
            prompt_text=prompt_text_token,
            prompt_text_len=prompt_text_token_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            beam_size=beam_size,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            sample_method=sample_method,
            spk=None,
        )
        # Note: temperature, top_p, repetition_penalty are used inside ras_sampling 
        # but require modifying the GLMTTS class to pass through
        
        return tts_speech_token[0].tolist()
    
    def _flow_forward(self, token_list, prompt_speech_tokens, speech_feat, embedding):
        """Run Flow forward pass to generate audio"""
        wav, full_mel = self.flow.token2wav_with_cache(
            token_list,
            n_timesteps=self.flow_steps,  # Use configurable flow steps
            prompt_token=prompt_speech_tokens,
            prompt_feat=speech_feat,
            embedding=embedding,
        )
        return wav.detach().cpu(), full_mel
    
    def _trim_trailing_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Trim trailing silence or low-energy audio.
        
        Args:
            audio: Audio samples as numpy array
            threshold_db: Energy threshold in dB below which audio is considered silence
            
        Returns:
            Audio with trailing silence removed
        """
        if len(audio) == 0:
            return audio
        
        # Convert threshold from dB to amplitude
        threshold_amp = 10 ** (threshold_db / 20)
        
        # Find last sample above threshold using a sliding window
        window_size = int(self._sample_rate * 0.05)  # 50ms window
        
        # Scan from end to find where audio energy is above threshold
        last_active = len(audio)
        for i in range(len(audio) - window_size, 0, -window_size):
            window = audio[max(0, i):i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > threshold_amp:
                last_active = min(i + window_size + int(self._sample_rate * 0.1), len(audio))  # Keep 100ms buffer
                break
        
        if last_active < len(audio):
            self.logger.debug(
                f"Trimmed {(len(audio) - last_active) / self._sample_rate:.2f}s of trailing silence"
            )
        
        return audio[:last_active]
    
    def _validate_audio(self, audio: np.ndarray, text: str) -> np.ndarray:
        """
        Validate and clean audio based on expected speech characteristics.
        
        Uses speech rate bounds (120-180 WPM = 2-3 words/sec) to detect
        abnormally long output that likely contains garbage audio.
        
        Args:
            audio: Generated audio samples
            text: Original input text
            
        Returns:
            Validated (and possibly trimmed) audio
        """
        if len(audio) == 0:
            return audio
        
        # Use configurable thresholds (set in __init__ from env vars or config)
        # MIN_SEC_PER_WORD: Fast speech threshold (180 WPM)
        MIN_SEC_PER_WORD = 0.33
        
        word_count = max(len(text.split()), 1)
        actual_duration = len(audio) / self._sample_rate
        sec_per_word = actual_duration / word_count
        
        original_duration = actual_duration
        
        # 1. Check if duration is within expected speech rate bounds
        if sec_per_word > self.garbage_threshold:
            # Likely garbage at the end - trim to max expected duration
            max_duration = word_count * self.max_sec_per_word
            max_samples = int(max_duration * self._sample_rate)
            self.logger.warning(
                f"Audio too long: {sec_per_word:.2f}s/word (threshold: {self.garbage_threshold}s). "
                f"Trimming from {actual_duration:.1f}s to {max_duration:.1f}s"
            )
            audio = audio[:max_samples]
        elif sec_per_word < MIN_SEC_PER_WORD:
            self.logger.warning(f"Suspiciously short audio: {sec_per_word:.2f}s/word")
        
        # 2. Trim trailing silence/low-energy sections
        audio = self._trim_trailing_silence(audio, threshold_db=self.silence_threshold_db)
        
        # 3. RMS sanity check
        if len(audio) > 0:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                self.logger.warning(f"Very low audio RMS: {rms:.4f}")
        
        # Log if we modified the output
        final_duration = len(audio) / self._sample_rate
        if abs(final_duration - original_duration) > 0.1:
            self.logger.info(
                f"Audio validation: {original_duration:.1f}s -> {final_duration:.1f}s "
                f"({original_duration - final_duration:.1f}s removed)"
            )
        
        return audio

    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "zh",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using GLM-TTS.
        
        Args:
            text: Text to convert to speech
            voice_name: Name of voice to use for cloning
            language: Language code ('zh' or 'en')
            **kwargs: Additional parameters:
                - sampling: Top-k sampling value (default: config value)
                - min_token_text_ratio: Min audio tokens per text token (default: config value)
                - max_token_text_ratio: Max audio tokens per text token (default: config value)
                - beam_size: Beam search width (default: config value)
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        import time
        timings = {}
        total_start = time.perf_counter()
        
        # Get per-request parameters or use defaults
        sampling = kwargs.get("sampling", self.sampling)
        min_token_text_ratio = kwargs.get("min_token_text_ratio", self.min_token_text_ratio)
        max_token_text_ratio = kwargs.get("max_token_text_ratio", self.max_token_text_ratio)
        beam_size = kwargs.get("beam_size", self.beam_size)
        temperature = kwargs.get("temperature", self.config.get("temperature", 1.0))
        top_p = kwargs.get("top_p", self.config.get("top_p", 0.8))
        repetition_penalty = kwargs.get("repetition_penalty", self.config.get("repetition_penalty", 0.1))
        sample_method = kwargs.get("sample_method", self.config.get("sample_method", "ras"))
        
        self.logger.debug(f"Using params: sampling={sampling}, min_ratio={min_token_text_ratio}, max_ratio={max_token_text_ratio}, beam={beam_size}, temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}, method={sample_method}")
        
        # Validate inputs
        text = self.validate_text(text)
        self.validate_voice(voice_name)
        
        if language not in self.SUPPORTED_LANGUAGES:
            self.logger.warning(
                f"Language '{language}' may not be fully supported. "
                f"GLM-TTS primarily supports: {self.SUPPORTED_LANGUAGES}"
            )
        
        self.logger.debug(f"Generating speech: '{text[:50]}...' with voice {voice_name}")
        
        # Get voice prompt path and transcription
        voice = self.voices[voice_name]
        prompt_speech_path = voice.file_paths[0]
        
        # Use transcribed prompt text (crucial for quality!)
        prompt_text = voice.metadata.get("prompt_text", "")
        if not prompt_text:
            self.logger.warning(f"Voice {voice_name} has no prompt_text - quality may be poor")
        
        # === PROFILING: Text normalization ===
        t0 = time.perf_counter()
        prompt_text_normalized = self.text_frontend.text_normalize(prompt_text + " ") if prompt_text else " "
        synth_text_normalized = self.text_frontend.text_normalize(text)
        
        if self.use_phoneme:
            synth_text_normalized = self.text_frontend.g2p_infer(synth_text_normalized)
        timings['text_normalize'] = time.perf_counter() - t0
        
        # === PROFILING: Prompt feature extraction (use cached features) ===
        t0 = time.perf_counter()
        prompt_text_token = self.frontend._extract_text_token(prompt_text_normalized)
        
        # Use cached features from voice loading (avoids ONNX re-extraction on each request)
        prompt_speech_token = voice.metadata.get("prompt_speech_token")
        speech_feat = voice.metadata.get("speech_feat")
        embedding = voice.metadata.get("embedding")
        
        # Fallback to extraction if cache miss (shouldn't happen normally)
        if prompt_speech_token is None or speech_feat is None or embedding is None:
            self.logger.warning(f"Cache miss for voice {voice_name}, re-extracting features...")
            prompt_speech_token = self.frontend._extract_speech_token([prompt_speech_path])
            speech_feat = self.frontend._extract_speech_feat(prompt_speech_path, sample_rate=self._sample_rate)
            embedding = self.frontend._extract_spk_embedding(prompt_speech_path)
        
        timings['prompt_extraction'] = time.perf_counter() - t0
        
        # Prepare for inference
        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(self.device)
        
        # Split text into manageable chunks (use our custom chunk sizes)
        short_text_list = self.text_frontend.split_by_len(
            synth_text_normalized,
            min_text_len=self.min_text_len,
            max_text_len=self.max_text_len
        )
        
        outputs = []
        cache = {
            "cache_text": [prompt_text_normalized],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": self.use_cache,
        }
        
        # Track per-chunk timing
        llm_total_time = 0.0
        flow_total_time = 0.0
        total_tokens_generated = 0
        
        for chunk_idx, tts_text in enumerate(short_text_list):
            # Process text
            tts_text_tn = self.text_frontend.text_normalize(tts_text)
            if self.use_phoneme:
                tts_text_tn = self.text_frontend.g2p_infer(tts_text_tn)
            
            tts_text_token = self.frontend._extract_text_token(tts_text_tn)
            
            # Get prompt from cache
            cache_text_token = cache["cache_text_token"]
            cache_speech_token_list = cache["cache_speech_token"]
            
            # Use initial prompt
            prompt_text_token_for_llm = cache_text_token[0].to(self.device)
            prompt_speech_token_for_llm = torch.tensor(
                [cache_speech_token_list[0]], dtype=torch.int32
            ).to(self.device)
            
            # === PROFILING: LLM inference ===
            t0 = time.perf_counter()
            token_list_res = self._llm_forward(
                prompt_text_token=prompt_text_token_for_llm,
                tts_text_token=tts_text_token,
                prompt_speech_token=prompt_speech_token_for_llm,
                sampling=sampling,
                min_token_text_ratio=min_token_text_ratio,
                max_token_text_ratio=max_token_text_ratio,
                beam_size=beam_size,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                sample_method=sample_method,
            )
            llm_time = time.perf_counter() - t0
            llm_total_time += llm_time
            total_tokens_generated += len(token_list_res)
            
            self.logger.debug(
                f"[Chunk {chunk_idx+1}/{len(short_text_list)}] LLM: {len(token_list_res)} tokens in {llm_time*1000:.1f}ms "
                f"({len(token_list_res)/llm_time:.1f} tok/s)"
            )
            
            # === PROFILING: Flow inference ===
            t0 = time.perf_counter()
            output, _ = self._flow_forward(
                token_list=token_list_res,
                prompt_speech_tokens=flow_prompt_token,
                speech_feat=speech_feat,
                embedding=embedding
            )
            flow_time = time.perf_counter() - t0
            flow_total_time += flow_time
            
            self.logger.debug(
                f"[Chunk {chunk_idx+1}/{len(short_text_list)}] Flow: {flow_time*1000:.1f}ms"
            )
            
            outputs.append(output)
            
            # Update cache
            cache["cache_text"].append(tts_text_tn)
            cache["cache_text_token"].append(tts_text_token)
            cache["cache_speech_token"].append(token_list_res)
        
        timings['llm_inference'] = llm_total_time
        timings['flow_inference'] = flow_total_time
        
        # === PROFILING: Concatenation ===
        t0 = time.perf_counter()
        if outputs:
            tts_speech = torch.concat(outputs, dim=1)
            audio = tts_speech.squeeze().numpy()
        else:
            audio = np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)
        timings['concat'] = time.perf_counter() - t0
        
        # === Audio Validation: Check speech rate and trim garbage ===
        t0 = time.perf_counter()
        audio = self._validate_audio(audio, text)
        timings['validation'] = time.perf_counter() - t0
        
        # Calculate total and audio duration
        timings['total'] = time.perf_counter() - total_start
        audio_duration = len(audio) / self._sample_rate
        rtf = timings['total'] / audio_duration if audio_duration > 0 else 0
        
        # Store timings for access by API layer (exposed via headers)
        timings['audio_duration'] = audio_duration
        timings['tokens_generated'] = total_tokens_generated
        timings['rtf'] = rtf
        self._last_timings = timings
        
        # Log comprehensive profiling summary
        self.logger.info(
            f"[TTS Profiling] Text: {len(text)} chars, Audio: {audio_duration:.2f}s, RTF: {rtf:.2f}x | "
            f"Normalize: {timings['text_normalize']*1000:.1f}ms, "
            f"Prompt: {timings['prompt_extraction']*1000:.1f}ms, "
            f"LLM: {timings['llm_inference']*1000:.1f}ms ({total_tokens_generated} tokens, {total_tokens_generated/timings['llm_inference']:.1f} tok/s), "
            f"Flow: {timings['flow_inference']*1000:.1f}ms, "
            f"Total: {timings['total']*1000:.1f}ms"
        )
        
        return audio, self._sample_rate
    
    def get_last_timings(self) -> dict:
        """Get profiling timings from the last generate_speech call."""
        return getattr(self, '_last_timings', {})

