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
try:
    from transformers import AutoTokenizer, LlamaForCausalLM
    from llm.glmtts import GLMTTS
    from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
    from utils import tts_model_util, yaml_util
    from utils.audio import mel_spectrogram
    GLM_DEPS_AVAILABLE = True
except ImportError as e:
    GLM_DEPS_AVAILABLE = False
    _import_error = str(e)


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
        self.sampling = self.config.get("sampling", 25)
        self.use_phoneme = self.config.get("use_phoneme", False)
        self.use_cache = self.config.get("use_cache", True)
        
        # Token generation limits (affects how much audio is generated per text)
        self.max_token_text_ratio = self.config.get("max_token_text_ratio", 20.0)
        self.min_token_text_ratio = self.config.get("min_token_text_ratio", 2.0)
        
        # Text chunking parameters
        self.max_text_len = self.config.get("max_text_len", 200)  # Increased from 60
        self.min_text_len = self.config.get("min_text_len", 50)   # Increased from 30
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Initializing GLMTTSBackend with model_path: {self._model_path}")
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
        
        # 3. Load LLM
        self.logger.debug("Loading LLM...")
        llama_path = str(model_path / "llm")
        
        # Get configs path
        configs_dir = GLM_TTS_DIR / "configs"
        
        self.llm = GLMTTS(
            llama_cfg_path=os.path.join(llama_path, "config.json"),
            mode="PRETRAIN",
            lora_adapter_config=str(configs_dir / "lora_adapter_configV3.1.json"),
            spk_prompt_dict_path=str(configs_dir / "spk_prompt_dict.yaml")
        )
        self.llm.llama = LlamaForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.float32
        ).to(self.device)
        self.llm.llama_embedding = self.llm.llama.model.embed_tokens
        
        # Set special token IDs
        self.special_token_ids = self._get_special_token_ids(tokenize_fn)
        self.llm.set_runtime_vars(special_token_ids=self.special_token_ids)
        
        # 4. Load Flow model (Token2Wav)
        self.logger.debug("Loading Flow model...")
        flow_model = yaml_util.load_flow_model(
            str(model_path / "flow" / "flow.pt"),
            str(model_path / "flow" / "config.yaml"),
            self.device
        )
        
        # Load vocoder with absolute paths
        self.logger.debug("Loading vocoder...")
        self.flow = self._create_token2wav(flow_model, model_path)
        
        self.logger.info("GLM-TTS model initialization complete")
    
    def _create_token2wav(self, flow_model, model_path: Path):
        """Create Token2Wav with proper absolute paths for vocoder"""
        from utils.hift_util import HiFTInference
        
        class Token2WavWrapper:
            def __init__(self, flow, sample_rate: int, device: str, hift_path: str):
                self.device = device
                self.flow = flow
                self.input_frame_rate = flow.input_frame_rate
                self.sample_rate = sample_rate
                self.hop_size = 480 if sample_rate == 24000 else 640
                
                # Load HiFT vocoder with absolute path
                self.vocoder = HiFTInference(hift_path, device=device)
            
            def token2wav_with_cache(self, token_bt, n_timesteps=10, 
                                     prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                                     prompt_feat=torch.zeros(1, 0, 80),
                                     embedding=torch.zeros(1, 192)):
                import numpy as np
                if isinstance(token_bt, (list, np.ndarray)):
                    token_bt = torch.tensor(token_bt, dtype=torch.long)[None]
                
                assert prompt_token.shape[1] != 0 and prompt_feat.shape[1] != 0
                mel, _ = self.flow.inference_with_cache(
                    token=token_bt.to(self.device),
                    prompt_token=prompt_token.to(self.device),
                    prompt_feat=prompt_feat.to(self.device),
                    embedding=embedding.to(self.device),
                    n_timesteps=n_timesteps,
                )
                
                wav = self.vocoder(mel)
                return wav, mel
        
        hift_path = str(model_path / "hift" / "hift.pt")
        return Token2WavWrapper(flow_model, self._sample_rate, self.device, hift_path)
    
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
        
        # Fallback to local Whisper
        try:
            import whisper
            if not hasattr(self, '_whisper_model'):
                self.logger.debug("Loading local Whisper model for voice transcription...")
                self._whisper_model = whisper.load_model("base")
            
            result = self._whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            self.logger.warning(f"Failed to transcribe {audio_path}: {e}. Using empty prompt_text.")
            return ""
    
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
            prompt_token=prompt_speech_tokens,
            prompt_feat=speech_feat,
            embedding=embedding,
        )
        return wav.detach().cpu(), full_mel
    
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
        
        # Text normalization
        prompt_text_normalized = self.text_frontend.text_normalize(prompt_text + " ") if prompt_text else " "
        synth_text_normalized = self.text_frontend.text_normalize(text)
        
        if self.use_phoneme:
            synth_text_normalized = self.text_frontend.g2p_infer(synth_text_normalized)
        
        # Extract tokens and features from prompt audio
        prompt_text_token = self.frontend._extract_text_token(prompt_text_normalized)
        prompt_speech_token = self.frontend._extract_speech_token([prompt_speech_path])
        speech_feat = self.frontend._extract_speech_feat(prompt_speech_path, sample_rate=self._sample_rate)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_path)
        
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
        
        for tts_text in short_text_list:
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
            
            # LLM inference
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
            
            self.logger.debug(
                f"LLM generated {len(token_list_res)} tokens for {len(tts_text)} chars "
                f"(ratio: {len(token_list_res)/len(tts_text):.1f})"
            )
            
            # Flow inference
            output, _ = self._flow_forward(
                token_list=token_list_res,
                prompt_speech_tokens=flow_prompt_token,
                speech_feat=speech_feat,
                embedding=embedding
            )
            
            outputs.append(output)
            
            # Update cache
            cache["cache_text"].append(tts_text_tn)
            cache["cache_text_token"].append(tts_text_token)
            cache["cache_speech_token"].append(token_list_res)
        
        # Concatenate outputs
        if outputs:
            tts_speech = torch.concat(outputs, dim=1)
            audio = tts_speech.squeeze().numpy()
        else:
            audio = np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)
        
        return audio, self._sample_rate
