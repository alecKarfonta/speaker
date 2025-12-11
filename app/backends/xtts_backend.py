"""
XTTS v2 TTS Backend implementation.
Wraps Coqui TTS XTTS v2 model for text-to-speech synthesis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import yaml
import torch
import soundfile as sf

from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig

from app.tts_backend_base import TTSBackendBase, Voice


class XTTSBackend(TTSBackendBase):
    """XTTS v2 TTS Backend using Coqui TTS"""
    
    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Supported languages for XTTS v2
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh", "ja", "ko", "hu", "hi"
    ]
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ):
        super().__init__(logger, config)
        
        self._model_name = model_name or self.config.get("model_name") or self.DEFAULT_MODEL
        self.model: Optional[Xtts] = None
        self.xtts_config: Optional[XttsConfig] = None
        self._sample_rate = 24000
        
        # Load backend-specific config
        self._load_backend_config()
        
        self.logger.debug(f"Initializing XTTSBackend with model: {self._model_name}")
        self.initialize_model()
        self.load_voices()
    
    def _load_backend_config(self, config_path: str = "config.yaml"):
        """Load XTTS-specific configuration"""
        default_config = {
            "tau": 0.75,
            "gpt_cond_len": 3,
            "top_k": 3,
            "top_p": 5,
            "use_deepspeed": False,
            "max_audio_duration": 30.0,
            "expected_wpm": 150
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f) or {}
                default_config.update(file_config)
            except yaml.YAMLError as e:
                self.logger.error(f"Failed to parse config file: {e}")
        
        self.config.update(default_config)
    
    @property
    def backend_name(self) -> str:
        return "xtts"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES
    
    def initialize_model(self) -> None:
        """Initialize the XTTS v2 model"""
        self.logger.debug("Starting XTTS model initialization")
        
        # Set environment variable to automatically accept TOS
        os.environ["TTS_TOS"] = "y"
        
        # Check if model already exists
        model_dir = os.path.expanduser(
            f"~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        )
        model_path = f"{model_dir}/model.pth"
        config_path = f"{model_dir}/config.json"
        vocab_path = f"{model_dir}/vocab.json"
        
        # Download if model doesn't exist
        if not os.path.exists(model_path):
            self.logger.info("Model not found, downloading...")
            try:
                download_result = ModelManager().download_model(self._model_name)
                model_dir = str(download_result[0])
                model_path = f"{model_dir}/model.pth"
                config_path = f"{model_dir}/config.json"
                vocab_path = f"{model_dir}/vocab.json"
            except Exception as e:
                self.logger.error(f"Failed to download model: {e}")
                raise
        else:
            self.logger.info("Model already exists, using cached version")
        
        # Load config and model
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(config_path)
        
        use_deepspeed = self.config.get("use_deepspeed", False)
        self.logger.debug(f"use_deepspeed = {use_deepspeed}")
        
        self.model = Xtts.init_from_config(self.xtts_config)
        self.model.load_checkpoint(
            self.xtts_config,
            checkpoint_path=model_path,
            vocab_path=vocab_path,
            eval=True,
            use_deepspeed=use_deepspeed
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.logger.debug("Model moved to CUDA")
        else:
            self.logger.warning("CUDA not available, using CPU. This will be slower.")
        
        self.model.eval()
        self.logger.debug("XTTS model initialized successfully")
    
    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """Load a voice file into memory"""
        self.logger.debug(f"Loading voice: {voice_name} from {voice_path}")
        
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        if os.path.isdir(voice_path):
            # Load all wav files in the directory
            paths = []
            for wav_path in Path(voice_path).glob("*.wav"):
                paths.append(str(wav_path))
            paths.sort()
            
            if not paths:
                raise ValueError(f"No .wav files found in {voice_path}")
            
            self.voices[voice_name] = Voice(voice_name, paths)
        else:
            # Single file
            self.voices[voice_name] = Voice(voice_name, [voice_path])
        
        self.logger.debug(f"Loaded voice: {self.voices[voice_name]}")
    
    def load_voices(self, voices_dir: str = "data/voices") -> None:
        """Load all voices from the voices directory"""
        voices_path = Path(voices_dir)
        if not voices_path.exists():
            self.logger.warning(f"Voices directory not found: {voices_dir}")
            return
        
        for voice_dir in voices_path.glob("*"):
            if voice_dir.is_dir():
                wav_files = list(voice_dir.glob("*.wav"))
                if wav_files:
                    wav_files.sort()
                    selected_file = wav_files[0]
                    self.load_voice(voice_dir.name, str(selected_file))
    
    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        tau: float = -1,
        gpt_cond_len: int = -1,
        top_k: int = -1,
        top_p: int = -1,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using XTTS v2"""
        
        # Validate inputs
        text = self.validate_text(text)
        self.validate_voice(voice_name)
        
        # Get parameters from config or use provided values
        tau = tau if tau != -1 else self.config.get("tau", 0.75)
        gpt_cond_len = gpt_cond_len if gpt_cond_len != -1 else self.config.get("gpt_cond_len", 3)
        top_k = top_k if top_k != -1 else self.config.get("top_k", 3)
        top_p = top_p if top_p != -1 else self.config.get("top_p", 5)
        
        speaker_wav = self.voices[voice_name].file_paths[0]
        self.logger.debug(f"Generating speech: '{text[:50]}...' with voice {voice_name}")
        
        outputs = self.model.synthesize(
            text,
            self.xtts_config,
            speaker_wav=speaker_wav,
            gpt_cond_len=gpt_cond_len,
            language=language,
            top_k=top_k,
            top_p=top_p,
        )
        
        wav = outputs["wav"]
        return wav, self._sample_rate


