import os
import soundfile as sf
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import yaml
import torch
import random
import tempfile
import subprocess
import time
import argparse

from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig

class Voice:
    def __init__(self, name: str, audios: np.ndarray, file_path: str | list[str]):
        self.name = name
        # If given a single audio, convert to array
        if audios.ndim == 1:
            # Single audio array - wrap in list
            self.audios = np.array([audios])
        else:
            self.audios = audios
        # Convert single string path to list for consistency
        self.file_paths = [file_path] if isinstance(file_path, str) else file_path
        self.config = {}
    def __repr__(self):
        return f"Voice(name={self.name}, file_paths={self.file_paths}, audios={self.audios.shape})"

class TTSService:
    def __init__(self, logger: Optional[logging.Logger] = None, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("Logger must be an instance of logging.Logger")
        import TTS
        print("TTS version: " + TTS.__version__)

        self.logger.debug(f"({logger = }, {model_name = })")
        self.config = self.load_config()
        self.logger.debug(f"(): Config = {self.config}")

        # Model name hierarchy
        self.model_name = self._determine_model_name(model_name)
        self.logger.debug(f"(): Model name: {self.model_name}")

        # Get xtts version from model name
        self.xtts_version = self.model_name.split("xtts_v")[-1] if self.model_name and "xtts_v" in self.model_name else "2"
        self.logger.debug(f"(): XTTS version: {self.xtts_version}")

        self.model:TTS.tts.models.xtts.Xtts = None
        self.xtts_config:TTS.tts.configs.xtts_config.XttsConfig = None
        self.voices: Dict[str, Voice] = {}

        self.logger.debug(f"Initializing TTSService with model: {self.model_name}")
        self.initialize_model()

        self.load_voices()

    def _determine_model_name(self, model_name: str) -> str:
        """Determine the model name from various sources"""
        if model_name:
            return model_name
        elif os.getenv("TTS_MODEL_NAME"):
            return str(os.getenv("TTS_MODEL_NAME"))
        elif self.config.get("model_name"):
            return self.config["model_name"]
        else:
            raise ValueError("No model name provided")

    def load_config(self, config_path: str = "config.yaml") -> dict:
        """Load the config file with defaults"""
        self.logger.debug(f"Loading config from {config_path}")
        
        # Default configuration - EXACT match to working notebook
        default_config = {
            "tau": 0.75,
            "gpt_cond_len": 3,  # Working notebook value
            "top_k": 3,         # Working notebook value
            "top_p": 5,         # Working notebook value
            "decoder_iterations": -1,  # Not used in working notebook
            "use_deepspeed": False,
            "max_audio_duration": 30.0,
            "expected_wpm": 150  # Words per minute for duration estimation
        }
        
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return default_config
            
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            # Merge with defaults
            default_config.update(config)
            return default_config
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse config file: {e}, using defaults")
            return default_config

    def _set_seeds(self, seed: int = 42):
        """Set deterministic seeds"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def initialize_model(self):
        """Initialize TTS model with multiple fallback strategies"""
        self.logger.debug("Starting model initialization")

        # Set environment variable to automatically accept TOS
        os.environ["TTS_TOS"] = "y"
        
        # Check if model already exists
        model_dir = os.path.expanduser(f"~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
        model_path = f"{model_dir}/model.pth"
        config_path = f"{model_dir}/config.json"
        vocab_path = f"{model_dir}/vocab.json"
        
        # Only download if model doesn't exist
        if not os.path.exists(model_path):
            self.logger.info("Model not found, downloading...")
            try:
                # Try to download with TOS acceptance
                download_result = ModelManager().download_model(self.model_name)
                model_dir = str(download_result[0])
                model_path = f"{model_dir}/model.pth"
                config_path = f"{model_dir}/config.json"
                vocab_path = f"{model_dir}/vocab.json"
            except Exception as e:
                self.logger.error(f"Failed to download model: {e}")
                raise
        else:
            self.logger.info("Model already exists, using cached version")
        
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(config_path)

        use_deepspeed = self.config.get("use_deepspeed", False)
        use_deepspeed = False
        self.logger.debug(f"({use_deepspeed = })")

        self.model = Xtts.init_from_config(self.xtts_config)
        self.model.load_checkpoint(
            self.xtts_config,
            checkpoint_path=model_path,
            vocab_path=vocab_path,
            eval=True,
            use_deepspeed=use_deepspeed
        )

        # Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.logger.debug("Model moved to CUDA")
        else:
            self.logger.warning("CUDA not available, using CPU. This will be slower.")
        self.model.eval()
        self.logger.debug("Model initialized successfully")

    def load_voice(self, voice_name: str, voice_path: str):
        """Load a voice file into memory"""
        self.logger.debug(f"({voice_name = }, {voice_path = })")
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        if os.path.isdir(voice_path):
            # Load all wav files in the directory
            audios = []
            paths = []
            for wav_path in Path(voice_path).glob("*.wav"):
                self.logger.debug(f"Loading audio file: {wav_path}")
                audio, sample_rate = sf.read(wav_path)
                audios.append(audio)
                paths.append(str(wav_path))
            audios = np.array(audios)
            self.voices[voice_name] = Voice(voice_name, audios, paths)
        else:
            self.logger.debug(f"Loading single audio file: {voice_path}")
            audio, sample_rate = sf.read(voice_path)
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError(f"Empty audio file: {voice_path}")
            if sample_rate != 22050:
                self.logger.warning(f"Audio sample rate is {sample_rate}, expected 22050")
            
            audios = np.array([audio])
            file_paths = [voice_path]
            self.voices[voice_name] = Voice(voice_name, audios, file_paths)

        self.logger.debug(f"Loaded voice: {self.voices[voice_name]}")

    def load_voices(self):
        """Load all voices in the data/voices directory"""
        voices_dir = Path("data/voices")
        if not voices_dir.exists():
            self.logger.warning(f"Voices directory not found: {voices_dir}")
            return
            
        for voice_dir in voices_dir.glob("*"):
            if voice_dir.is_dir():
                wav_files = list(voice_dir.glob("*.wav"))
                if wav_files:
                    # Sort files to ensure consistent selection, prioritize _01.wav files (like working notebook)
                    wav_files.sort()
                    selected_file = wav_files[0]
                    
                    # For batman specifically, prioritize batman_01.wav (used in working notebook)
                    if voice_dir.name == "batman":
                        batman_01 = voice_dir / "batman_01.wav"
                        if batman_01.exists():
                            selected_file = batman_01
                            self.logger.debug(f"Using batman_01.wav for working notebook compatibility")
                    
                    self.load_voice(voice_dir.name, str(selected_file))

    def get_voices(self):
        """Get all voices"""
        return list(self.voices.keys())

    def _estimate_duration(self, text: str) -> float:
        """Estimate expected audio duration for text"""
        word_count = len(text.split())
        wpm = self.config.get("expected_wpm", 150)
        duration = (word_count / wpm) * 60 + 1.0  # Add 1s buffer
        return min(duration, self.config.get("max_audio_duration", 15.0))

    def _validate_and_limit_audio(self, wav: np.ndarray, text: str, sample_rate: int = 24000) -> np.ndarray:
        """Validate and limit audio duration"""
        if wav.size == 0:
            raise ValueError("Generated audio is empty")
        
        duration = len(wav) / sample_rate
        expected_duration = self._estimate_duration(text)
        max_duration = max(expected_duration * 3, 30.0)  # Allow 3x expected or min 10s
        
        if duration > max_duration:
            self.logger.warning(f"Audio too long ({duration:.2f}s)") #, truncating to {max_duration:.2f}s")
            #max_samples = int(max_duration * sample_rate)
            #wav = wav[:max_samples]
        
        return wav

    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        tau: float = -1,
        gpt_cond_len: int = -1,
        top_k: int = -1,
        top_p: int = -1,
        decoder_iterations: int = -1,
        split_sentences: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with multiple fallback methods"""
        
        # Validate inputs
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if voice_name not in self.voices:
            raise ValueError(f"Voice not found: {voice_name}")
        
        # Set parameters
        tau = tau if tau != -1 else self.config.get("tau", 0.75)
        gpt_cond_len = gpt_cond_len if gpt_cond_len != -1 else self.config.get("gpt_cond_len", 3)
        top_k = top_k if top_k != -1 else self.config.get("top_k", 3)
        top_p = top_p if top_p != -1 else self.config.get("top_p", 5)
        
        
        # Set deterministic seeds
        #self._set_seeds()
        
        speaker_wav = self.voices[voice_name].file_paths[0]
        self.logger.debug(f"Generating speech: '{text[:50]}...'")

        outputs = self.model.synthesize(
            text,
            self.xtts_config,
            speaker_wav=speaker_wav,
            gpt_cond_len=gpt_cond_len,
            language=language,
            top_k=top_k,
            top_p=5,
            #decoder_iterations=10,
            #repetition_penalty=.75
        )

        wav = outputs["wav"]
        return wav, 24000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TTS Service")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("--voice", default="batman", help="Voice to use (default: batman)")
    parser.add_argument("--output", default="output.wav", help="Output audio file (default: output.wav)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize TTS service
        logger.info("Initializing TTS service...")
        tts_service = TTSService(logger=logger)
        
        # Load voices
        logger.info("Loading voices...")
        tts_service.load_voices()
        
        available_voices = tts_service.get_voices()
        logger.info(f"Available voices: {available_voices}")
        
        # Check if requested voice is available
        if args.voice not in available_voices:
            if available_voices:
                logger.warning(f"Voice '{args.voice}' not found. Using '{available_voices[0]}'")
                voice_to_use = available_voices[0]
            else:
                logger.error("No voices available!")
                exit(1)
        else:
            voice_to_use = args.voice
        
        # Generate speech
        logger.info(f"Generating speech with voice '{voice_to_use}' for text: '{args.text}'")
        audio_data, sample_rate = tts_service.generate_speech(
            text=args.text,
            voice_name=voice_to_use,
            language=args.language
        )
        
        # Save audio to file
        logger.info(f"Saving audio to {args.output}")
        sf.write(args.output, audio_data, sample_rate)
        
        logger.info(f"Successfully generated speech and saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


