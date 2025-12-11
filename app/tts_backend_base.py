"""
Abstract base class for TTS backends.
Defines the common interface that all TTS backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging


class Voice:
    """Common voice representation across backends"""
    def __init__(self, name: str, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Voice(name={self.name}, file_paths={self.file_paths})"


class TTSBackendBase(ABC):
    """Abstract base class for TTS backends"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.voices: Dict[str, Voice] = {}
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'xtts', 'glm-tts')"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier"""
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate of this backend"""
        pass
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes. Override in subclass."""
        return ["en"]
    
    @abstractmethod
    def initialize_model(self) -> None:
        """Initialize the TTS model. Called during __init__ or lazily."""
        pass
    
    @abstractmethod
    def load_voice(self, voice_name: str, voice_path: str) -> None:
        """
        Load a voice from file(s).
        
        Args:
            voice_name: Name identifier for the voice
            voice_path: Path to voice file or directory containing voice files
        """
        pass
    
    def load_voices(self, voices_dir: str = "data/voices") -> None:
        """
        Load all voices from the voices directory.
        Default implementation loads from subdirectories.
        Override if backend has different requirements.
        """
        from pathlib import Path
        
        voices_path = Path(voices_dir)
        if not voices_path.exists():
            self.logger.warning(f"Voices directory not found: {voices_dir}")
            return
        
        for voice_dir in voices_path.glob("*"):
            if voice_dir.is_dir():
                wav_files = list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.mp3"))
                if wav_files:
                    wav_files.sort()
                    # Load the first file by default
                    self.load_voice(voice_dir.name, str(wav_files[0]))
    
    def get_voices(self) -> List[str]:
        """Get list of available voice names"""
        return list(self.voices.keys())
    
    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "en",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice_name: Name of voice to use
            language: Language code (e.g., 'en', 'zh')
            **kwargs: Backend-specific parameters
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        pass
    
    def validate_voice(self, voice_name: str) -> None:
        """Validate that a voice exists. Raises ValueError if not found."""
        if voice_name not in self.voices:
            available = ", ".join(self.get_voices()) or "none"
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")
    
    def validate_text(self, text: str, max_length: int = 2000) -> str:
        """Validate and clean input text. Returns cleaned text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        if len(text) > max_length:
            raise ValueError(f"Text too long ({len(text)} chars). Maximum: {max_length}")
        
        return text


