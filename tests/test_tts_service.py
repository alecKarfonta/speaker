import pytest
import numpy as np
from pathlib import Path
from tts_api.app.xtts_service import TTSService
import logging

@pytest.fixture
def tts_service():
    """Fixture to create a TTSService instance for testing"""
    logger = logging.getLogger("test_logger")
    service = TTSService(logger)
    return service

@pytest.fixture
def sample_voice_file(tmp_path):
    """Create a temporary sample voice file for testing"""
    import soundfile as sf
    
    # Create a simple sine wave as test audio
    sample_rate = 22050
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    voice_path = tmp_path / "test_voice.wav"
    sf.write(voice_path, audio, sample_rate)
    return str(voice_path)

def test_tts_service_initialization(tts_service):
    """Test that the service initializes correctly"""
    assert tts_service is not None
    assert tts_service.model is not None
    assert tts_service.xtts_config is not None

def test_load_voice(tts_service, sample_voice_file):
    """Test loading a voice file"""
    voice_name = "test_voice"
    tts_service.load_voice(voice_name, sample_voice_file)
    assert voice_name in tts_service.voices
    assert isinstance(tts_service.voices[voice_name], np.ndarray)

def test_generate_speech(tts_service, sample_voice_file):
    """Test speech generation"""
    # Load test voice
    voice_name = "test_voice"
    tts_service.load_voice(voice_name, sample_voice_file)
    
    # Generate speech
    text = "Hello, this is a test."
    audio, sample_rate = tts_service.generate_speech(
        text=text,
        voice_name=voice_name,
        language="en",
        tau=0.3
    )
    
    assert isinstance(audio, np.ndarray)
    assert isinstance(sample_rate, int)
    assert len(audio) > 0
    assert sample_rate > 0

def test_invalid_voice_name(tts_service):
    """Test error handling for invalid voice name"""
    with pytest.raises(ValueError, match="Voice not found: nonexistent_voice"):
        tts_service.generate_speech(
            text="Test",
            voice_name="nonexistent_voice"
        )

def test_invalid_voice_path(tts_service):
    """Test error handling for invalid voice file path"""
    with pytest.raises(FileNotFoundError):
        tts_service.load_voice("test_voice", "nonexistent_path.wav")