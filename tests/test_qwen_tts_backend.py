"""Unit tests for Qwen3-TTS backend."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.backends.qwen_tts_config import QwenTTSConfig
from app.backends.qwen_voices import (
    QWEN_SPEAKERS, 
    QwenVoice, 
    get_speaker_by_id,
    get_speakers_by_language,
    get_speakers_by_gender
)


class TestQwenTTSConfig:
    """Tests for QwenTTSConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QwenTTSConfig()
        assert config.model_size == "1.7B"
        assert config.enable_custom_voice is True
        assert config.enable_voice_design is True
        assert config.enable_voice_clone is True
        assert config.dtype == "bfloat16"
        assert config.default_speaker == "Vivian"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config = QwenTTSConfig.from_dict({
            "model_size": "0.6B",
            "enable_voice_design": False,
            "dtype": "float16",
            "invalid_key": "should be ignored"
        })
        assert config.model_size == "0.6B"
        assert config.enable_voice_design is False
        assert config.dtype == "float16"
    
    def test_config_validation_valid(self):
        """Test validation passes for valid config."""
        config = QwenTTSConfig(model_size="1.7B", dtype="bfloat16")
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_model_size(self):
        """Test validation fails for invalid model size."""
        config = QwenTTSConfig(model_size="2.0B")
        with pytest.raises(ValueError, match="Invalid model_size"):
            config.validate()
    
    def test_config_validation_invalid_dtype(self):
        """Test validation fails for invalid dtype."""
        config = QwenTTSConfig(dtype="int8")
        with pytest.raises(ValueError, match="Invalid dtype"):
            config.validate()
    
    def test_config_validation_voice_design_0_6b(self):
        """Test validation fails when enabling voice_design with 0.6B model."""
        config = QwenTTSConfig(model_size="0.6B", enable_voice_design=True)
        with pytest.raises(ValueError, match="VoiceDesign not available"):
            config.validate()
    
    def test_config_validation_0_6b_without_voice_design(self):
        """Test validation passes for 0.6B without voice_design."""
        config = QwenTTSConfig(model_size="0.6B", enable_voice_design=False)
        config.validate()  # Should not raise


class TestQwenVoices:
    """Tests for voice definitions."""
    
    def test_all_speakers_defined(self):
        """Test all 9 speakers are defined."""
        assert len(QWEN_SPEAKERS) == 9
        expected_names = [
            "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
            "Ryan", "Aiden", "Ono_Anna", "Sohee"
        ]
        for name in expected_names:
            assert name in QWEN_SPEAKERS
    
    def test_speaker_structure(self):
        """Test speaker objects have correct structure."""
        vivian = QWEN_SPEAKERS["Vivian"]
        assert isinstance(vivian, QwenVoice)
        assert vivian.id == "vivian"
        assert vivian.name == "Vivian"
        assert vivian.gender == "female"
        assert "Chinese" in vivian.supported_languages
        assert "English" in vivian.supported_languages
    
    def test_speaker_languages(self):
        """Test all speakers support Chinese and English."""
        for voice in QWEN_SPEAKERS.values():
            assert "Chinese" in voice.supported_languages
            assert "English" in voice.supported_languages
    
    def test_speaker_to_dict(self):
        """Test speaker serialization to dict."""
        voice_dict = QWEN_SPEAKERS["Ryan"].to_dict()
        assert voice_dict["id"] == "ryan"
        assert voice_dict["name"] == "Ryan"
        assert voice_dict["backend"] == "qwen_tts"
        assert "style_tags" in voice_dict
    
    def test_get_speaker_by_id_exact(self):
        """Test getting speaker by exact ID."""
        voice = get_speaker_by_id("vivian")
        assert voice is not None
        assert voice.name == "Vivian"
    
    def test_get_speaker_by_id_case_insensitive(self):
        """Test getting speaker by ID is case-insensitive."""
        voice = get_speaker_by_id("VIVIAN")
        assert voice is not None
        assert voice.name == "Vivian"
    
    def test_get_speaker_by_name(self):
        """Test getting speaker by display name."""
        voice = get_speaker_by_id("Uncle_Fu")
        assert voice is not None
        assert voice.id == "uncle_fu"
    
    def test_get_speaker_not_found(self):
        """Test None returned for unknown speaker."""
        voice = get_speaker_by_id("NonexistentSpeaker")
        assert voice is None
    
    def test_get_speakers_by_language_chinese(self):
        """Test filtering speakers by Chinese language."""
        chinese_speakers = get_speakers_by_language("Chinese")
        assert len(chinese_speakers) >= 5
        for voice in chinese_speakers:
            assert voice.native_language == "Chinese"
    
    def test_get_speakers_by_language_english(self):
        """Test filtering speakers by English language."""
        english_speakers = get_speakers_by_language("English")
        assert len(english_speakers) >= 2
        for voice in english_speakers:
            assert voice.native_language == "English"
    
    def test_get_speakers_by_gender_female(self):
        """Test filtering speakers by female gender."""
        female_speakers = get_speakers_by_gender("female")
        assert len(female_speakers) >= 4
        for voice in female_speakers:
            assert voice.gender == "female"
    
    def test_get_speakers_by_gender_male(self):
        """Test filtering speakers by male gender."""
        male_speakers = get_speakers_by_gender("male")
        assert len(male_speakers) >= 5
        for voice in male_speakers:
            assert voice.gender == "male"


class TestQwenTTSBackend:
    """Tests for QwenTTSBackend class."""
    
    @pytest.fixture
    def config(self):
        """Fixture for basic config."""
        return QwenTTSConfig(
            enable_custom_voice=True,
            enable_voice_design=False,
            enable_voice_clone=False
        )
    
    @pytest.fixture
    def mock_model(self):
        """Fixture for mocked Qwen model."""
        model = Mock()
        model.generate_custom_voice.return_value = (
            [np.zeros(24000, dtype=np.float32)],  # 1 second at 24kHz
            24000
        )
        return model
    
    def test_backend_import_without_qwen_tts(self):
        """Test backend can be imported even without qwen-tts package."""
        from app.backends.qwen_tts import QwenTTSBackend
        assert QwenTTSBackend is not None
    
    def test_backend_creation(self, config):
        """Test backend can be created."""
        from app.backends.qwen_tts import QwenTTSBackend
        backend = QwenTTSBackend(config=config.__dict__)
        
        assert backend.backend_name == "qwen-tts"
        assert backend.sample_rate == 24000
        assert "Chinese" in backend.supported_languages
    
    def test_backend_with_typed_config(self):
        """Test backend with typed QwenTTSConfig."""
        from app.backends.qwen_tts import QwenTTSBackend
        config = QwenTTSConfig(model_size="0.6B", enable_voice_design=False)
        backend = QwenTTSBackend(qwen_config=config)
        
        assert backend.qwen_config.model_size == "0.6B"
        assert "Qwen3-TTS-0.6B" in backend.model_name
    
    def test_generate_speech_not_initialized(self, config):
        """Test error when calling generate_speech before initialize."""
        from app.backends.qwen_tts import QwenTTSBackend
        backend = QwenTTSBackend(config=config.__dict__)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.generate_speech("Hello", "Vivian")
    
    def test_validate_text(self, config):
        """Test text validation."""
        from app.backends.qwen_tts import QwenTTSBackend
        backend = QwenTTSBackend(config=config.__dict__)
        
        # Valid text
        result = backend.validate_text("Hello world")
        assert result == "Hello world"
        
        # Empty text
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.validate_text("")
        
        # Text too long
        with pytest.raises(ValueError, match="too long"):
            backend.validate_text("x" * 3000)
    
    def test_get_predefined_speakers(self, config):
        """Test getting list of predefined speakers."""
        from app.backends.qwen_tts import QwenTTSBackend
        backend = QwenTTSBackend(config=config.__dict__)
        
        speakers = backend.get_predefined_speakers()
        assert len(speakers) == 9
        assert all(isinstance(s, QwenVoice) for s in speakers)
    
    def test_get_speaker_info(self, config):
        """Test getting speaker info by name."""
        from app.backends.qwen_tts import QwenTTSBackend
        backend = QwenTTSBackend(config=config.__dict__)
        
        info = backend.get_speaker_info("Ryan")
        assert info is not None
        assert info.native_language == "English"


class TestQwenTTSBackendWithMocks:
    """Tests with mocked qwen-tts package."""
    
    @pytest.fixture
    def mock_qwen_tts(self):
        """Fixture that mocks the qwen-tts package."""
        mock_model = MagicMock()
        mock_model.generate_custom_voice.return_value = (
            [np.zeros(24000, dtype=np.float32)],
            24000
        )
        mock_model.generate_voice_design.return_value = (
            [np.zeros(24000, dtype=np.float32)],
            24000
        )
        
        mock_tokenizer = MagicMock()
        
        with patch.dict('sys.modules', {
            'qwen_tts': MagicMock(
                Qwen3TTSModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
                Qwen3TTSTokenizer=MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer))
            )
        }):
            # Reload the module to pick up the mock
            import importlib
            from app.backends import qwen_tts
            importlib.reload(qwen_tts)
            qwen_tts._qwen_tts_available = True
            qwen_tts.Qwen3TTSModel = MagicMock(from_pretrained=MagicMock(return_value=mock_model))
            qwen_tts.Qwen3TTSTokenizer = MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer))
            
            yield mock_model
    
    @pytest.mark.skip(reason="Requires full mock setup - run with actual qwen-tts installed")
    def test_initialize_model(self, mock_qwen_tts):
        """Test model initialization with mocked dependencies."""
        from app.backends.qwen_tts import QwenTTSBackend
        
        config = QwenTTSConfig(
            enable_custom_voice=True,
            enable_voice_design=False,
            enable_voice_clone=False
        )
        backend = QwenTTSBackend(qwen_config=config)
        backend.initialize_model()
        
        assert backend._initialized
        assert "custom_voice" in backend._models
