"""Qwen3-TTS voice definitions and presets."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QwenVoice:
    """Represents a Qwen3-TTS predefined speaker.
    
    Attributes:
        id: Unique lowercase identifier
        name: Display name
        description: Brief voice characterization
        native_language: Primary language of the voice
        supported_languages: All languages this voice can speak
        gender: "male" or "female"
        age_range: "young_adult", "adult", or "mature"
        style_tags: Descriptive style keywords
    """
    
    id: str
    name: str
    description: str
    native_language: str
    supported_languages: List[str]
    gender: str
    age_range: str
    style_tags: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "native_language": self.native_language,
            "supported_languages": self.supported_languages,
            "gender": self.gender,
            "age_range": self.age_range,
            "style_tags": self.style_tags,
            "backend": "qwen_tts"
        }


# All supported languages for Qwen3-TTS
QWEN_SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", 
    "German", "French", "Russian", "Portuguese", 
    "Spanish", "Italian", "Auto"
]

# Common language set shared by all speakers
_ALL_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian"
]


# Predefined speakers from Qwen3-TTS-CustomVoice
QWEN_SPEAKERS = {
    "Vivian": QwenVoice(
        id="vivian",
        name="Vivian",
        description="Bright, slightly edgy young female voice",
        native_language="Chinese",
        supported_languages=_ALL_LANGUAGES,
        gender="female",
        age_range="young_adult",
        style_tags=["bright", "edgy", "energetic"]
    ),
    "Serena": QwenVoice(
        id="serena",
        name="Serena",
        description="Warm, gentle young female voice",
        native_language="Chinese",
        supported_languages=_ALL_LANGUAGES,
        gender="female",
        age_range="young_adult",
        style_tags=["warm", "gentle", "soothing"]
    ),
    "Uncle_Fu": QwenVoice(
        id="uncle_fu",
        name="Uncle Fu",
        description="Seasoned male voice with a low, mellow timbre",
        native_language="Chinese",
        supported_languages=_ALL_LANGUAGES,
        gender="male",
        age_range="mature",
        style_tags=["deep", "mellow", "authoritative"]
    ),
    "Dylan": QwenVoice(
        id="dylan",
        name="Dylan",
        description="Youthful Beijing male voice with clear, natural timbre",
        native_language="Chinese",
        supported_languages=_ALL_LANGUAGES,
        gender="male",
        age_range="young_adult",
        style_tags=["youthful", "clear", "natural", "beijing_dialect"]
    ),
    "Eric": QwenVoice(
        id="eric",
        name="Eric",
        description="Lively Chengdu male voice with slightly husky brightness",
        native_language="Chinese",
        supported_languages=_ALL_LANGUAGES,
        gender="male",
        age_range="young_adult",
        style_tags=["lively", "husky", "sichuan_dialect"]
    ),
    "Ryan": QwenVoice(
        id="ryan",
        name="Ryan",
        description="Dynamic male voice with strong rhythmic drive",
        native_language="English",
        supported_languages=_ALL_LANGUAGES,
        gender="male",
        age_range="adult",
        style_tags=["dynamic", "rhythmic", "energetic"]
    ),
    "Aiden": QwenVoice(
        id="aiden",
        name="Aiden",
        description="Sunny American male voice with a clear midrange",
        native_language="English",
        supported_languages=_ALL_LANGUAGES,
        gender="male",
        age_range="young_adult",
        style_tags=["sunny", "american", "clear"]
    ),
    "Ono_Anna": QwenVoice(
        id="ono_anna",
        name="Ono Anna",
        description="Playful Japanese female voice with light, nimble timbre",
        native_language="Japanese",
        supported_languages=_ALL_LANGUAGES,
        gender="female",
        age_range="young_adult",
        style_tags=["playful", "light", "nimble"]
    ),
    "Sohee": QwenVoice(
        id="sohee",
        name="Sohee",
        description="Warm Korean female voice with rich emotion",
        native_language="Korean",
        supported_languages=_ALL_LANGUAGES,
        gender="female",
        age_range="young_adult",
        style_tags=["warm", "emotional", "expressive"]
    ),
}


def get_speaker_by_id(speaker_id: str) -> Optional[QwenVoice]:
    """Get speaker by ID (case-insensitive).
    
    Args:
        speaker_id: Voice ID or name to search for
        
    Returns:
        QwenVoice if found, None otherwise
    """
    speaker_id_lower = speaker_id.lower()
    for name, voice in QWEN_SPEAKERS.items():
        if voice.id == speaker_id_lower or name.lower() == speaker_id_lower:
            return voice
    return None


def get_speakers_by_language(language: str) -> List[QwenVoice]:
    """Get speakers native to a specific language.
    
    Args:
        language: Language name (e.g., "Chinese", "English")
        
    Returns:
        List of voices native to that language
    """
    return [
        voice for voice in QWEN_SPEAKERS.values()
        if voice.native_language.lower() == language.lower()
    ]


def get_speakers_by_gender(gender: str) -> List[QwenVoice]:
    """Get speakers of a specific gender.
    
    Args:
        gender: "male" or "female"
        
    Returns:
        List of voices matching the gender
    """
    return [
        voice for voice in QWEN_SPEAKERS.values()
        if voice.gender.lower() == gender.lower()
    ]
