"""
TTS Backend implementations.

Available backends:
- GLMTTSBackend: GLM-TTS from Zhipu AI
"""

__all__ = []

# Try to import GLM-TTS backend if dependencies are available
try:
    from app.backends.glm_tts_backend import GLMTTSBackend
    __all__.append("GLMTTSBackend")
except ImportError as e:
    # GLM-TTS dependencies not installed
    import logging
    logging.getLogger(__name__).warning(f"GLM-TTS backend not available: {e}")

# Try to import XTTS backend if coqui-tts is installed
try:
    from app.backends.xtts_backend import XTTSBackend
    __all__.append("XTTSBackend")
except ImportError:
    # coqui-tts not installed, skip XTTS
    pass
