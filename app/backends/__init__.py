"""
TTS Backend implementations.

Available backends:
- XTTSBackend: XTTS v2 using Coqui TTS (always available)
- GLMTTSBackend: GLM-TTS from Zhipu AI (requires additional dependencies)
"""

from app.backends.xtts_backend import XTTSBackend

__all__ = ["XTTSBackend"]

# Try to import GLM-TTS backend if dependencies are available
try:
    from app.backends.glm_tts_backend import GLMTTSBackend
    __all__.append("GLMTTSBackend")
except ImportError:
    # GLM-TTS dependencies not installed
    pass

