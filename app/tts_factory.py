"""
TTS Backend Factory.
Creates and manages TTS backend instances based on configuration.
"""

import os
import logging
from typing import Dict, Type, Optional, Any

from app.tts_backend_base import TTSBackendBase


class TTSBackendFactory:
    """Factory for creating TTS backend instances"""
    
    _backends: Dict[str, Type[TTSBackendBase]] = {}
    _logger = logging.getLogger("TTSBackendFactory")
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[TTSBackendBase]) -> None:
        """Register a backend class with a name"""
        cls._backends[name] = backend_class
        cls._logger.debug(f"Registered backend: {name}")
    
    @classmethod
    def create_backend(
        cls,
        backend_name: str,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TTSBackendBase:
        """
        Create a TTS backend instance.
        
        Args:
            backend_name: Name of the backend to create ('xtts', 'glm-tts')
            logger: Logger instance to use
            config: Configuration dictionary
            **kwargs: Additional arguments passed to backend constructor
            
        Returns:
            Initialized TTS backend instance
            
        Raises:
            ValueError: If backend_name is not registered
        """
        if backend_name not in cls._backends:
            available = ", ".join(cls._backends.keys()) or "none"
            raise ValueError(
                f"Unknown backend: '{backend_name}'. Available backends: {available}"
            )
        
        backend_class = cls._backends[backend_name]
        cls._logger.info(f"Creating backend: {backend_name}")
        
        return backend_class(logger=logger, config=config, **kwargs)
    
    @classmethod
    def available_backends(cls) -> list:
        """Get list of available backend names"""
        return list(cls._backends.keys())
    
    @classmethod
    def is_backend_available(cls, backend_name: str) -> bool:
        """Check if a backend is registered"""
        return backend_name in cls._backends


def register_default_backends():
    """Register the default backends"""
    # Always register XTTS
    from app.backends.xtts_backend import XTTSBackend
    TTSBackendFactory.register_backend("xtts", XTTSBackend)
    
    # Try to register GLM-TTS if dependencies are available
    try:
        from app.backends.glm_tts_backend import GLMTTSBackend
        TTSBackendFactory.register_backend("glm-tts", GLMTTSBackend)
    except ImportError as e:
        logging.getLogger("TTSBackendFactory").warning(
            f"GLM-TTS backend not available (missing dependencies): {e}"
        )


def get_backend_from_env(
    logger: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None
) -> TTSBackendBase:
    """
    Get a TTS backend based on environment variable TTS_BACKEND.
    Defaults to 'xtts' if not set.
    """
    # Register backends if not already done
    if not TTSBackendFactory.available_backends():
        register_default_backends()
    
    backend_name = os.environ.get("TTS_BACKEND", "xtts").lower()
    return TTSBackendFactory.create_backend(backend_name, logger=logger, config=config)


