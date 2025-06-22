"""
Basic functionality tests for the TTS API.
Tests core components without requiring the full TTS service.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_config_import():
    """Test that configuration module can be imported"""
    from app.config import settings, validate_configuration
    assert settings is not None
    assert validate_configuration() is True

def test_monitoring_import():
    """Test that monitoring module can be imported"""
    from app.monitoring import get_metrics, get_health
    assert get_metrics is not None
    assert get_health is not None

def test_models_import():
    """Test that models can be imported"""
    from app.models import TTSRequest, LanguageCode, EmotionType
    assert TTSRequest is not None
    assert LanguageCode is not None
    assert EmotionType is not None

def test_config_validation():
    """Test configuration validation"""
    from app.config import validate_configuration
    assert validate_configuration() is True

def test_metrics_collection():
    """Test metrics collection functionality"""
    from app.monitoring import metrics_collector
    metrics_collector.reset_metrics()
    
    # Test initial state
    metrics = metrics_collector.get_metrics()
    assert metrics["total_requests"] == 0
    assert metrics["successful_requests"] == 0
    assert metrics["failed_requests"] == 0

def test_health_monitoring():
    """Test health monitoring functionality"""
    from app.monitoring import health_monitor
    health_data = health_monitor.get_system_health()
    
    # Check that health data contains expected fields
    assert "uptime" in health_data
    assert health_data["uptime"] > 0

def test_model_validation():
    """Test Pydantic model validation"""
    from app.models import TTSRequest, LanguageCode
    
    # Test valid request
    valid_request = TTSRequest(
        text="Hello world",
        voice_name="demo_1",
        language=LanguageCode.ENGLISH
    )
    assert valid_request.text == "Hello world"
    assert valid_request.voice_name == "demo_1"
    assert valid_request.language == LanguageCode.ENGLISH

def test_environment_detection():
    """Test environment detection"""
    from app.config import is_development, is_production, is_staging
    
    # Default should be development
    assert is_development() is True
    assert is_production() is False
    assert is_staging() is False

def test_directory_creation():
    """Test that necessary directories can be created"""
    from app.config import settings
    import pathlib
    
    # Test directory creation
    pathlib.Path(settings.voice_data_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path("logs").mkdir(exist_ok=True)
    
    assert os.path.exists(settings.voice_data_dir)
    assert os.path.exists("logs")

def test_audit_logging():
    """Test audit logging functionality"""
    from app.monitoring import audit_logger
    
    # Test that audit logger can be created
    assert audit_logger is not None
    
    # Test that logs directory exists
    assert os.path.exists("logs")

if __name__ == "__main__":
    pytest.main([__file__]) 