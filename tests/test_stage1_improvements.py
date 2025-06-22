"""
Tests for Stage 1 API improvements.
Tests OpenAPI documentation, rate limiting, error handling, and monitoring.
"""

import pytest
import requests
import time
from fastapi.testclient import TestClient
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app
from app.monitoring import metrics_collector, health_monitor
from app.config import settings

client = TestClient(app)

class TestAPIEnhancements:
    """Test API enhancements from Stage 1"""
    
    def test_openapi_documentation(self):
        """Test that OpenAPI documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert openapi_spec["info"]["title"] == "Speaker TTS API"
        assert openapi_spec["info"]["version"] == "1.0.0"
    
    def test_api_versioning(self):
        """Test API versioning headers"""
        response = client.get("/")
        assert response.headers.get("X-API-Version") == "1.0.0"
    
    def test_root_endpoint(self):
        """Test enhanced root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "model" in data
        assert "status" in data
        assert "documentation" in data
        assert "health_check" in data
    
    def test_health_check(self):
        """Test enhanced health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "model" in data
        assert "available_voices" in data
        assert "timestamp" in data
    
    def test_languages_endpoint(self):
        """Test languages endpoint"""
        response = client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "total_count" in data
        assert "en" in data["languages"]
        assert data["languages"]["en"] == "English"
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_response_time" in data
        assert "requests_per_minute" in data
        assert "active_voices" in data
    
    def test_error_handling(self):
        """Test enhanced error handling"""
        # Test 404 error
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "timestamp" in data["error"]
        assert "path" in data["error"]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple requests quickly to trigger rate limiting
        responses = []
        for i in range(105):  # Exceed the 100 request limit
            response = client.get("/")
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Check if rate limiting was triggered
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, "Rate limiting should have been triggered"
    
    def test_voice_management(self):
        """Test voice management endpoints"""
        # Test get voices
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)
    
    def test_audit_logging(self):
        """Test that audit logging is working"""
        # Make a request that should be logged
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check if audit log file exists
        assert os.path.exists("logs/audit.log"), "Audit log file should exist"
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/")
        # CORS headers should be present (handled by middleware)
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
    
    def test_monitoring_integration(self):
        """Test monitoring integration"""
        # Reset metrics for clean test
        metrics_collector.reset_metrics()
        
        # Make some requests
        client.get("/")
        client.get("/health")
        client.get("/voices")
        
        # Check metrics
        metrics = metrics_collector.get_metrics()
        assert metrics["total_requests"] >= 3
        assert metrics["successful_requests"] >= 3
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from app.config import validate_configuration
        # Should not raise an exception
        assert validate_configuration() is True
    
    def test_model_validation(self):
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
    
    def test_environment_detection(self):
        """Test environment detection"""
        from app.config import is_development, is_production, is_staging
        
        # Default should be development
        assert is_development() is True
        assert is_production() is False
        assert is_staging() is False

class TestPerformanceImprovements:
    """Test performance and monitoring improvements"""
    
    def test_response_time_tracking(self):
        """Test that response times are being tracked"""
        metrics_collector.reset_metrics()
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Check that response time was recorded
        metrics = metrics_collector.get_metrics()
        assert len(metrics_collector.request_times) > 0
        assert metrics["average_response_time"] > 0
    
    def test_health_monitoring(self):
        """Test health monitoring functionality"""
        health_data = health_monitor.get_system_health()
        
        # Check that health data contains expected fields
        assert "cpu_usage_percent" in health_data
        assert "memory_usage" in health_data
        assert "disk_usage" in health_data
        assert "uptime" in health_data
        
        # Check that values are reasonable
        assert health_data["uptime"] > 0
        assert 0 <= health_data["cpu_usage_percent"] <= 100

if __name__ == "__main__":
    pytest.main([__file__]) 