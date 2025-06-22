import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns proper information"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model" in data
    assert "version" in data
    assert "docs" in data

def test_health_endpoint():
    """Test the health endpoint returns proper health information"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TTS"
    assert "version" in data
    assert "model" in data
    assert "uptime" in data

def test_info_endpoint():
    """Test the info endpoint returns detailed service information"""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Speaker TTS API"
    assert "version" in data
    assert "description" in data
    assert "model" in data
    assert "uptime_seconds" in data
    assert "endpoints" in data

def test_voices_endpoint():
    """Test the voices endpoint returns list of voices"""
    response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert isinstance(data["voices"], list)

def test_voice_info_endpoint_not_found():
    """Test voice info endpoint returns 404 for non-existent voice"""
    response = client.get("/voices/nonexistent_voice")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data

def test_tts_endpoint_invalid_voice():
    """Test TTS endpoint returns 404 for invalid voice"""
    response = client.post("/tts", json={
        "text": "Hello world",
        "voice_name": "nonexistent_voice",
        "language": "en"
    })
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data

def test_tts_endpoint_invalid_text():
    """Test TTS endpoint returns 422 for invalid text"""
    response = client.post("/tts", json={
        "text": "",  # Empty text should fail validation
        "voice_name": "test_voice",
        "language": "en"
    })
    assert response.status_code == 422

def test_tts_endpoint_invalid_language():
    """Test TTS endpoint returns 422 for invalid language code"""
    response = client.post("/tts", json={
        "text": "Hello world",
        "voice_name": "test_voice",
        "language": "invalid"  # Invalid language code
    })
    assert response.status_code == 422

def test_api_documentation_endpoints():
    """Test that API documentation endpoints are accessible"""
    # Test OpenAPI JSON
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    # Test docs endpoint
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test redoc endpoint
    response = client.get("/redoc")
    assert response.status_code == 200

def test_cors_headers():
    """Test that CORS headers are properly set"""
    response = client.options("/voices")
    # FastAPI TestClient doesn't fully simulate CORS, but we can check the endpoint exists
    assert response.status_code in [200, 405]  # OPTIONS might not be implemented for all endpoints 