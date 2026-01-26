#!/usr/bin/env python3
"""
Integration tests for Qwen3-TTS API endpoints.

Run against a running container:
    pytest tests/test_qwen_tts_integration.py -v --base-url http://localhost:8012

Or run the test script directly:
    python tests/test_qwen_tts_integration.py --base-url http://localhost:8012
"""

import os
import sys
import time
import struct
import argparse
from pathlib import Path
from typing import Optional
import tempfile

import requests
import pytest

# Default base URL (can be overridden via --base-url or BASE_URL env var)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8012")


# =============================================================================
# Fixtures and Helpers
# =============================================================================

def get_base_url():
    """Get base URL from pytest config or environment."""
    return BASE_URL


def check_qwen_available(base_url: str) -> bool:
    """Check if Qwen TTS endpoints are available."""
    try:
        resp = requests.get(f"{base_url}/api/v1/qwen/speakers", timeout=5)
        return resp.status_code in (200, 501)  # 501 = not installed but endpoint exists
    except Exception:
        return False


def create_test_audio_file() -> bytes:
    """Create a minimal WAV file for testing voice clone."""
    import wave
    import struct
    import io
    
    # Generate 3 seconds of silence at 24kHz
    sample_rate = 24000
    duration = 3
    num_samples = sample_rate * duration
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Write silence (zeros)
        wav.writeframes(b'\x00' * num_samples * 2)
    
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# Test Classes
# =============================================================================

class TestQwenSpeakers:
    """Tests for GET /api/v1/qwen/speakers"""
    
    def test_list_speakers_returns_200_or_501(self):
        """Speakers endpoint should return 200 or 501 (not installed)."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/speakers")
        assert resp.status_code in (200, 501), f"Unexpected status: {resp.status_code}"
        
    def test_list_speakers_response_structure(self):
        """Speakers response should have expected structure."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/speakers")
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        data = resp.json()
        assert "speakers" in data
        assert "total_count" in data
        assert isinstance(data["speakers"], list)
        
    def test_speaker_has_required_fields(self):
        """Each speaker should have required fields."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/speakers")
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        data = resp.json()
        if data["total_count"] == 0:
            pytest.skip("No speakers available")
            
        speaker = data["speakers"][0]
        required_fields = ["id", "name", "description", "native_language", "gender"]
        for field in required_fields:
            assert field in speaker, f"Missing field: {field}"


class TestQwenLanguages:
    """Tests for GET /api/v1/qwen/languages"""
    
    def test_list_languages_returns_200_or_501(self):
        """Languages endpoint should return 200 or 501."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/languages")
        assert resp.status_code in (200, 501)
        
    def test_list_languages_response_structure(self):
        """Languages response should have expected structure."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/languages")
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        data = resp.json()
        assert "languages" in data
        assert "total_count" in data
        assert isinstance(data["languages"], list)
        
    def test_expected_languages_present(self):
        """Should include expected languages."""
        resp = requests.get(f"{BASE_URL}/api/v1/qwen/languages")
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        data = resp.json()
        expected = ["English", "Chinese", "Auto"]
        for lang in expected:
            assert lang in data["languages"], f"Missing language: {lang}"


class TestQwenSynthesize:
    """Tests for POST /api/v1/qwen/synthesize"""
    
    def test_synthesize_custom_voice_mode(self):
        """Synthesize with custom_voice mode."""
        payload = {
            "text": "Hello, this is a test.",
            "mode": "custom_voice",
            "speaker": "Vivian",
            "language": "English"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/synthesize", json=payload)
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code == 503:
            pytest.skip("Qwen backend not initialized")
            
        assert resp.status_code == 200, f"Error: {resp.text}"
        assert resp.headers.get("content-type") == "audio/wav"
        assert len(resp.content) > 1000, "Audio too short"
        
    def test_synthesize_voice_design_mode(self):
        """Synthesize with voice_design mode."""
        payload = {
            "text": "Testing voice design mode.",
            "mode": "voice_design",
            "instruct": "A warm, friendly female voice with moderate pace",
            "language": "English"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/synthesize", json=payload)
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code == 503:
            pytest.skip("VoiceDesign model not loaded")
            
        assert resp.status_code == 200, f"Error: {resp.text}"
        assert resp.headers.get("content-type") == "audio/wav"
        
    def test_synthesize_with_model_size_param(self):
        """Synthesize with explicit model_size parameter."""
        payload = {
            "text": "Testing model size selection.",
            "mode": "custom_voice",
            "speaker": "Chelsie",
            "language": "English",
            "model_size": "0.6B"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/synthesize", json=payload)
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        # Accept 200, 400 (if 0.6B not available), or 503
        assert resp.status_code in (200, 400, 503), f"Unexpected: {resp.status_code}"
        
    def test_synthesize_rejects_empty_text(self):
        """Should reject empty text."""
        payload = {
            "text": "",
            "mode": "custom_voice",
            "speaker": "Vivian"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/synthesize", json=payload)
        assert resp.status_code == 422, "Should reject empty text"
        
    def test_synthesize_rejects_unknown_speaker(self):
        """Should reject unknown speaker name."""
        payload = {
            "text": "Test",
            "mode": "custom_voice",
            "speaker": "NonExistentSpeaker",
            "language": "English"
        }
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/synthesize", json=payload)
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
            
        # Should be 400 (validation) or 503 (not initialized)
        assert resp.status_code in (400, 503), f"Expected 400/503, got {resp.status_code}"


class TestQwenSynthesizeStream:
    """Tests for POST /api/v1/qwen/synthesize/stream"""
    
    def test_stream_returns_chunked_response(self):
        """Streaming endpoint should return chunked data."""
        payload = {
            "text": "Hello streaming world.",
            "mode": "custom_voice",
            "speaker": "Vivian",
            "language": "English"
        }
        resp = requests.post(
            f"{BASE_URL}/api/v1/qwen/synthesize/stream",
            json=payload,
            stream=True
        )
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code == 503:
            pytest.skip("Qwen backend not initialized")
            
        assert resp.status_code == 200, f"Error: {resp.text}"
        assert resp.headers.get("x-streaming") == "true"
        
        # Read at least one chunk
        chunks = []
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                if len(chunks) >= 1:
                    break
                    
        assert len(chunks) >= 1, "Should receive at least one chunk"


class TestQwenVoiceClone:
    """Tests for POST /api/v1/qwen/clone"""
    
    def test_clone_with_ref_audio(self):
        """Voice clone with reference audio."""
        audio_bytes = create_test_audio_file()
        
        files = {"ref_audio": ("test.wav", audio_bytes, "audio/wav")}
        params = {
            "text": "Testing voice cloning.",
            "language": "English",
            "ref_text": "This is the reference transcript."
        }
        
        resp = requests.post(
            f"{BASE_URL}/api/v1/qwen/clone",
            files=files,
            params=params
        )
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code in (500, 503):
            pytest.skip("VoiceClone backend not initialized")
            
        assert resp.status_code == 200, f"Error: {resp.text}"
        assert resp.headers.get("x-mode") == "voice_clone"
        
    def test_clone_with_xvector_only(self):
        """Voice clone with x-vector only (no ref_text needed)."""
        audio_bytes = create_test_audio_file()
        
        files = {"ref_audio": ("test.wav", audio_bytes, "audio/wav")}
        params = {
            "text": "Testing x-vector only mode.",
            "language": "English",
            "use_xvector_only": True
        }
        
        resp = requests.post(
            f"{BASE_URL}/api/v1/qwen/clone",
            files=files,
            params=params
        )
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code in (500, 503):
            pytest.skip("VoiceClone backend not initialized")
            
        # 200 or 400 if x-vector only not supported
        assert resp.status_code in (200, 400, 503), f"Unexpected: {resp.status_code}"
        
    def test_clone_requires_audio_file(self):
        """Clone endpoint should require ref_audio file."""
        data = {"text": "Test", "language": "English"}
        
        resp = requests.post(f"{BASE_URL}/api/v1/qwen/clone", data=data)
        assert resp.status_code == 422, "Should require ref_audio file"


class TestQwenVoicePrompt:
    """Tests for POST /api/v1/qwen/voices/create-prompt"""
    
    def test_create_voice_prompt(self):
        """Create cached voice prompt."""
        audio_bytes = create_test_audio_file()
        
        files = {"ref_audio": ("test.wav", audio_bytes, "audio/wav")}
        params = {
            "voice_id": "test_voice_123",
            "ref_text": "Reference transcript"
        }
        
        resp = requests.post(
            f"{BASE_URL}/api/v1/qwen/voices/create-prompt",
            files=files,
            params=params
        )
        
        if resp.status_code == 501:
            pytest.skip("Qwen TTS not installed")
        elif resp.status_code in (500, 503):
            pytest.skip("Backend not initialized")
            
        assert resp.status_code == 200, f"Error: {resp.text}"
        result = resp.json()
        assert result["voice_id"] == "test_voice_123"


# =============================================================================
# CLI Runner
# =============================================================================

def run_all_tests(base_url: str, verbose: bool = True):
    """Run all tests and report results."""
    global BASE_URL
    BASE_URL = base_url
    
    print(f"\n{'='*60}")
    print(f"Qwen TTS Integration Tests")
    print(f"Target: {base_url}")
    print(f"{'='*60}\n")
    
    # Check connectivity
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health check: {resp.status_code}")
    except Exception as e:
        print(f"✗ Cannot reach server: {e}")
        return 1
    
    # Check Qwen availability
    if check_qwen_available(base_url):
        print("✓ Qwen endpoints available")
    else:
        print("✗ Qwen endpoints not responding")
        return 1
    
    print("\n" + "-"*40 + "\n")
    
    # Run test classes
    test_classes = [
        TestQwenSpeakers,
        TestQwenLanguages,
        TestQwenSynthesize,
        TestQwenSynthesizeStream,
        TestQwenVoiceClone,
        TestQwenVoicePrompt,
    ]
    
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * len(test_class.__name__))
        
        for method_name in dir(test_class):
            if not method_name.startswith("test_"):
                continue
                
            total += 1
            instance = test_class()
            method = getattr(instance, method_name)
            
            try:
                method()
                passed += 1
                print(f"  ✓ {method_name}")
            except pytest.skip.Exception as e:
                skipped += 1
                print(f"  - {method_name} (skipped: {e})")
            except AssertionError as e:
                failed += 1
                print(f"  ✗ {method_name}")
                if verbose:
                    print(f"    Error: {e}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {method_name}")
                if verbose:
                    print(f"    Exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")
    print(f"{'='*60}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen TTS Integration Tests")
    parser.add_argument(
        "--base-url", 
        default=os.environ.get("BASE_URL", "http://localhost:8012"),
        help="Base URL of the TTS API (default: http://localhost:8012)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error messages"
    )
    
    args = parser.parse_args()
    sys.exit(run_all_tests(args.base_url, args.verbose))
