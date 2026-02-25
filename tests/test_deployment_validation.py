"""
TTS Deployment Validation Tests

Comprehensive test suite to validate TTS deployment quality, including:
- Health checks
- Audio quality metrics
- STT round-trip verification
- Generation consistency
- Performance metrics

Usage:
    # Run all tests
    TTS_API_URL=http://localhost:8012 \
    STT_API_URL=http://192.168.1.77:8603/v1/audio/transcriptions \
    STT_API_KEY=stt-api-key \
    python -m pytest tests/test_deployment_validation.py -v

    # Skip STT tests
    TTS_API_URL=http://localhost:8012 \
    python -m pytest tests/test_deployment_validation.py -v -k "not stt"
"""

import os
import io
import time
import wave
import struct
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Dict
from difflib import SequenceMatcher

import pytest
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deployment_validation")

# Test configuration from environment variables
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8012")
STT_API_URL = os.environ.get("STT_API_URL", "http://192.168.1.77:8603/v1/audio/transcriptions")
STT_API_KEY = os.environ.get("STT_API_KEY", "stt-api-key")
TEST_VOICE = os.environ.get("TEST_VOICE", "biden")
OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    return SequenceMatcher(None, t1, t2).ratio()


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """
    Convert raw PCM float32 audio to WAV format.
    
    Args:
        pcm_bytes: Raw PCM audio data (float32)
        sample_rate: Audio sample rate
        
    Returns:
        WAV file bytes
    """
    # Convert float32 PCM to numpy array
    audio_data = np.frombuffer(pcm_bytes, dtype=np.float32)
    
    # Normalize and convert to int16 for WAV
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return wav_buffer.getvalue()


def call_tts_api(text: str, voice: str = TEST_VOICE) -> Tuple[bytes, float, Dict]:
    """
    Call TTS API and return audio bytes, latency, and metadata.
    
    Returns:
        Tuple of (audio_bytes, latency_seconds, metadata_dict)
    """
    start = time.perf_counter()
    response = requests.post(
        f"{TTS_API_URL}/tts",
        json={"text": text, "voice_name": voice, "language": "en"},
        timeout=120
    )
    latency = time.perf_counter() - start
    response.raise_for_status()
    
    # Extract metadata from headers
    metadata = {
        "sample_rate": int(response.headers.get("x-sample-rate", 24000)),
        "duration": float(response.headers.get("x-audio-duration", 0)),
        "format": response.headers.get("x-audio-format", "raw"),
        "content_type": response.headers.get("content-type", "audio/pcm"),
    }
    
    return response.content, latency, metadata


def call_stt_api(wav_bytes: bytes) -> str:
    """
    Call STT API to transcribe audio.
    
    Args:
        wav_bytes: WAV audio data
        
    Returns:
        Transcribed text
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        temp_path = f.name
    
    try:
        with open(temp_path, "rb") as audio_file:
            response = requests.post(
                STT_API_URL,
                headers={"Authorization": f"Bearer {STT_API_KEY}"},
                files={"file": ("audio.wav", audio_file, "audio/wav")},
                data={"model": "base", "language": "en"},
                timeout=60
            )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()
    finally:
        os.unlink(temp_path)


def analyze_audio(pcm_bytes: bytes, sample_rate: int = 24000) -> dict:
    """
    Analyze audio quality metrics from raw PCM data.
    
    Args:
        pcm_bytes: Raw PCM audio data (float32)
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary with audio metrics
    """
    # Parse float32 PCM data
    data = np.frombuffer(pcm_bytes, dtype=np.float32)
    
    if len(data) == 0:
        return {
            "duration": 0,
            "sample_rate": sample_rate,
            "rms": 0,
            "peak": 0,
            "silence_ratio": 1.0,
            "samples": 0,
        }
    
    duration = len(data) / sample_rate
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))
    
    # Calculate silence ratio (samples below -40dB)
    silence_threshold = 10 ** (-40 / 20)
    silent_samples = np.sum(np.abs(data) < silence_threshold)
    silence_ratio = silent_samples / len(data)
    
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "samples": len(data),
    }


class TestDeploymentValidation:
    """TTS Deployment Validation Test Suite."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        ensure_output_dir()
    
    def test_health_check(self):
        """Test that the TTS API is healthy and responding."""
        response = requests.get(f"{TTS_API_URL}/health", timeout=10)
        assert response.status_code == 200, f"Health check failed: {response.text}"
        
        data = response.json()
        assert data.get("status") == "healthy", f"Unhealthy status: {data}"
        logger.info(f"Health check passed: {data}")
    
    def test_voices_available(self):
        """Test that voices are loaded and available."""
        response = requests.get(f"{TTS_API_URL}/voices", timeout=10)
        assert response.status_code == 200, f"Voices endpoint failed: {response.text}"
        
        data = response.json()
        voices = data.get("voices", [])
        assert len(voices) > 0, "No voices available"
        logger.info(f"Available voices: {voices}")
        
        assert TEST_VOICE in voices, f"Test voice '{TEST_VOICE}' not found in {voices}"
    
    def test_basic_tts_generation(self):
        """Test basic TTS generation returns valid audio."""
        test_text = "Hello, this is a basic text to speech test."
        
        pcm_bytes, latency, metadata = call_tts_api(test_text)
        
        assert len(pcm_bytes) > 0, "Empty audio response"
        assert metadata["sample_rate"] in [22050, 24000, 44100, 48000], f"Unexpected sample rate: {metadata['sample_rate']}"
        
        # Verify it's valid PCM data (float32)
        data = np.frombuffer(pcm_bytes, dtype=np.float32)
        assert len(data) > 0, "No audio samples"
        
        # Save as WAV for manual inspection
        wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
        output_path = OUTPUT_DIR / "test_basic_generation.wav"
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        
        logger.info(f"Basic TTS test passed - Latency: {latency:.2f}s, Samples: {len(data)}, Rate: {metadata['sample_rate']}")
    
    def test_audio_quality_metrics(self):
        """Test that generated audio meets quality thresholds."""
        test_text = "Testing audio quality with this medium length sentence for validation."
        word_count = len(test_text.split())
        
        pcm_bytes, _, metadata = call_tts_api(test_text)
        metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
        
        # Duration check: should be reasonable for word count
        sec_per_word = metrics["duration"] / word_count
        assert 0.2 < sec_per_word < 2.0, f"Suspicious duration: {sec_per_word:.2f}s/word"
        
        # RMS check: audio should not be silent
        assert metrics["rms"] > 0.001, f"Audio too quiet (RMS: {metrics['rms']:.4f})"
        
        # Silence ratio check
        assert metrics["silence_ratio"] < 0.8, f"Too much silence: {metrics['silence_ratio']:.1%}"
        
        logger.info(f"Audio quality metrics: {metrics}")
    
    @pytest.mark.skipif(
        not os.environ.get("STT_API_URL"),
        reason="STT_API_URL not configured"
    )
    def test_stt_roundtrip(self):
        """
        Test TTS→STT round-trip: generate audio, transcribe, compare to original.
        Uses diverse test cases to verify TTS output quality.
        """
        test_cases = [
            # Simple sentences
            "Hello world, this is a simple test.",
            "The quick brown fox jumps over the lazy dog.",
            # Questions
            "How are you doing today?",
            "What time is the meeting scheduled for?",
            # Exclamations
            "That's absolutely amazing!",
            "I can't believe it worked!",
            # Technical/complex words
            "The algorithm processes data asynchronously.",
            "Artificial intelligence is transforming technology.",
            # Names and proper nouns
            "Welcome to San Francisco, California.",
            "Doctor Smith recommended the treatment.",
            # Longer passages
            "This is a longer sentence that tests the system's ability to handle extended passages of text with multiple clauses.",
            "Please speak clearly and at a moderate pace so that the transcription system can accurately capture your words.",
        ]
        
        results = []
        for original_text in test_cases:
            logger.info(f"Testing: '{original_text}'")
            
            pcm_bytes, tts_latency, metadata = call_tts_api(original_text)
            
            # Convert PCM to WAV for STT
            wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
            
            try:
                transcribed = call_stt_api(wav_bytes)
            except Exception as e:
                pytest.fail(f"STT API failed: {e}")
            
            similarity = text_similarity(original_text, transcribed)
            
            results.append({
                "original": original_text,
                "transcribed": transcribed,
                "similarity": similarity,
                "tts_latency": tts_latency,
            })
            
            logger.info(f"  Transcribed: '{transcribed}' (similarity: {similarity:.1%})")
        
        # Save results
        output_path = OUTPUT_DIR / "stt_roundtrip_results.txt"
        with open(output_path, "w") as f:
            for r in results:
                f.write(f"Original:    {r['original']}\n")
                f.write(f"Transcribed: {r['transcribed']}\n")
                f.write(f"Similarity:  {r['similarity']:.1%}\n\n")
        
        avg_similarity = sum(r["similarity"] for r in results) / len(results)
        logger.info(f"Average similarity: {avg_similarity:.1%}")
        
        assert avg_similarity >= 0.70, (
            f"Poor STT round-trip quality: {avg_similarity:.1%}. See {output_path}"
        )
        
        for r in results:
            assert r["similarity"] >= 0.40, (
                f"Very poor transcription for '{r['original']}': "
                f"got '{r['transcribed']}' ({r['similarity']:.1%})"
            )
    
    def test_generation_consistency(self):
        """Test that multiple generations produce consistent quality."""
        test_text = "Consistency test sentence number one."
        num_runs = 3
        
        metrics_list = []
        for i in range(num_runs):
            pcm_bytes, latency, metadata = call_tts_api(test_text)
            metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
            metrics["latency"] = latency
            metrics_list.append(metrics)
            
            # Save as WAV
            wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
            output_path = OUTPUT_DIR / f"consistency_test_{i+1}.wav"
            with open(output_path, "wb") as f:
                f.write(wav_bytes)
            
            logger.info(f"Run {i+1}: duration={metrics['duration']:.2f}s, rms={metrics['rms']:.4f}")
        
        durations = [m["duration"] for m in metrics_list]
        rms_values = [m["rms"] for m in metrics_list]
        
        mean_duration = sum(durations) / len(durations)
        for d in durations:
            variance = abs(d - mean_duration) / mean_duration if mean_duration > 0 else 0
            assert variance < 0.5, f"Inconsistent durations: {durations}"
        
        for i, rms in enumerate(rms_values):
            assert rms > 0.001, f"Run {i+1} produced silent output (RMS: {rms:.4f})"
        
        logger.info(f"Consistency test passed across {num_runs} runs")
    
    def test_performance_metrics(self):
        """Test and report performance metrics including real-time factor (RTF)."""
        test_texts = [
            ("Short", "Short."),
            ("Medium", "This is a medium length sentence for testing."),
            ("Long", "This is a longer paragraph that contains multiple sentences. It should take more time to process but still complete within reasonable bounds."),
        ]
        
        results = []
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE METRICS")
        logger.info("="*70)
        
        for label, text in test_texts:
            pcm_bytes, latency, metadata = call_tts_api(text)
            metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
            
            # Speed: audio_duration / compute_time (higher = faster)
            # e.g., 2.5x means generating 2.5 seconds of audio per 1 second of compute
            speed = metrics["duration"] / latency if latency > 0 else 0
            
            results.append({
                "label": label,
                "text_length": len(text),
                "word_count": len(text.split()),
                "latency": latency,
                "duration": metrics["duration"],
                "speed": speed,
            })
            
            logger.info(
                f"  [{label:6}] {len(text):3} chars | "
                f"Latency: {latency:5.2f}s | "
                f"Audio: {metrics['duration']:5.2f}s | "
                f"Speed: {speed:5.2f}x real-time"
            )
        
        # Summary
        avg_speed = sum(r["speed"] for r in results) / len(results)
        total_audio = sum(r["duration"] for r in results)
        total_compute = sum(r["latency"] for r in results)
        overall_speed = total_audio / total_compute if total_compute > 0 else 0
        
        logger.info("-"*70)
        logger.info(f"  SUMMARY: {total_audio:.2f}s audio generated in {total_compute:.2f}s")
        logger.info(f"  Average Speed: {avg_speed:.2f}x real-time")
        logger.info(f"  Overall Speed: {overall_speed:.2f}x real-time")
        logger.info("="*70 + "\n")
        
        # Save performance report
        output_path = OUTPUT_DIR / "performance_report.txt"
        with open(output_path, "w") as f:
            f.write("TTS Performance Report\n")
            f.write("="*50 + "\n\n")
            for r in results:
                f.write(f"{r['label']:10} | {r['text_length']:3} chars | ")
                f.write(f"Latency: {r['latency']:.2f}s | ")
                f.write(f"Audio: {r['duration']:.2f}s | ")
                f.write(f"Speed: {r['speed']:.2f}x\n")
            f.write("\n" + "-"*50 + "\n")
            f.write(f"Total Audio:    {total_audio:.2f}s\n")
            f.write(f"Total Compute:  {total_compute:.2f}s\n")
            f.write(f"Overall Speed:  {overall_speed:.2f}x real-time\n")
        
        # Assertions
        for r in results:
            assert r["latency"] < 30, f"Latency too high: {r['latency']:.1f}s"
            # Speed should be at least 0.2x (5s compute for 1s audio max)
            assert r["speed"] > 0.2, f"Speed too slow: {r['speed']:.2f}x"


class TestGarbageDetection:
    """Tests for detecting garbage/degenerate output."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        ensure_output_dir()
    
    def test_no_extreme_duration(self):
        """Test that output duration is reasonable for text length."""
        test_cases = [
            ("Hi.", 0.2, 3.0),
            ("Hello world.", 0.5, 5.0),
            ("This is a normal sentence that should be spoken clearly.", 2.0, 15.0),
        ]
        
        for text, min_dur, max_dur in test_cases:
            pcm_bytes, _, metadata = call_tts_api(text)
            metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
            
            assert metrics["duration"] >= min_dur, f"Audio too short for '{text}': {metrics['duration']:.2f}s"
            assert metrics["duration"] <= max_dur, f"Audio too long for '{text}': {metrics['duration']:.2f}s"
            
            # Save as WAV
            wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
            output_path = OUTPUT_DIR / f"duration_test_{len(text)}.wav"
            with open(output_path, "wb") as f:
                f.write(wav_bytes)
            
            logger.info(f"'{text[:30]}' -> {metrics['duration']:.2f}s (expected {min_dur}-{max_dur}s)")
    
    def test_no_repetitive_audio(self):
        """Test that audio doesn't contain obvious repetitive patterns."""
        test_text = "Testing for repetitive patterns in the audio output."
        pcm_bytes, _, metadata = call_tts_api(test_text)
        
        data = np.frombuffer(pcm_bytes, dtype=np.float32)
        
        chunk_size = 4800  # ~0.2 seconds at 24kHz
        if len(data) > chunk_size * 4:
            chunks = [data[i:i+chunk_size] for i in range(0, len(data)-chunk_size, chunk_size)]
            
            repetitions = 0
            for i in range(len(chunks) - 1):
                if len(chunks[i]) == len(chunks[i+1]):
                    corr = np.corrcoef(chunks[i], chunks[i+1])[0, 1]
                    if not np.isnan(corr) and corr > 0.95:
                        repetitions += 1
            
            repetition_ratio = repetitions / (len(chunks) - 1) if len(chunks) > 1 else 0
            assert repetition_ratio < 0.5, f"Audio appears repetitive: {repetition_ratio:.1%}"
            
            logger.info(f"Repetition check: {repetitions} similar chunks out of {len(chunks)-1}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
