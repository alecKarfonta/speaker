#!/usr/bin/env python3
"""
Speaker TTS Service Test Suite

Comprehensive testing script for the Speaker TTS API including:
- Legacy /tts endpoint testing
- OpenAI-compatible /v1/audio/speech endpoint testing  
- STT-based accuracy evaluation (round-trip testing)
- Performance benchmarking
- Multi-format output testing

Usage:
    python test_tts_service.py                    # Run all tests
    python test_tts_service.py --endpoint openai  # Test only OpenAI endpoint
    python test_tts_service.py --benchmark        # Run performance benchmark
"""

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
from difflib import SequenceMatcher

# Configuration
DEFAULT_TTS_URL = os.environ.get("TTS_API_URL", "http://localhost:8012")
DEFAULT_STT_URL = os.environ.get("STT_API_URL", "http://192.168.1.196:8603")
DEFAULT_STT_ENDPOINT = "/v1/audio/transcriptions"


@dataclass
class TestResult:
    """Result of a single test case"""
    name: str
    passed: bool
    input_text: str
    transcribed_text: Optional[str] = None
    accuracy: float = 0.0
    latency_ms: float = 0.0
    audio_size_bytes: int = 0
    audio_format: str = ""
    error: Optional[str] = None

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        if self.error:
            return f"{status} {self.name}: {self.error}"
        return (
            f"{status} {self.name}: "
            f"accuracy={self.accuracy:.1%}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"size={self.audio_size_bytes}B"
        )


class TTSTestSuite:
    """Comprehensive TTS testing suite with STT evaluation"""

    def __init__(
        self,
        tts_url: str = DEFAULT_TTS_URL,
        stt_url: str = DEFAULT_STT_URL,
        stt_endpoint: str = DEFAULT_STT_ENDPOINT,
        stt_api_key: str = "your-api-key",
        verbose: bool = False
    ):
        self.tts_url = tts_url.rstrip("/")
        self.stt_url = stt_url.rstrip("/")
        self.stt_endpoint = stt_endpoint
        self.stt_api_key = stt_api_key
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, message: str):
        """Print verbose output"""
        if self.verbose:
            print(f"  [DEBUG] {message}")

    # -------------------------------------------------------------------------
    # Core API Methods
    # -------------------------------------------------------------------------

    def get_voices(self) -> List[str]:
        """Fetch available voices from the TTS API"""
        try:
            resp = requests.get(f"{self.tts_url}/voices", timeout=10)
            resp.raise_for_status()
            return resp.json().get("voices", [])
        except Exception as e:
            print(f"⚠️  Failed to get voices: {e}")
            return []

    def generate_speech_legacy(
        self,
        text: str,
        voice_name: str,
        output_format: str = "wav",
        language: str = "en"
    ) -> tuple[bytes, float]:
        """
        Generate speech using legacy /tts endpoint.
        Returns (audio_bytes, latency_ms)
        """
        start = time.time()
        resp = requests.post(
            f"{self.tts_url}/tts",
            json={
                "text": text,
                "voice_name": voice_name,
                "language": language,
                "output_format": output_format
            },
            timeout=120
        )
        latency_ms = (time.time() - start) * 1000
        resp.raise_for_status()
        return resp.content, latency_ms

    def generate_speech_openai(
        self,
        text: str,
        voice: str,
        response_format: str = "mp3",
        model: str = "tts-1",
        speed: float = 1.0
    ) -> tuple[bytes, float]:
        """
        Generate speech using OpenAI-compatible /v1/audio/speech endpoint.
        Returns (audio_bytes, latency_ms)
        """
        start = time.time()
        resp = requests.post(
            f"{self.tts_url}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "model": model,
                "response_format": response_format,
                "speed": speed
            },
            timeout=120
        )
        latency_ms = (time.time() - start) * 1000
        resp.raise_for_status()
        return resp.content, latency_ms

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribe audio using the STT API.
        Returns transcribed text.
        """
        files = {
            "file": (filename, io.BytesIO(audio_bytes), "audio/wav")
        }
        headers = {
            "Authorization": f"Bearer {self.stt_api_key}"
        }
        data = {
            "model": "whisper-1"
        }

        resp = requests.post(
            f"{self.stt_url}{self.stt_endpoint}",
            files=files,
            data=data,
            headers=headers,
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("text", "").strip()

    def calculate_accuracy(self, original: str, transcribed: str) -> float:
        """Calculate text similarity between original and transcribed text"""
        # Normalize texts
        orig = original.lower().strip()
        trans = transcribed.lower().strip()
        
        # Use SequenceMatcher for similarity ratio
        return SequenceMatcher(None, orig, trans).ratio()

    # -------------------------------------------------------------------------
    # Test Cases
    # -------------------------------------------------------------------------

    def test_health(self) -> TestResult:
        """Test the health endpoint"""
        try:
            resp = requests.get(f"{self.tts_url}/health", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return TestResult(
                name="Health Check",
                passed=data.get("status") == "healthy",
                input_text="N/A",
                latency_ms=0
            )
        except Exception as e:
            return TestResult(
                name="Health Check",
                passed=False,
                input_text="N/A",
                error=str(e)
            )

    def test_voices_endpoint(self) -> TestResult:
        """Test the voices listing endpoint"""
        try:
            voices = self.get_voices()
            return TestResult(
                name="Voices Endpoint",
                passed=len(voices) > 0,
                input_text=f"Found {len(voices)} voices",
                latency_ms=0
            )
        except Exception as e:
            return TestResult(
                name="Voices Endpoint",
                passed=False,
                input_text="N/A",
                error=str(e)
            )

    def test_legacy_endpoint(
        self,
        text: str,
        voice_name: str,
        evaluate_with_stt: bool = True
    ) -> TestResult:
        """Test the legacy /tts endpoint with optional STT evaluation"""
        try:
            self.log(f"Generating speech via /tts: '{text[:50]}...'")
            audio, latency = self.generate_speech_legacy(text, voice_name, "wav")
            
            transcribed = ""
            accuracy = 0.0
            
            if evaluate_with_stt:
                self.log("Transcribing audio via STT...")
                transcribed = self.transcribe_audio(audio)
                accuracy = self.calculate_accuracy(text, transcribed)
                self.log(f"Transcribed: '{transcribed}'")
                self.log(f"Accuracy: {accuracy:.1%}")

            return TestResult(
                name=f"Legacy /tts ({voice_name})",
                passed=len(audio) > 1000 and (not evaluate_with_stt or accuracy > 0.5),
                input_text=text,
                transcribed_text=transcribed,
                accuracy=accuracy,
                latency_ms=latency,
                audio_size_bytes=len(audio),
                audio_format="wav"
            )
        except Exception as e:
            return TestResult(
                name=f"Legacy /tts ({voice_name})",
                passed=False,
                input_text=text,
                error=str(e)
            )

    def test_openai_endpoint(
        self,
        text: str,
        voice: str,
        response_format: str = "mp3",
        evaluate_with_stt: bool = True
    ) -> TestResult:
        """Test the OpenAI-compatible /v1/audio/speech endpoint"""
        try:
            self.log(f"Generating speech via /v1/audio/speech: '{text[:50]}...'")
            audio, latency = self.generate_speech_openai(text, voice, response_format)
            
            transcribed = ""
            accuracy = 0.0
            
            # For STT evaluation, we need WAV format
            if evaluate_with_stt:
                # If not WAV, generate WAV version for STT
                if response_format != "wav":
                    self.log("Generating WAV version for STT evaluation...")
                    audio_wav, _ = self.generate_speech_openai(text, voice, "wav")
                else:
                    audio_wav = audio
                
                self.log("Transcribing audio via STT...")
                transcribed = self.transcribe_audio(audio_wav)
                accuracy = self.calculate_accuracy(text, transcribed)
                self.log(f"Transcribed: '{transcribed}'")
                self.log(f"Accuracy: {accuracy:.1%}")

            return TestResult(
                name=f"OpenAI /v1/audio/speech ({voice}, {response_format})",
                passed=len(audio) > 100 and (not evaluate_with_stt or accuracy > 0.5),
                input_text=text,
                transcribed_text=transcribed,
                accuracy=accuracy,
                latency_ms=latency,
                audio_size_bytes=len(audio),
                audio_format=response_format
            )
        except Exception as e:
            return TestResult(
                name=f"OpenAI /v1/audio/speech ({voice}, {response_format})",
                passed=False,
                input_text=text,
                error=str(e)
            )

    def test_all_formats(self, text: str, voice: str) -> List[TestResult]:
        """Test all supported audio formats on the OpenAI endpoint"""
        formats = ["mp3", "wav", "opus", "aac", "flac", "pcm"]
        results = []
        
        for fmt in formats:
            print(f"  Testing format: {fmt}...")
            result = self.test_openai_endpoint(text, voice, fmt, evaluate_with_stt=False)
            results.append(result)
            print(f"    {result}")
        
        return results

    # -------------------------------------------------------------------------
    # Test Suites
    # -------------------------------------------------------------------------

    def run_basic_tests(self, voice: Optional[str] = None) -> List[TestResult]:
        """Run basic functionality tests"""
        print("\n🔍 Running Basic Tests...")
        
        results = []
        
        # Health check
        result = self.test_health()
        print(f"  {result}")
        results.append(result)
        
        # Voices endpoint
        result = self.test_voices_endpoint()
        print(f"  {result}")
        results.append(result)
        
        # Get a voice if not provided
        if not voice:
            voices = self.get_voices()
            if voices:
                voice = voices[0]
            else:
                print("  ⚠️  No voices available, skipping TTS tests")
                return results
        
        print(f"  Using voice: {voice}")
        
        return results

    def run_legacy_tests(
        self,
        voice: Optional[str] = None,
        evaluate_with_stt: bool = True
    ) -> List[TestResult]:
        """Run legacy endpoint tests"""
        print("\n🎙️  Running Legacy Endpoint Tests...")
        
        results = []
        
        # Get a voice if not provided
        if not voice:
            voices = self.get_voices()
            voice = voices[0] if voices else None
        
        if not voice:
            print("  ⚠️  No voices available")
            return results
        
        test_texts = [
            "Hello, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing one two three four five.",
        ]
        
        for text in test_texts:
            result = self.test_legacy_endpoint(text, voice, evaluate_with_stt)
            print(f"  {result}")
            if result.transcribed_text:
                print(f"      Input: '{text}'")
                print(f"      Output: '{result.transcribed_text}'")
            results.append(result)
        
        return results

    def run_openai_tests(
        self,
        voice: Optional[str] = None,
        evaluate_with_stt: bool = True
    ) -> List[TestResult]:
        """Run OpenAI-compatible endpoint tests"""
        print("\n🤖 Running OpenAI-Compatible Endpoint Tests...")
        
        results = []
        
        # Get a voice if not provided
        if not voice:
            voices = self.get_voices()
            voice = voices[0] if voices else None
        
        if not voice:
            print("  ⚠️  No voices available")
            return results
        
        # Test with various inputs
        test_texts = [
            "Hello, this is a test of the OpenAI compatible endpoint.",
            "The quick brown fox jumps over the lazy dog.",
            "One two three four five six seven eight nine ten.",
        ]
        
        for text in test_texts:
            result = self.test_openai_endpoint(text, voice, "wav", evaluate_with_stt)
            print(f"  {result}")
            if result.transcribed_text:
                print(f"      Input: '{text}'")
                print(f"      Output: '{result.transcribed_text}'")
            results.append(result)
        
        # Test OpenAI standard voice names (should map to available voice)
        print("\n  Testing OpenAI voice names...")
        for openai_voice in ["alloy", "nova", "echo"]:
            result = self.test_openai_endpoint(
                "Testing OpenAI voice mapping.",
                openai_voice,
                "mp3",
                evaluate_with_stt=False
            )
            print(f"  {result}")
            results.append(result)
        
        return results

    def run_format_tests(self, voice: Optional[str] = None) -> List[TestResult]:
        """Test all audio output formats"""
        print("\n📀 Running Audio Format Tests...")
        
        if not voice:
            voices = self.get_voices()
            voice = voices[0] if voices else None
        
        if not voice:
            print("  ⚠️  No voices available")
            return []
        
        return self.test_all_formats("Testing audio format output.", voice)

    def run_benchmark(
        self,
        voice: Optional[str] = None,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Run performance benchmark"""
        print(f"\n⚡ Running Performance Benchmark ({iterations} iterations)...")
        
        if not voice:
            voices = self.get_voices()
            voice = voices[0] if voices else None
        
        if not voice:
            print("  ⚠️  No voices available")
            return {}
        
        test_text = "The quick brown fox jumps over the lazy dog."
        
        # Legacy endpoint benchmark
        legacy_latencies = []
        for i in range(iterations):
            try:
                _, latency = self.generate_speech_legacy(test_text, voice, "wav")
                legacy_latencies.append(latency)
                print(f"  Legacy {i+1}/{iterations}: {latency:.0f}ms")
            except Exception as e:
                print(f"  Legacy {i+1}/{iterations}: FAILED - {e}")
        
        # OpenAI endpoint benchmark
        openai_latencies = []
        for i in range(iterations):
            try:
                _, latency = self.generate_speech_openai(test_text, voice, "wav")
                openai_latencies.append(latency)
                print(f"  OpenAI {i+1}/{iterations}: {latency:.0f}ms")
            except Exception as e:
                print(f"  OpenAI {i+1}/{iterations}: FAILED - {e}")
        
        results = {
            "legacy": {
                "avg_ms": sum(legacy_latencies) / len(legacy_latencies) if legacy_latencies else 0,
                "min_ms": min(legacy_latencies) if legacy_latencies else 0,
                "max_ms": max(legacy_latencies) if legacy_latencies else 0,
            },
            "openai": {
                "avg_ms": sum(openai_latencies) / len(openai_latencies) if openai_latencies else 0,
                "min_ms": min(openai_latencies) if openai_latencies else 0,
                "max_ms": max(openai_latencies) if openai_latencies else 0,
            }
        }
        
        print("\n  📊 Benchmark Results:")
        print(f"  Legacy: avg={results['legacy']['avg_ms']:.0f}ms, "
              f"min={results['legacy']['min_ms']:.0f}ms, "
              f"max={results['legacy']['max_ms']:.0f}ms")
        print(f"  OpenAI: avg={results['openai']['avg_ms']:.0f}ms, "
              f"min={results['openai']['min_ms']:.0f}ms, "
              f"max={results['openai']['max_ms']:.0f}ms")
        
        return results

    def run_all_tests(
        self,
        voice: Optional[str] = None,
        evaluate_with_stt: bool = True
    ) -> List[TestResult]:
        """Run all tests"""
        all_results = []
        
        all_results.extend(self.run_basic_tests(voice))
        all_results.extend(self.run_legacy_tests(voice, evaluate_with_stt))
        all_results.extend(self.run_openai_tests(voice, evaluate_with_stt))
        all_results.extend(self.run_format_tests(voice))
        
        self.results = all_results
        return all_results

    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print(f"📋 Test Summary: {passed}/{total} passed")
        print("=" * 60)
        
        if passed < total:
            print("\n❌ Failed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r}")
        
        # Calculate average accuracy for STT-evaluated tests
        stt_results = [r for r in self.results if r.transcribed_text is not None]
        if stt_results:
            avg_accuracy = sum(r.accuracy for r in stt_results) / len(stt_results)
            print(f"\n🎯 Average STT Accuracy: {avg_accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Speaker TTS Service Test Suite")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL, help="TTS API URL")
    parser.add_argument("--stt-url", default=DEFAULT_STT_URL, help="STT API URL")
    parser.add_argument("--voice", help="Voice to use for testing")
    parser.add_argument("--endpoint", choices=["legacy", "openai", "all"], default="all",
                        help="Which endpoint(s) to test")
    parser.add_argument("--no-stt", action="store_true", help="Skip STT evaluation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--formats", action="store_true", help="Test all audio formats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔊 Speaker TTS Service Test Suite")
    print("=" * 60)
    print(f"TTS URL: {args.tts_url}")
    print(f"STT URL: {args.stt_url}")
    
    suite = TTSTestSuite(
        tts_url=args.tts_url,
        stt_url=args.stt_url,
        verbose=args.verbose
    )
    
    evaluate_with_stt = not args.no_stt
    
    if args.benchmark:
        suite.run_benchmark(args.voice)
    elif args.formats:
        suite.run_format_tests(args.voice)
    elif args.endpoint == "legacy":
        suite.run_basic_tests(args.voice)
        suite.run_legacy_tests(args.voice, evaluate_with_stt)
        suite.print_summary()
    elif args.endpoint == "openai":
        suite.run_basic_tests(args.voice)
        suite.run_openai_tests(args.voice, evaluate_with_stt)
        suite.print_summary()
    else:
        suite.run_all_tests(args.voice, evaluate_with_stt)
        suite.print_summary()
    
    # Return exit code based on test results
    if suite.results:
        failed = sum(1 for r in suite.results if not r.passed)
        sys.exit(min(failed, 1))


if __name__ == "__main__":
    main()
