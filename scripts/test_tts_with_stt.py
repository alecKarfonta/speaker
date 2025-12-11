#!/usr/bin/env python3
"""
Test TTS output using Speech-to-Text (Whisper).
Sends text to TTS API, then transcribes the audio to verify correctness.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Installing openai-whisper...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
    import whisper

import requests


def generate_tts(text: str, voice: str = "batman", api_url: str = "http://localhost:8016") -> Path:
    """Generate TTS audio and save to temp file."""
    response = requests.post(
        f"{api_url}/tts",
        json={
            "text": text,
            "voice_name": voice,
            "language": "en",
            "output_format": "wav"
        },
        timeout=120
    )
    response.raise_for_status()
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    return Path(temp_file.name)


def transcribe_audio(audio_path: Path, model_name: str = "base", 
                      stt_api_url: str = "http://localhost:8603/v1/audio/transcriptions",
                      language: str = "en") -> str:
    """Transcribe audio using STT API or local Whisper."""
    # Try STT API first
    try:
        with open(audio_path, 'rb') as f:
            response = requests.post(
                stt_api_url,
                headers={"Authorization": "Bearer stt-api-key"},
                files={"file": f},
                data={"model": model_name, "language": language},
                timeout=60
            )
        if response.status_code == 200:
            return response.json().get("text", "").strip()
        print(f"  STT API error: {response.status_code}")
    except Exception as e:
        print(f"  STT API not available: {e}")
    
    # Fallback to local Whisper
    print("  Falling back to local Whisper...")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def calculate_similarity(original: str, transcribed: str) -> float:
    """Calculate word-level similarity between original and transcribed text."""
    original_words = set(original.lower().split())
    transcribed_words = set(transcribed.lower().split())
    
    if not original_words:
        return 0.0
    
    intersection = original_words & transcribed_words
    return len(intersection) / len(original_words)


def main():
    parser = argparse.ArgumentParser(description="Test TTS output with STT")
    parser.add_argument("--text", "-t", default="Hello, this is a test of the text to speech system.",
                        help="Text to synthesize and verify")
    parser.add_argument("--voice", "-v", default="batman", help="Voice to use")
    parser.add_argument("--api-url", "-u", default="http://localhost:8016", help="TTS API URL")
    parser.add_argument("--whisper-model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--keep-audio", "-k", action="store_true", help="Keep audio file after test")
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"TTS → STT Verification Test")
    print(f"=" * 60)
    print(f"Original text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Whisper model: {args.whisper_model}")
    print()
    
    # Generate TTS
    print("Generating TTS audio...")
    try:
        audio_path = generate_tts(args.text, args.voice, args.api_url)
        print(f"  Audio saved to: {audio_path}")
        print(f"  File size: {audio_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"  ERROR: Failed to generate TTS: {e}")
        return 1
    
    # Transcribe
    print("\nTranscribing with Whisper...")
    try:
        transcribed = transcribe_audio(audio_path, args.whisper_model)
        print(f"  Transcribed: {transcribed}")
    except Exception as e:
        print(f"  ERROR: Failed to transcribe: {e}")
        return 1
    
    # Compare
    similarity = calculate_similarity(args.text, transcribed)
    print()
    print(f"=" * 60)
    print(f"Results:")
    print(f"  Original:    {args.text}")
    print(f"  Transcribed: {transcribed}")
    print(f"  Similarity:  {similarity * 100:.1f}%")
    print(f"=" * 60)
    
    # Cleanup
    if not args.keep_audio:
        audio_path.unlink()
        print(f"\nAudio file deleted. Use --keep-audio to preserve it.")
    else:
        print(f"\nAudio file kept at: {audio_path}")
    
    # Return status based on similarity
    if similarity >= 0.8:
        print("\n✓ PASS: TTS output is readable!")
        return 0
    elif similarity >= 0.5:
        print("\n⚠ PARTIAL: Some words recognized, but quality issues detected.")
        return 1
    else:
        print("\n✗ FAIL: TTS output is not readable or significantly different.")
        return 2


if __name__ == "__main__":
    sys.exit(main())

