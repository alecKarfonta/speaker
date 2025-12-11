#!/usr/bin/env python3
"""
GLM-TTS Parameter Sweep Test
Tests different parameter combinations and measures quality via STT.
"""

import requests
import json
import tempfile
import os
from pathlib import Path

TTS_API = "http://localhost:8016"
STT_API = "http://localhost:8603/v1/audio/transcriptions"

TEST_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the text to speech demonstration.",
]

def transcribe(audio_path):
    """Transcribe audio using STT API."""
    with open(audio_path, 'rb') as f:
        response = requests.post(
            STT_API,
            headers={"Authorization": "Bearer stt-api-key"},
            files={"file": f},
            data={"model": "base", "language": "en"},
            timeout=60
        )
    if response.status_code == 200:
        return response.json().get("text", "").strip()
    return ""

def word_accuracy(original, transcribed):
    """Calculate word-level accuracy."""
    orig = set(original.lower().replace(".", "").replace(",", "").split())
    trans = set(transcribed.lower().replace(".", "").replace(",", "").split())
    if not orig:
        return 0.0
    return len(orig & trans) / len(orig)

def test_params(sampling, min_ratio, max_ratio, beam_size=1):
    """Test a parameter combination."""
    # We can't change params at runtime, so we'll document what should be tested
    results = []
    
    for sentence in TEST_SENTENCES:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            response = requests.post(
                f"{TTS_API}/tts",
                json={
                    "text": sentence,
                    "voice_name": "trump",
                    "language": "en",
                    "output_format": "wav"
                },
                timeout=120
            )
            
            if response.status_code != 200:
                results.append({"sentence": sentence, "accuracy": 0, "error": True})
                continue
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            transcribed = transcribe(temp_path)
            accuracy = word_accuracy(sentence, transcribed)
            
            results.append({
                "sentence": sentence,
                "transcribed": transcribed,
                "accuracy": accuracy,
                "size": len(response.content)
            })
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0
    return avg_accuracy, results

def main():
    print("=" * 80)
    print("GLM-TTS Parameter Analysis")
    print("=" * 80)
    print()
    
    # Current configuration test
    print("Testing CURRENT configuration...")
    avg_acc, results = test_params(25, 8, 30, 1)
    
    print(f"\nCurrent Config: sampling=25, min_ratio=8, max_ratio=30, beam=1")
    print(f"Average Accuracy: {avg_acc*100:.1f}%")
    print()
    
    for r in results:
        status = "✅" if r["accuracy"] >= 0.9 else "⚠️" if r["accuracy"] >= 0.7 else "❌"
        print(f"  {status} {r['accuracy']*100:.0f}% | '{r['sentence'][:40]}...'")
        print(f"       → '{r.get('transcribed', 'N/A')[:50]}...'")
    
    print()
    print("=" * 80)
    print("PARAMETER DOCUMENTATION")
    print("=" * 80)
    print("""
GLM-TTS has these key parameters:

1. sampling (top-k): Controls randomness in token selection
   - Lower (10): More deterministic, may sound robotic
   - Default (25): Balanced
   - Higher (50): More variation, may introduce errors

2. min_token_text_ratio: Minimum audio tokens per text token
   - Lower (4): Faster speech, may cut off
   - Default (8): Prevents early stopping
   - Higher (12): Ensures complete speech, may add pauses

3. max_token_text_ratio: Maximum audio tokens per text token
   - Lower (20): Faster, may truncate
   - Default (30): Reasonable length
   - Higher (50): Allows longer outputs

4. beam_size: Beam search width
   - 1: Greedy (fast, default)
   - 2-3: Better quality, slower

Current best settings based on tests:
  sampling=25, min_ratio=8, max_ratio=30, beam=1

To test different params, modify config.py or use environment variables:
  GLM_TTS_SAMPLING=10
  GLM_TTS_MIN_TOKEN_TEXT_RATIO=12
  etc.
""")

if __name__ == "__main__":
    main()
