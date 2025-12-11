#!/usr/bin/env python3
"""
Extensive GLM-TTS Parameter Testing with proper text normalization.
"""

import requests
import json
import tempfile
import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# APIs
TTS_API = "http://localhost:8016"
STT_API = "http://localhost:8603/v1/audio/transcriptions"

# Number word mappings
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten',
}

# Test sentences - varied content
TEST_SENTENCES = [
    "Hello.",
    "Hello world.",
    "Good morning everyone.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the text to speech demonstration.",
    "Testing one two three four five.",
    "Please call me at your earliest convenience.",
    "The weather today is sunny and warm.",
    "I would like to order a pizza please.",
    "Thank you for your attention.",
]

# Parameter configurations to test
PARAM_CONFIGS = [
    {"name": "baseline", "sampling": 25, "min_ratio": 8, "max_ratio": 30},
    {"name": "low_sampling", "sampling": 10, "min_ratio": 8, "max_ratio": 30},
    {"name": "very_low_sampling", "sampling": 5, "min_ratio": 8, "max_ratio": 30},
    {"name": "high_sampling", "sampling": 50, "min_ratio": 8, "max_ratio": 30},
    {"name": "higher_min_ratio", "sampling": 25, "min_ratio": 12, "max_ratio": 40},
    {"name": "lower_min_ratio", "sampling": 25, "min_ratio": 4, "max_ratio": 30},
    {"name": "high_max_ratio", "sampling": 25, "min_ratio": 8, "max_ratio": 50},
    {"name": "balanced_high", "sampling": 15, "min_ratio": 10, "max_ratio": 35},
    {"name": "conservative", "sampling": 10, "min_ratio": 10, "max_ratio": 40},
]


def normalize_text(text: str) -> str:
    """Normalize text for comparison - handle numbers, punctuation, case."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[.,!?;:\'"()-]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Convert number words to digits and vice versa for comparison
    words = text.split()
    normalized = []
    for word in words:
        # Keep both forms for matching
        normalized.append(word)
        if word in NUMBER_WORDS:
            normalized.append(NUMBER_WORDS[word])
    
    return ' '.join(normalized)


def word_similarity(original: str, transcribed: str) -> float:
    """Calculate word-level similarity with number normalization."""
    orig_norm = normalize_text(original)
    trans_norm = normalize_text(transcribed)
    
    orig_words = set(orig_norm.split())
    trans_words = set(trans_norm.split())
    
    if not orig_words:
        return 0.0
    
    # Count matches (including number equivalents)
    matches = len(orig_words & trans_words)
    
    # Original word count (without duplicates from normalization)
    orig_count = len(set(normalize_text(original).split()[:len(original.split())]))
    
    return min(1.0, matches / max(orig_count, 1))


def transcribe(audio_path: str) -> str:
    """Transcribe audio using STT API."""
    try:
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
    except Exception as e:
        print(f"    STT Error: {e}")
    return ""


def generate_tts(text: str, voice: str = "trump") -> Tuple[bytes, bool]:
    """Generate TTS audio."""
    try:
        response = requests.post(
            f"{TTS_API}/tts",
            json={
                "text": text,
                "voice_name": voice,
                "language": "en",
                "output_format": "wav"
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.content, True
    except Exception as e:
        print(f"    TTS Error: {e}")
    return b"", False


def wait_for_service(timeout: int = 180) -> bool:
    """Wait for TTS service to be healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{TTS_API}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return True
        except:
            pass
        time.sleep(5)
    return False


def restart_with_config(sampling: int, min_ratio: float, max_ratio: float) -> bool:
    """Restart container with new config."""
    subprocess.run(["docker", "rm", "-f", "speaker-glm"], 
                   capture_output=True, timeout=30)
    
    cmd = [
        "docker", "run", "-d", "--name", "speaker-glm",
        "-p", "8016:8000",
        "-e", "TTS_BACKEND=glm-tts",
        "-e", "PYTHONPATH=/app:/app/GLM-TTS",
        "-e", "COQUI_TOS_AGREED=1",
        "-e", f"GLM_TTS_SAMPLING={sampling}",
        "-e", f"GLM_TTS_MIN_TOKEN_TEXT_RATIO={min_ratio}",
        "-e", f"GLM_TTS_MAX_TOKEN_TEXT_RATIO={max_ratio}",
        "--gpus", "all",
        "--add-host=host.docker.internal:host-gateway",
        "-v", "/home/alec/git/speaker/GLM-TTS/ckpt:/app/GLM-TTS/ckpt:ro",
        "-v", "/home/alec/git/speaker/data/voices:/app/data/voices",
        "ghcr.io/aleckarfonta/speaker:latest",
        "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def test_config(config: Dict) -> Dict:
    """Test a single configuration."""
    results = {
        "config": config,
        "tests": [],
        "passed": 0,
        "total": 0,
        "avg_similarity": 0.0
    }
    
    total_sim = 0.0
    
    for sentence in TEST_SENTENCES:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            audio, success = generate_tts(sentence)
            if not success:
                results["tests"].append({
                    "sentence": sentence,
                    "error": True,
                    "similarity": 0.0
                })
                results["total"] += 1
                continue
            
            with open(temp_path, 'wb') as f:
                f.write(audio)
            
            transcribed = transcribe(temp_path)
            similarity = word_similarity(sentence, transcribed)
            
            test_result = {
                "sentence": sentence,
                "transcribed": transcribed,
                "similarity": similarity,
                "passed": similarity >= 0.8
            }
            results["tests"].append(test_result)
            results["total"] += 1
            if test_result["passed"]:
                results["passed"] += 1
            total_sim += similarity
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    results["avg_similarity"] = total_sim / results["total"] if results["total"] > 0 else 0
    return results


def main():
    print("=" * 80)
    print("GLM-TTS EXTENSIVE PARAMETER TESTING")
    print("=" * 80)
    print()
    
    all_results = []
    
    for i, config in enumerate(PARAM_CONFIGS):
        print(f"\n[{i+1}/{len(PARAM_CONFIGS)}] Testing: {config['name']}")
        print(f"    sampling={config['sampling']}, min_ratio={config['min_ratio']}, max_ratio={config['max_ratio']}")
        
        # Restart with new config
        print("    Restarting container...")
        if not restart_with_config(config["sampling"], config["min_ratio"], config["max_ratio"]):
            print("    ❌ Failed to start container")
            continue
        
        print("    Waiting for service (up to 3 min)...")
        if not wait_for_service(180):
            print("    ❌ Service failed to start")
            continue
        
        # Run tests
        print("    Running tests...")
        results = test_config(config)
        all_results.append(results)
        
        # Print results
        print(f"    Results: {results['passed']}/{results['total']} passed ({results['avg_similarity']*100:.1f}% avg)")
        for t in results["tests"]:
            if t.get("error"):
                print(f"      ❌ ERROR | {t['sentence']}")
            elif t["passed"]:
                print(f"      ✅ {t['similarity']*100:.0f}% | {t['sentence'][:40]}")
            else:
                print(f"      ❌ {t['similarity']*100:.0f}% | {t['sentence'][:40]}")
                print(f"              → {t['transcribed'][:50]}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Config':<20} {'Sampling':>8} {'MinRatio':>8} {'MaxRatio':>8} {'Passed':>8} {'Avg%':>8}")
    print("-" * 80)
    
    # Sort by avg similarity
    all_results.sort(key=lambda x: x["avg_similarity"], reverse=True)
    
    for r in all_results:
        c = r["config"]
        print(f"{c['name']:<20} {c['sampling']:>8} {c['min_ratio']:>8} {c['max_ratio']:>8} "
              f"{r['passed']}/{r['total']:>5} {r['avg_similarity']*100:>7.1f}%")
    
    print()
    print("=" * 80)
    print("BEST CONFIG:", all_results[0]["config"]["name"] if all_results else "N/A")
    print("=" * 80)


if __name__ == "__main__":
    main()

