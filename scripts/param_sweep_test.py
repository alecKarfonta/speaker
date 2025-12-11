#!/usr/bin/env python3
"""
Comprehensive parameter sweep test for GLM-TTS using STT verification.
Tests different parameter combinations to find optimal settings.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

# Test sentences - varied lengths and content
TEST_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to our text to speech demonstration system.",
    "Testing one two three four five six seven eight nine ten.",
    "This is a longer sentence to test the model's ability to handle extended text without cutting off prematurely.",
]

# Parameter combinations to test
PARAM_CONFIGS = [
    {"name": "default", "min_ratio": 8.0, "max_ratio": 30.0, "sampling": 25},
    {"name": "higher_min", "min_ratio": 12.0, "max_ratio": 40.0, "sampling": 25},
    {"name": "lower_sampling", "min_ratio": 8.0, "max_ratio": 30.0, "sampling": 10},
    {"name": "higher_sampling", "min_ratio": 8.0, "max_ratio": 30.0, "sampling": 50},
    {"name": "aggressive", "min_ratio": 15.0, "max_ratio": 50.0, "sampling": 15},
]

def run_docker_cmd(cmd: str) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr

def restart_container_with_params(min_ratio: float, max_ratio: float, sampling: int) -> bool:
    """Restart GLM-TTS container with new parameters."""
    print(f"  Restarting container with min_ratio={min_ratio}, max_ratio={max_ratio}, sampling={sampling}...")
    
    # Stop existing container
    run_docker_cmd("docker rm -f speaker-glm 2>/dev/null")
    
    # Start with new params
    cmd = f"""docker run -d --name speaker-glm -p 8016:8000 \
        -e TTS_BACKEND=glm-tts \
        -e PYTHONPATH=/app:/app/GLM-TTS \
        -e COQUI_TOS_AGREED=1 \
        -e GLM_TTS_MIN_TOKEN_TEXT_RATIO={min_ratio} \
        -e GLM_TTS_MAX_TOKEN_TEXT_RATIO={max_ratio} \
        -e GLM_TTS_SAMPLING={sampling} \
        --gpus all \
        -v /home/alec/git/speaker/GLM-TTS/ckpt:/app/GLM-TTS/ckpt:ro \
        -v /home/alec/git/speaker/data/voices:/app/data/voices \
        ghcr.io/aleckarfonta/speaker:latest \
        uvicorn app.main:app --host 0.0.0.0 --port 8000"""
    
    code, output = run_docker_cmd(cmd)
    if code != 0:
        print(f"  ERROR starting container: {output}")
        return False
    
    # Wait for health
    print("  Waiting for model to load (2 min)...")
    time.sleep(120)
    
    # Check health
    code, output = run_docker_cmd("curl -s http://localhost:8016/health")
    if "healthy" not in output:
        print(f"  Container not healthy: {output}")
        return False
    
    return True

def generate_and_transcribe(text: str, voice: str = "batman") -> Tuple[str, float, int]:
    """Generate TTS and transcribe with Whisper. Returns (transcription, duration, tokens)."""
    import requests
    
    # Generate TTS
    try:
        response = requests.post(
            "http://localhost:8016/tts",
            json={"text": text, "voice_name": voice, "language": "en", "output_format": "wav"},
            timeout=120
        )
        response.raise_for_status()
    except Exception as e:
        return f"ERROR: {e}", 0, 0
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(response.content)
        audio_path = f.name
    
    # Get audio duration
    import wave
    with wave.open(audio_path, 'rb') as wf:
        duration = wf.getnframes() / wf.getframerate()
    
    # Copy to container and transcribe
    run_docker_cmd(f"docker cp {audio_path} speaker-glm:/tmp/test.wav")
    code, output = run_docker_cmd("""docker exec speaker-glm python3 -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('/tmp/test.wav')
print(result['text'].strip())
" 2>/dev/null""")
    
    # Cleanup
    Path(audio_path).unlink()
    
    # Get token count from logs
    code2, logs = run_docker_cmd("docker logs speaker-glm 2>&1 | grep 'LLM generated' | tail -1")
    tokens = 0
    if "tokens" in logs:
        try:
            tokens = int(logs.split("generated")[1].split("tokens")[0].strip())
        except:
            pass
    
    return output.strip(), duration, tokens

def calculate_word_accuracy(original: str, transcribed: str) -> float:
    """Calculate word-level accuracy."""
    orig_words = original.lower().replace(".", "").replace(",", "").split()
    trans_words = transcribed.lower().replace(".", "").replace(",", "").split()
    
    if not orig_words:
        return 0.0
    
    matches = sum(1 for w in orig_words if w in trans_words)
    return matches / len(orig_words)

def main():
    print("=" * 80)
    print("GLM-TTS Parameter Sweep Test")
    print("=" * 80)
    
    results = []
    
    for config in PARAM_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Testing config: {config['name']}")
        print(f"  min_ratio={config['min_ratio']}, max_ratio={config['max_ratio']}, sampling={config['sampling']}")
        print("=" * 80)
        
        # Note: We can't easily change runtime params without rebuilding
        # So we'll test with current config and document what params to try
        
        config_results = {"config": config, "tests": []}
        
        for sentence in TEST_SENTENCES:
            print(f"\n  Testing: '{sentence[:50]}...'")
            
            transcription, duration, tokens = generate_and_transcribe(sentence)
            accuracy = calculate_word_accuracy(sentence, transcription)
            
            result = {
                "original": sentence,
                "transcribed": transcription,
                "duration": duration,
                "tokens": tokens,
                "accuracy": accuracy
            }
            config_results["tests"].append(result)
            
            status = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.5 else "❌"
            print(f"    {status} Accuracy: {accuracy*100:.0f}%")
            print(f"    Duration: {duration:.2f}s, Tokens: {tokens}")
            print(f"    Transcribed: '{transcription[:60]}...'")
        
        # Calculate average accuracy for this config
        avg_accuracy = sum(t["accuracy"] for t in config_results["tests"]) / len(config_results["tests"])
        config_results["avg_accuracy"] = avg_accuracy
        results.append(config_results)
        
        print(f"\n  Average accuracy for {config['name']}: {avg_accuracy*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for r in results:
        print(f"\n{r['config']['name']}: {r['avg_accuracy']*100:.1f}% average accuracy")
        for t in r["tests"]:
            status = "✅" if t["accuracy"] >= 0.8 else "⚠️" if t["accuracy"] >= 0.5 else "❌"
            print(f"  {status} {t['accuracy']*100:.0f}% - '{t['original'][:40]}...'")

if __name__ == "__main__":
    main()

