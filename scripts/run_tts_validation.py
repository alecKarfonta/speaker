#!/usr/bin/env python3
"""
TTS Deployment Validation - Test Runner with Performance Report

Runs all TTS validation tests and generates a comprehensive performance report.

Usage:
    ./scripts/run_tts_validation.py
    
    # With custom endpoints
    TTS_API_URL=http://localhost:8012 \
    STT_API_URL=http://192.168.1.77:8603/v1/audio/transcriptions \
    STT_API_KEY=stt-api-key \
    ./scripts/run_tts_validation.py

Output:
    - Console: Real-time test progress and summary
    - tests/output/validation_report.txt: Full performance report
    - tests/output/*.wav: Generated audio files for inspection
"""

import os
import sys
import time
import wave
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

import requests
import numpy as np

# Configuration
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8012")
STT_API_URL = os.environ.get("STT_API_URL", "http://192.168.1.77:8603/v1/audio/transcriptions")
STT_API_KEY = os.environ.get("STT_API_KEY", "stt-api-key")
TEST_VOICE = os.environ.get("TEST_VOICE", "biden")
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "output"


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM float32 audio to WAV format."""
    audio_data = np.frombuffer(pcm_bytes, dtype=np.float32)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return wav_buffer.getvalue()


def call_tts_api(text: str, voice: str = TEST_VOICE) -> Tuple[bytes, float, Dict]:
    """Call TTS API and return audio bytes, latency, and metadata including profiling."""
    start = time.perf_counter()
    response = requests.post(
        f"{TTS_API_URL}/tts",
        json={"text": text, "voice_name": voice, "language": "en"},
        timeout=120
    )
    latency = time.perf_counter() - start
    response.raise_for_status()
    
    # Extract all metadata including profiling headers
    metadata = {
        "sample_rate": int(response.headers.get("x-sample-rate", 24000)),
        "duration": float(response.headers.get("x-audio-duration", 0)),
        # Profiling data
        "timing_total_ms": float(response.headers.get("x-timing-total-ms", 0)),
        "timing_llm_ms": float(response.headers.get("x-timing-llm-ms", 0)),
        "timing_flow_ms": float(response.headers.get("x-timing-flow-ms", 0)),
        "timing_normalize_ms": float(response.headers.get("x-timing-normalize-ms", 0)),
        "timing_prompt_ms": float(response.headers.get("x-timing-prompt-ms", 0)),
        "timing_validation_ms": float(response.headers.get("x-timing-validation-ms", 0)),
        "tokens_generated": int(response.headers.get("x-tokens-generated", 0)),
        "rtf": float(response.headers.get("x-rtf", 0)),
    }
    return response.content, latency, metadata


def call_stt_api(wav_bytes: bytes) -> str:
    """Call STT API to transcribe audio."""
    import tempfile
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
        return response.json().get("text", "").strip()
    finally:
        os.unlink(temp_path)


def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()


def analyze_audio(pcm_bytes: bytes, sample_rate: int) -> Dict:
    """Analyze audio quality metrics."""
    data = np.frombuffer(pcm_bytes, dtype=np.float32)
    if len(data) == 0:
        return {"duration": 0, "rms": 0, "peak": 0, "has_nan": False, "is_valid": False}
    
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    
    # Clean data for analysis (replace NaN/Inf with 0)
    clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        "duration": len(data) / sample_rate,
        "rms": float(np.sqrt(np.mean(clean_data ** 2))),
        "peak": float(np.max(np.abs(clean_data))),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_valid": not has_nan and not has_inf,
    }


def classify_failure(
    original: str, 
    transcribed: str, 
    similarity: float, 
    audio_metrics: Dict,
    text_word_count: int
) -> str:
    """
    Classify the type of failure for diagnostic purposes.
    
    Returns a failure category:
    - GARBAGE: Audio contains NaN/Inf values (corrupt)
    - SILENT: Audio is too quiet (RMS < 0.001)
    - NOISE: Audio is too loud/noisy (RMS > 0.5)
    - TRUNCATED: Audio too short for text (< 0.2s per word)
    - EXTENDED: Audio way too long (> 2s per word, possible loop)
    - MINOR: Transcription differs slightly (>40% match)
    - MAJOR: Transcription differs significantly (<40% match)
    - EMPTY: No transcription returned
    """
    # Check audio quality first
    if audio_metrics.get("has_nan") or audio_metrics.get("has_inf"):
        return "🔥 GARBAGE (NaN/corrupt data)"
    
    if audio_metrics.get("rms", 0) < 0.001:
        return "🔇 SILENT (no audio)"
    
    if audio_metrics.get("rms", 0) > 0.5:
        return "📢 NOISE (extremely loud)"
    
    duration = audio_metrics.get("duration", 0)
    sec_per_word = duration / text_word_count if text_word_count > 0 else 0
    
    if sec_per_word < 0.15 and text_word_count > 1:
        return f"✂️ TRUNCATED ({sec_per_word:.2f}s/word)"
    
    if sec_per_word > 3.0 and text_word_count > 1:
        return f"🔁 EXTENDED ({sec_per_word:.2f}s/word - possible loop)"
    
    # Check transcription quality
    if not transcribed or len(transcribed.strip()) == 0:
        return "❓ EMPTY (no transcription)"
    
    if similarity >= 0.6:
        return "✅ OK"
    elif similarity >= 0.4:
        return "⚠️ MINOR mismatch"
    elif similarity >= 0.2:
        return "❌ MAJOR mismatch"
    else:
        return "💀 SEVERE mismatch (likely garbage)"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title: str):
    """Print a section divider."""
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}")


def run_validation():
    """Run all validation tests and generate report."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_lines = []
    all_passed = True
    
    def log(msg: str):
        print(msg)
        report_lines.append(msg)
    
    # Header
    print_header("TTS DEPLOYMENT VALIDATION")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"TTS API: {TTS_API_URL}")
    log(f"STT API: {STT_API_URL}")
    log(f"Voice: {TEST_VOICE}")
    
    # 1. Health Check
    print_section("1. HEALTH CHECK")
    try:
        response = requests.get(f"{TTS_API_URL}/health", timeout=10)
        health = response.json()
        log(f"  Status: {health.get('status', 'unknown')}")
        log(f"  Model: {health.get('model', 'unknown')}")
        log(f"  Voices: {health.get('available_voices', 0)}")
        log("  ✅ PASSED")
    except Exception as e:
        log(f"  ❌ FAILED: {e}")
        all_passed = False
    
    # 2. Performance Tests with Profiling Breakdown
    print_section("2. PERFORMANCE METRICS")
    perf_tests = [
        # === LENGTH VARIATIONS ===
        ("Tiny", "biden", "Hi."),
        ("Short", "biden", "Hello, how are you today?"),
        ("Medium", "biden", "This is a medium length sentence for testing the text to speech system."),
        ("Long", "biden", "This is a longer paragraph that contains multiple sentences. It should take more time to process but still complete within reasonable bounds. The quality should remain consistent."),
        ("Very Long", "biden", "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. A wizard's job is to vex chumps quickly in fog."),
        ("Paragraph", "biden", "In the heart of the bustling city, where skyscrapers reached toward the heavens and streets hummed with the rhythm of countless footsteps, there existed a small coffee shop that time seemed to have forgotten. Its weathered wooden sign creaked gently in the breeze, while the aroma of freshly ground beans drifted through the air, beckoning passersby to step inside and leave the chaos of the world behind them."),
        
        # === VOICE VARIATIONS ===
        ("Loli Short", "loli", "Hello! This is so fun and exciting!"),
        ("Loli Medium", "loli", "I really love playing games and having adventures with my friends. It's the best thing ever!"),
        ("Batman Short", "batman", "I am vengeance. I am the night."),
        ("Batman Medium", "batman", "The criminal underworld will learn to fear me. Gotham needs a guardian, and I will be that shadow in the darkness."),
        
        # === CONTENT TYPES ===
        ("Technical", "biden", "The API returns JSON with status codes two hundred for success and four hundred for client errors."),
        ("Emotional", "biden", "Oh my goodness, I can't believe this is actually happening! This is the most incredible moment of my entire life!"),
        ("Questions", "biden", "What time is it? Where are you going? How did this happen? Why would anyone do that?"),
        ("Numbers", "biden", "The meeting is scheduled for three thirty PM on January fifteenth, twenty twenty six."),
        
        # === STRESS TESTS ===
        ("Rapid Fire", "biden", "Go! Run! Stop! Wait! Now! Yes! No! Here! There! Quick!"),
    ]
    
    perf_results = []
    total_audio = 0
    total_compute = 0
    total_llm = 0
    total_flow = 0
    
    log(f"\n  {'Label':<14} {'Voice':<8} {'Chars':>5} {'Audio':>7} {'Speed':>7} | {'LLM':>7} {'Flow':>7} {'Other':>7}")
    log(f"  {'-'*14} {'-'*8} {'-'*5} {'-'*7} {'-'*7} | {'-'*7} {'-'*7} {'-'*7}")
    
    for label, voice, text in perf_tests:
        try:
            pcm_bytes, latency, metadata = call_tts_api(text, voice)
            metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
            speed = metrics["duration"] / latency if latency > 0 else 0
            
            # Extract profiling data
            llm_ms = metadata.get("timing_llm_ms", 0)
            flow_ms = metadata.get("timing_flow_ms", 0)
            total_ms = metadata.get("timing_total_ms", latency * 1000)
            other_ms = total_ms - llm_ms - flow_ms
            
            perf_results.append({
                "label": label,
                "voice": voice,
                "chars": len(text),
                "latency": latency,
                "duration": metrics["duration"],
                "speed": speed,
                "llm_ms": llm_ms,
                "flow_ms": flow_ms,
                "other_ms": other_ms,
                "tokens": metadata.get("tokens_generated", 0),
            })
            
            total_audio += metrics["duration"]
            total_compute += latency
            total_llm += llm_ms
            total_flow += flow_ms
            
            log(f"  {label:<14} {voice:<8} {len(text):>5} {metrics['duration']:>6.2f}s {speed:>6.2f}x | {llm_ms:>6.0f}ms {flow_ms:>6.0f}ms {other_ms:>6.0f}ms")
            
            # Save audio
            wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
            with open(OUTPUT_DIR / f"perf_{label.lower().replace(' ', '_')}.wav", "wb") as f:
                f.write(wav_bytes)
                
        except Exception as e:
            log(f"  {label:<14} {voice:<8} ❌ FAILED: {e}")
            all_passed = False
    
    overall_speed = total_audio / total_compute if total_compute > 0 else 0
    llm_pct = (total_llm / (total_llm + total_flow)) * 100 if (total_llm + total_flow) > 0 else 0
    flow_pct = (total_flow / (total_llm + total_flow)) * 100 if (total_llm + total_flow) > 0 else 0
    
    log(f"\n  Summary:")
    log(f"    • Overall Speed: {overall_speed:.2f}x real-time")
    log(f"    • Total Audio: {total_audio:.2f}s in {total_compute:.2f}s compute")
    log(f"    • Stage Breakdown: LLM {total_llm:.0f}ms ({llm_pct:.0f}%) | Flow {total_flow:.0f}ms ({flow_pct:.0f}%)")
    log(f"  ✅ Performance test passed")
    
    # 3. STT Round-Trip Tests - Extended test cases
    print_section("3. STT ROUND-TRIP VALIDATION")
    
    # Comprehensive test set with varied text inputs
    stt_tests = [
        # === Basic sentences (biden) ===
        ("biden", "Hello world, this is a simple test."),
        ("biden", "The quick brown fox jumps over the lazy dog."),
        ("biden", "Good morning, how are you today?"),
        ("biden", "Thank you very much for your help."),
        
        # === Numbers and dates ===
        ("biden", "The meeting is at three thirty in the afternoon."),
        ("biden", "She was born on July fourth, nineteen eighty five."),
        ("biden", "The total comes to forty seven dollars and fifty cents."),
        ("biden", "Call me at five five five, one two three four."),
        
        # === Questions ===
        ("biden", "What time does the movie start tonight?"),
        ("biden", "Have you ever been to Paris before?"),
        ("biden", "Would you like some coffee or tea?"),
        ("biden", "Why did the chicken cross the road?"),
        
        # === Exclamations and emotion ===
        ("biden", "That's absolutely incredible! I love it!"),
        ("biden", "Oh my goodness, I can't believe this is happening!"),
        ("biden", "Congratulations on your promotion!"),
        ("biden", "What a beautiful sunset we're having tonight!"),
        
        # === Contractions ===
        ("biden", "I'm going to the store, I'll be back soon."),
        ("biden", "They're planning to move next month, aren't they?"),
        ("biden", "We've been waiting for hours and she's still not here."),
        ("biden", "It's not that I don't care, it's just complicated."),
        
        # === Technical and complex words ===
        ("biden", "The algorithm optimizes performance through parallelization."),
        ("biden", "Cryptocurrency and blockchain technology are revolutionizing finance."),
        ("biden", "The pharmaceutical company announced a breakthrough in oncology."),
        ("biden", "Artificial neural networks process data through layers of nodes."),
        
        # === Names, places, proper nouns ===
        ("biden", "President Abraham Lincoln gave the Gettysburg Address."),
        ("biden", "The Amazon River flows through Brazil and Peru."),
        ("biden", "Doctor Martin Luther King Junior had a dream."),
        ("biden", "The Eiffel Tower in Paris is three hundred meters tall."),
        
        # === Commands and instructions ===
        ("biden", "Please close the door when you leave the room."),
        ("biden", "Take the first left and then go straight for two blocks."),
        ("biden", "Remember to save your work before shutting down."),
        ("biden", "Don't forget to call your mother on her birthday."),
        
        # === Loli voice - expressive ===
        ("loli", "Hi there! This is so exciting!"),
        ("loli", "I really love this new game, it's super fun!"),
        ("loli", "Can we go to the park? Pretty please?"),
        ("loli", "Wow, look at all the pretty flowers in the garden!"),
        ("loli", "Yay! That's my favorite song ever!"),
        ("loli", "Oh no, what happened here? Are you okay?"),
        ("loli", "I want ice cream! Chocolate please!"),
        ("loli", "Let's play hide and seek together!"),
        
        # === Batman voice - dramatic ===
        ("batman", "I am vengeance. I am the night."),
        ("batman", "The criminal underworld will fear my presence."),
        ("batman", "Justice will be served. No exceptions."),
        ("batman", "Gotham needs a hero. Someone has to step up."),
        ("batman", "This ends tonight. I will find you."),
        ("batman", "I made a promise to protect this city."),
        ("batman", "The shadows are my domain."),
        ("batman", "Every criminal in this city knows my name."),
        
        # === Longer passages ===
        ("biden", "This is a longer passage designed to test the system's ability to handle extended content. It includes multiple sentences with various punctuation marks."),
        ("biden", "The weather forecast for today shows partly cloudy skies with a high of seventy five degrees and a chance of rain in the evening hours."),
        ("loli", "Once upon a time, there was a little bunny who lived in a cozy burrow. Every day, the bunny would hop through the meadow looking for yummy carrots."),
        ("batman", "In the shadows of Gotham City, evil lurks in every corner. But the darkness that criminals embrace is the same darkness that will become their undoing."),
        
        # === Edge cases ===
        ("biden", "One."),
        ("biden", "Yes."),
        ("biden", "Hello?"),
        ("biden", "Hmm..."),
        ("biden", "Okay, okay, okay."),
        ("loli", "Hahaha!"),
        ("loli", "Ehh?"),
        ("batman", "Hmph."),
        ("batman", "Go."),
        
        # === Difficult sounds ===
        ("biden", "She sells seashells by the seashore."),
        ("biden", "Peter Piper picked a peck of pickled peppers."),
        ("biden", "How much wood would a woodchuck chuck?"),
        ("biden", "Unique New York, unique New York."),
    ]
    
    stt_results = []
    log(f"\n  {'#':>3} {'Voice':<8} {'Sim':>6} {'Diagnosis':<28} Text")
    log(f"  {'-'*3} {'-'*8} {'-'*6} {'-'*28} {'-'*30}")
    
    for i, (voice, text) in enumerate(stt_tests, 1):
        try:
            pcm_bytes, _, metadata = call_tts_api(text, voice)
            audio_metrics = analyze_audio(pcm_bytes, metadata["sample_rate"])
            wav_bytes = pcm_to_wav(pcm_bytes, metadata["sample_rate"])
            transcribed = call_stt_api(wav_bytes)
            similarity = text_similarity(text, transcribed)
            
            # Classify the failure type
            word_count = len(text.split())
            diagnosis = classify_failure(text, transcribed, similarity, audio_metrics, word_count)
            
            stt_results.append({
                "voice": voice, 
                "original": text, 
                "transcribed": transcribed, 
                "similarity": similarity,
                "diagnosis": diagnosis,
                "audio_metrics": audio_metrics,
            })
            
            # Truncate text for display
            text_short = text[:30] + "..." if len(text) > 30 else text
            log(f"  {i:>3} {voice:<8} {similarity:>5.0%} {diagnosis:<28} {text_short}")
            
            # Save audio for inspection
            safe_text = text[:20].replace(" ", "_").replace(".", "").replace("!", "").replace("?", "").replace(",", "")
            wav_path = OUTPUT_DIR / f"stt_{i:02d}_{voice}_{safe_text}.wav"
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            
            if similarity < 0.6:
                all_passed = False
                
        except Exception as e:
            log(f"  {i:>3} {voice:<8}    -- 🚫 ERROR: {str(e)[:40]}")
            all_passed = False
    
    avg_similarity = sum(r["similarity"] for r in stt_results) / len(stt_results) if stt_results else 0
    log(f"\n  Average Similarity: {avg_similarity:.1%}")
    
    # Group failures by diagnosis type
    failures = [r for r in stt_results if r.get("similarity", 1.0) < 0.6]
    if failures:
        log(f"\n  ⚠️ Failed Tests ({len(failures)}):")
        # Group by diagnosis
        by_diagnosis = {}
        for r in failures:
            diag = r.get("diagnosis", "Unknown")
            if diag not in by_diagnosis:
                by_diagnosis[diag] = []
            by_diagnosis[diag].append(r)
        
        for diag, items in by_diagnosis.items():
            log(f"\n    {diag}:")
            for r in items[:3]:  # Show max 3 examples per category
                log(f"      • [{r['voice']}] \"{r['original'][:35]}...\" -> \"{r['transcribed'][:25]}...\" ({r['similarity']:.0%})")
            if len(items) > 3:
                log(f"      ... and {len(items) - 3} more")
    
    # 4. Summary
    print_header("VALIDATION SUMMARY")
    log(f"\n  Performance:")
    log(f"    • Overall Speed: {overall_speed:.2f}x real-time")
    log(f"    • Total Audio Generated: {total_audio:.2f}s")
    log(f"    • Total Compute Time: {total_compute:.2f}s")
    log(f"    • Stage Breakdown: LLM {llm_pct:.0f}% | Flow {flow_pct:.0f}%")
    log(f"\n  Quality:")
    log(f"    • STT Round-Trip Tests: {len(stt_tests)}")
    log(f"    • Average Accuracy: {avg_similarity:.1%}")
    log(f"    • Tests Passed: {len([r for r in stt_results if r['similarity'] >= 0.6])}/{len(stt_results)}")
    log(f"    • Tests Failed: {len(failures)}")
    
    log(f"\n  {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Save report
    report_path = OUTPUT_DIR / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n  Report saved to: {report_path}")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_validation())
