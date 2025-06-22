#!/usr/bin/env python3
"""
TTS + STT Diagnostic Tool
Tests TTS generation and uses STT to verify if audio contains actual speech
"""

import requests
import json
import time
import os
import sys
import wave
import numpy as np
import speech_recognition as sr
import whisper
from pydub import AudioSegment
from pydub.utils import which
import io

# API Configuration
API_BASE = "http://localhost:8010"

def analyze_audio_properties(audio_file):
    """Analyze basic audio properties to detect silence"""
    print(f"📊 Analyzing audio properties for {audio_file}")
    
    try:
        # Load with pydub for analysis
        audio = AudioSegment.from_wav(audio_file)
        
        # Get audio properties
        duration = len(audio) / 1000.0  # Convert to seconds
        channels = audio.channels
        frame_rate = audio.frame_rate
        sample_width = audio.sample_width
        
        # Get raw audio data
        raw_data = audio.get_array_of_samples()
        audio_array = np.array(raw_data)
        
        # Calculate statistics
        max_amplitude = np.max(np.abs(audio_array))
        rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
        dynamic_range = np.max(audio_array) - np.min(audio_array)
        
        # Detect if audio is essentially silent
        max_possible = 2**(sample_width * 8 - 1) - 1
        is_silent = max_amplitude < (max_possible * 0.001)  # Less than 0.1% of max
        
        print(f"   Duration: {duration:.2f}s")
        print(f"   Channels: {channels}, Sample Rate: {frame_rate}Hz")
        print(f"   Max Amplitude: {max_amplitude:,} (of {max_possible:,})")
        print(f"   RMS: {rms:.2f}")
        print(f"   Dynamic Range: {dynamic_range:,}")
        print(f"   Is Silent: {'🔇 YES' if is_silent else '🔊 NO'}")
        
        return {
            'duration': duration,
            'max_amplitude': max_amplitude,
            'rms': rms,
            'is_silent': is_silent,
            'dynamic_range': dynamic_range
        }
        
    except Exception as e:
        print(f"❌ Error analyzing audio: {str(e)}")
        return None

def transcribe_with_whisper(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    print(f"🎤 Transcribing with Whisper: {audio_file}")
    
    try:
        # Load Whisper model (using small model for speed)
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(audio_file)
        
        text = result["text"].strip()
        confidence = result.get("language_probability", 0)
        
        print(f"   Text: '{text}'")
        print(f"   Language: {result.get('language', 'unknown')}")
        print(f"   Confidence: {confidence:.3f}")
        
        return text, confidence
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {str(e)}")
        return None, 0

def transcribe_with_google(audio_file):
    """Transcribe audio using Google Speech Recognition"""
    print(f"🎤 Transcribing with Google STT: {audio_file}")
    
    try:
        r = sr.Recognizer()
        
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        
        # Try Google Speech Recognition
        text = r.recognize_google(audio)
        print(f"   Text: '{text}'")
        return text
        
    except sr.UnknownValueError:
        print("   ⚠️  Google STT could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"   ❌ Google STT error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def test_tts_with_stt_verification(voice_name, text):
    """Test TTS generation and verify with STT"""
    print(f"\n🎯 Testing TTS + STT for voice: {voice_name}")
    print(f"📝 Text: '{text}'")
    
    # Generate TTS
    payload = {
        "text": text,
        "voice_name": voice_name,
        "language": "en",
        "temperature": 0.8,
        "top_p": 0.9,
        "emotion": "",
        "speed": 1.0
    }
    
    filename = f"diagnostic_{voice_name}.wav"
    
    try:
        # Generate TTS
        print("🔄 Generating TTS...")
        start_time = time.time()
        response = requests.post(f"{API_BASE}/tts", json=payload)
        generation_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ TTS failed: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        # Save audio
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"✅ TTS generated: {len(response.content):,} bytes in {generation_time:.3f}s")
        
        # Analyze audio properties
        audio_props = analyze_audio_properties(filename)
        
        if audio_props and audio_props['is_silent']:
            print("🔇 WARNING: Audio appears to be silent!")
            return False
        
        # Transcribe with multiple STT engines
        whisper_text, whisper_conf = transcribe_with_whisper(filename)
        google_text = transcribe_with_google(filename)
        
        # Evaluate results
        success = False
        if whisper_text and len(whisper_text.strip()) > 0:
            print(f"✅ Whisper detected speech: '{whisper_text}'")
            success = True
        
        if google_text and len(google_text.strip()) > 0:
            print(f"✅ Google detected speech: '{google_text}'")
            success = True
        
        if not success:
            print("❌ No speech detected by any STT engine!")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

def main():
    """Run comprehensive TTS + STT diagnostic"""
    print("🔬 TTS + STT Diagnostic Tool")
    print("=" * 50)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"❌ API health check failed: HTTP {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        return
    
    # Get available voices
    try:
        response = requests.get(f"{API_BASE}/voices")
        if response.status_code == 200:
            voices = response.json()["voices"]
            print(f"📋 Available voices: {', '.join(voices)}")
        else:
            print(f"❌ Failed to get voices: HTTP {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting voices: {str(e)}")
        return
    
    # Test cases with different texts and voices
    test_cases = [
        ("dsp", "Hello world, this is a test of text to speech synthesis."),
        ("biden", "My fellow Americans, we must work together for a better future."),
        ("batman", "I am the Dark Knight, protector of Gotham City."),
    ]
    
    results = []
    
    print(f"\n🧪 Running {len(test_cases)} TTS + STT tests...")
    
    for voice, test_text in test_cases:
        if voice in voices:
            success = test_tts_with_stt_verification(voice, test_text)
            results.append((voice, success))
        else:
            print(f"⚠️  Skipping {voice} - not available")
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"✅ Speech Detected: {successful}/{total}")
    print(f"❌ Silent/Empty Audio: {total - successful}/{total}")
    
    if successful == 0:
        print(f"\n🚨 CRITICAL ISSUE: All TTS outputs are silent/empty!")
        print("This indicates a problem with the TTS generation.")
    elif successful < total:
        print(f"\n⚠️  PARTIAL ISSUE: Some voices produce silent audio.")
        print("This may indicate voice-specific problems.")
    else:
        print(f"\n🎉 ALL TESTS PASSED: TTS is generating proper speech!")
    
    print(f"\nVoice Results:")
    for voice, success in results:
        status = "🔊 SPEECH" if success else "🔇 SILENT"
        print(f"  • {voice}: {status}")

if __name__ == "__main__":
    main() 