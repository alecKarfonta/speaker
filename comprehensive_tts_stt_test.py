#!/usr/bin/env python3

import requests
import json
import subprocess
import numpy as np
import time
from pathlib import Path

def convert_wav_to_stt_format(wav_file):
    """Convert WAV file to the format expected by STT service"""
    cmd = [
        'ffmpeg', '-i', wav_file, '-f', 'f64le', '-acodec', 'pcm_f64le', 
        '-ac', '1', '-ar', '44100', '-'
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise Exception(f"FFmpeg conversion failed: {result.stderr.decode()}")
    
    # Convert raw audio data to numpy array
    audio_data = np.frombuffer(result.stdout, dtype=np.float64)
    
    # Convert to int16 format expected by STT service
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16.tolist()

def test_tts_stt_pipeline():
    """Test the complete TTS-STT pipeline"""
    
    # Get available voices
    print("🎤 Getting available voices...")
    voices_response = requests.get('http://localhost:8005/voices')
    if voices_response.status_code != 200:
        print(f"❌ Failed to get voices: {voices_response.status_code}")
        return
    
    voices = voices_response.json()
    print(f"✅ Found {len(voices)} voices: {', '.join(voices)}")
    
    # Test cases
    test_cases = [
        "Hello world",
        "This is a test",
        "The quick brown fox",
        "Testing one two three",
        "Speech synthesis test"
    ]
    
    results = []
    
    print(f"\n🧪 Testing {len(test_cases)} phrases across {len(voices)} voices...")
    print("=" * 80)
    
    for voice in voices:
        print(f"\n🎭 Testing voice: {voice}")
        voice_results = []
        
        for i, text in enumerate(test_cases):
            print(f"  📝 Test {i+1}: \"{text}\"")
            
            try:
                # Generate TTS
                start_time = time.time()
                tts_response = requests.post('http://localhost:8005/tts', json={
                    'text': text,
                    'voice_name': voice,
                    'language': 'en'
                }, timeout=30)
                
                if tts_response.status_code != 200:
                    print(f"    ❌ TTS failed: {tts_response.status_code}")
                    continue
                
                tts_time = time.time() - start_time
                
                # Save audio file
                audio_file = f"test_{voice}_{i+1}.wav"
                with open(audio_file, 'wb') as f:
                    f.write(tts_response.content)
                
                # Convert to STT format
                audio_list = convert_wav_to_stt_format(audio_file)
                
                # Send to STT
                stt_payload = {
                    'wav': audio_list,
                    'sample_rate': 44100,
                    'channels': 1,
                    'source_language': 'en'
                }
                
                start_time = time.time()
                stt_response = requests.post('http://localhost:8000/transcribe', 
                                           json=stt_payload, timeout=60)
                stt_time = time.time() - start_time
                
                if stt_response.status_code != 200:
                    print(f"    ❌ STT failed: {stt_response.status_code}")
                    continue
                
                result = stt_response.json()
                transcribed = result['text'].strip()
                language_confidence = result['language_probability']
                
                # Calculate accuracy
                original_words = text.lower().split()
                transcribed_words = transcribed.lower().split()
                
                word_accuracy = 0
                if len(transcribed_words) > 0:
                    matches = sum(1 for word in original_words if word in transcribed_words)
                    word_accuracy = matches / len(original_words) * 100
                
                # Character accuracy
                char_accuracy = 0
                if len(transcribed) > 0:
                    import difflib
                    matcher = difflib.SequenceMatcher(None, text.lower(), transcribed.lower())
                    char_accuracy = matcher.ratio() * 100
                
                # Store result
                test_result = {
                    'voice': voice,
                    'original': text,
                    'transcribed': transcribed,
                    'word_accuracy': word_accuracy,
                    'char_accuracy': char_accuracy,
                    'language_confidence': language_confidence,
                    'tts_time': tts_time,
                    'stt_time': stt_time
                }
                
                voice_results.append(test_result)
                
                print(f"    ✅ \"{transcribed}\" | Word: {word_accuracy:.1f}% | Char: {char_accuracy:.1f}% | Lang: {language_confidence:.2f}")
                
                # Clean up
                Path(audio_file).unlink(missing_ok=True)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        results.extend(voice_results)
        
        # Voice summary
        if voice_results:
            avg_word_acc = sum(r['word_accuracy'] for r in voice_results) / len(voice_results)
            avg_char_acc = sum(r['char_accuracy'] for r in voice_results) / len(voice_results)
            avg_tts_time = sum(r['tts_time'] for r in voice_results) / len(voice_results)
            avg_stt_time = sum(r['stt_time'] for r in voice_results) / len(voice_results)
            
            print(f"  📊 {voice} Summary: Word={avg_word_acc:.1f}% | Char={avg_char_acc:.1f}% | TTS={avg_tts_time:.3f}s | STT={avg_stt_time:.2f}s")
    
    # Overall summary
    if results:
        print(f"\n📈 OVERALL RESULTS ({len(results)} tests)")
        print("=" * 80)
        
        overall_word_acc = sum(r['word_accuracy'] for r in results) / len(results)
        overall_char_acc = sum(r['char_accuracy'] for r in results) / len(results)
        overall_tts_time = sum(r['tts_time'] for r in results) / len(results)
        overall_stt_time = sum(r['stt_time'] for r in results) / len(results)
        overall_lang_conf = sum(r['language_confidence'] for r in results) / len(results)
        
        print(f"Average Word Accuracy: {overall_word_acc:.1f}%")
        print(f"Average Character Accuracy: {overall_char_acc:.1f}%")
        print(f"Average TTS Time: {overall_tts_time:.3f}s")
        print(f"Average STT Time: {overall_stt_time:.2f}s")
        print(f"Average Language Confidence: {overall_lang_conf:.3f}")
        
        # Best and worst results
        best_word = max(results, key=lambda r: r['word_accuracy'])
        worst_word = min(results, key=lambda r: r['word_accuracy'])
        
        print(f"\n🏆 Best Result: {best_word['voice']} - \"{best_word['original']}\" → \"{best_word['transcribed']}\" ({best_word['word_accuracy']:.1f}%)")
        print(f"🚫 Worst Result: {worst_word['voice']} - \"{worst_word['original']}\" → \"{worst_word['transcribed']}\" ({worst_word['word_accuracy']:.1f}%)")
        
        # Performance by voice
        print(f"\n🎭 Performance by Voice:")
        voice_stats = {}
        for result in results:
            voice = result['voice']
            if voice not in voice_stats:
                voice_stats[voice] = []
            voice_stats[voice].append(result['word_accuracy'])
        
        for voice, accuracies in sorted(voice_stats.items()):
            avg_acc = sum(accuracies) / len(accuracies)
            print(f"  {voice}: {avg_acc:.1f}% (n={len(accuracies)})")

if __name__ == "__main__":
    test_tts_stt_pipeline() 