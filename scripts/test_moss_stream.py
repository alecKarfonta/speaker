#!/usr/bin/env python3
"""
MOSS TTS Streaming Endpoint Test

Tests the /tts/stream endpoint by:
1. Sending text to the streaming endpoint
2. Parsing the binary framing (4B audio_len + 4B meta_len + audio + meta)
3. Concatenating all WAV chunks into a single audio file
4. Sending to STT API for transcription
5. Comparing original vs transcribed text
"""

import argparse
import io
import json
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests


def stream_tts(text: str, voice: str = None, api_url: str = "http://localhost:8013") -> tuple[bytes, list[dict]]:
    """Call /tts/stream and parse the binary framing protocol."""
    payload = {"text": text, "language": "en"}
    if voice:
        payload["voice_name"] = voice

    t0 = time.perf_counter()
    response = requests.post(f"{api_url}/tts/stream", json=payload, stream=True, timeout=300)
    response.raise_for_status()

    raw = response.content
    elapsed = time.perf_counter() - t0

    # Parse binary frames: [4B audio_len LE] [4B meta_len LE] [audio_bytes] [meta_json]
    offset = 0
    all_pcm = bytearray()
    metadata_chunks = []
    sample_rate = 24000

    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8

        if audio_len > 0:
            wav_data = raw[offset:offset + audio_len]
            offset += audio_len
            # Extract PCM from WAV container
            try:
                with wave.open(io.BytesIO(wav_data), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    all_pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                # Raw PCM fallback
                all_pcm.extend(wav_data)
        else:
            offset += audio_len  # skip 0 audio

        if meta_len > 0:
            meta_json = raw[offset:offset + meta_len]
            offset += meta_len
            try:
                meta = json.loads(meta_json)
                metadata_chunks.append(meta)
            except json.JSONDecodeError:
                pass

    # Build final WAV
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(all_pcm))

    wav_bytes = wav_buf.getvalue()
    audio_duration = len(all_pcm) / (sample_rate * 2)  # 16-bit = 2 bytes/sample

    # Gather metadata
    final_meta = metadata_chunks[-1] if metadata_chunks else {}
    total_gen_time = final_meta.get("total_generation_time", elapsed)
    total_audio_dur = final_meta.get("total_audio_duration", audio_duration)
    num_chunks = final_meta.get("chunk_index", len(metadata_chunks))
    error = final_meta.get("error")

    print(f"  Chunks:     {num_chunks}")
    print(f"  Audio:      {total_audio_dur:.1f}s")
    print(f"  Gen time:   {total_gen_time:.1f}s")
    if total_audio_dur > 0:
        print(f"  RTF:        {total_gen_time/total_audio_dur:.2f}x")
    if error:
        print(f"  ⚠ ERROR:    {error}")

    return wav_bytes, metadata_chunks


def transcribe_stt(wav_bytes: bytes, stt_url: str = "http://localhost:8603/v1/audio/transcriptions") -> str:
    """Transcribe WAV audio using the STT service."""
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    data = {"model": "base", "language": "en"}
    response = requests.post(stt_url, files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json().get("text", "").strip()


def word_similarity(original: str, transcribed: str) -> float:
    """Calculate word-level overlap between original and transcribed text."""
    orig_words = set(original.lower().split())
    trans_words = set(transcribed.lower().split())
    if not orig_words:
        return 0.0
    return len(orig_words & trans_words) / len(orig_words)


def main():
    parser = argparse.ArgumentParser(description="Test MOSS TTS /tts/stream endpoint")
    parser.add_argument("--api-url", default="http://localhost:8013", help="MOSS TTS API URL")
    parser.add_argument("--stt-url", default="http://localhost:8603/v1/audio/transcriptions", help="STT API URL")
    parser.add_argument("--text", "-t", default="Hello, this is a test of the streaming text to speech system.",
                        help="Text to synthesize")
    parser.add_argument("--voice", "-v", default=None, help="Voice name (optional)")
    parser.add_argument("--save", "-s", default=None, help="Save audio to this path")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT transcription")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 MOSS TTS Streaming Test")
    print("=" * 60)
    print(f"  Text:  {args.text}")
    print(f"  Voice: {args.voice or '(default)'}")
    print(f"  API:   {args.api_url}")
    print()

    # 1. Stream TTS
    print("▶ Streaming TTS...")
    try:
        wav_bytes, metadata = stream_tts(args.text, args.voice, args.api_url)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return 1

    # Check for errors in metadata
    for m in metadata:
        if m.get("error"):
            print(f"\n  ✗ Stream error: {m['error']}")
            return 1

    if len(wav_bytes) < 100:
        print("  ✗ FAILED: Empty or too-small audio output")
        return 1

    # Save if requested
    if args.save:
        Path(args.save).write_bytes(wav_bytes)
        print(f"  Saved: {args.save}")

    # 2. STT Transcription
    if not args.skip_stt:
        print("\n▶ Transcribing with STT...")
        try:
            transcribed = transcribe_stt(wav_bytes, args.stt_url)
            print(f"  Transcribed: {transcribed}")
        except Exception as e:
            print(f"  ⚠ STT failed: {e}")
            transcribed = None

        if transcribed:
            sim = word_similarity(args.text, transcribed)
            print(f"\n{'=' * 60}")
            print(f"  Original:    {args.text}")
            print(f"  Transcribed: {transcribed}")
            print(f"  Similarity:  {sim * 100:.0f}%")
            print(f"{'=' * 60}")

            if sim >= 0.7:
                print("\n✓ PASS: Streaming TTS output is intelligible!")
                return 0
            elif sim >= 0.4:
                print("\n⚠ PARTIAL: Some words recognized but quality issues.")
                return 1
            else:
                print("\n✗ FAIL: Output is gibberish/unintelligible.")
                return 2

    print("\n✓ Stream completed successfully (STT skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
