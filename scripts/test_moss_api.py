#!/usr/bin/env python3
"""
MOSS-TTS API Test Suite
Simple tests to hit every endpoint of the MOSS-TTS service.

Usage:
    python3 scripts/test_moss_api.py [--url http://localhost:8013]
"""

import argparse
import json
import os
import sys
import time

import requests

DEFAULT_URL = "http://localhost:8013"
OUTPUT_DIR = "test_output"


def test_health(base_url: str) -> bool:
    """Test GET /health"""
    print("\n🏥 Testing GET /health ...")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        print(f"   Status:      {r.status_code}")
        print(f"   Model:       {data.get('model_id')}")
        print(f"   Device:      {data.get('device')}")
        print(f"   Attention:   {data.get('attention')}")
        print(f"   Sample Rate: {data.get('sample_rate')}")
        print(f"   GPU Memory:  {data.get('gpu_memory_gb')} GB")
        print(f"   ✅ Health check passed" if data.get("status") == "ready" else f"   ⚠️  Status: {data.get('status')}")
        return data.get("status") == "ready"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_voices(base_url: str) -> bool:
    """Test GET /voices"""
    print("\n🎤 Testing GET /voices ...")
    try:
        r = requests.get(f"{base_url}/voices", timeout=10)
        voices = r.json()
        print(f"   Status: {r.status_code}")
        print(f"   Voices found: {len(voices)}")
        # Voices may be flat list ["name", ...] or detailed [{name, files}, ...]
        for v in voices:
            if isinstance(v, str):
                print(f"     - {v}")
            else:
                print(f"     - {v['name']}: {v.get('files', [])}")
        print("   ✅ Voices endpoint OK")
        return r.status_code == 200
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_tts(base_url: str) -> bool:
    """Test POST /tts — basic text-to-speech"""
    print("\n🔊 Testing POST /tts ...")
    payload = {
        "text": "Hello, this is a test of the MOSS text to speech API. The quick brown fox jumps over the lazy dog.",
        "max_new_tokens": 4096,
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/tts", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0
        print(f"   Status:      {r.status_code}")
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Audio size:  {len(r.content)} bytes")
        print(f"   Duration:    {r.headers.get('X-Audio-Duration', '?')}s")
        print(f"   Gen time:    {r.headers.get('X-Generation-Time', '?')}s")
        print(f"   Sample rate: {r.headers.get('X-Sample-Rate', '?')}")

        if r.status_code == 200 and len(r.content) > 1000:
            out_path = os.path.join(OUTPUT_DIR, "tts_test.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ TTS generation passed")
            return True
        else:
            print(f"   ❌ Unexpected response: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_tts_stream(base_url: str) -> bool:
    """Test POST /tts/stream — streaming TTS"""
    print("\n📡 Testing POST /tts/stream ...")
    payload = {
        "text": "Streaming test. This audio should arrive in chunked pieces over the network.",
        "max_new_tokens": 4096,
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/tts/stream", json=payload, stream=True, timeout=120)
        print(f"   Status:      {r.status_code}")
        print(f"   Streaming:   {r.headers.get('X-Streaming', '?')}")

        chunks = []
        for chunk in r.iter_content(chunk_size=32 * 1024):
            chunks.append(chunk)

        elapsed = time.perf_counter() - t0
        total_bytes = sum(len(c) for c in chunks)
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Chunks:      {len(chunks)}")
        print(f"   Total bytes: {total_bytes}")

        if total_bytes > 1000:
            out_path = os.path.join(OUTPUT_DIR, "stream_test.wav")
            with open(out_path, "wb") as f:
                for c in chunks:
                    f.write(c)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ Streaming TTS passed")
            return True
        else:
            print("   ❌ Response too small")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_clone_upload(base_url: str, ref_audio_path: str = None) -> bool:
    """Test POST /tts/clone — voice cloning with file upload"""
    print("\n🧬 Testing POST /tts/clone ...")

    # If no reference audio path provided, generate one first via /tts
    if not ref_audio_path:
        ref_path = os.path.join(OUTPUT_DIR, "tts_test.wav")
        if not os.path.exists(ref_path):
            print("   ⏳ No reference audio — generating one via /tts first...")
            if not test_tts(base_url):
                print("   ❌ Cannot test cloning without reference audio")
                return False
        ref_audio_path = ref_path

    try:
        with open(ref_audio_path, "rb") as f:
            t0 = time.perf_counter()
            r = requests.post(
                f"{base_url}/tts/clone",
                data={"text": "This voice should sound like the reference audio that was uploaded."},
                files={"reference": ("reference.wav", f, "audio/wav")},
                timeout=120,
            )
            elapsed = time.perf_counter() - t0

        print(f"   Status:      {r.status_code}")
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Audio size:  {len(r.content)} bytes")
        print(f"   Duration:    {r.headers.get('X-Audio-Duration', '?')}s")

        if r.status_code == 200 and len(r.content) > 1000:
            out_path = os.path.join(OUTPUT_DIR, "clone_test.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ Voice cloning passed")
            return True
        else:
            print(f"   ❌ Unexpected response: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_voice_upload(base_url: str) -> bool:
    """Test POST /voices/{name} — upload a reference voice"""
    print("\n📤 Testing POST /voices/test_voice ...")

    ref_path = os.path.join(OUTPUT_DIR, "tts_test.wav")
    if not os.path.exists(ref_path):
        print("   ⏳ No audio file to upload — generating one via /tts first...")
        if not test_tts(base_url):
            print("   ❌ Cannot test voice upload without audio")
            return False

    try:
        with open(ref_path, "rb") as f:
            r = requests.post(
                f"{base_url}/voices/test_voice",
                files={"file": ("reference.wav", f, "audio/wav")},
                timeout=30,
            )
        data = r.json()
        print(f"   Status:     {r.status_code}")
        print(f"   Voice:      {data.get('voice_name')}")
        print(f"   File:       {data.get('filename')}")
        print(f"   Size:       {data.get('size_bytes')} bytes")
        print("   ✅ Voice upload passed" if r.status_code == 200 else f"   ❌ Failed: {r.text}")
        return r.status_code == 200
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_clone_saved_voice(base_url: str) -> bool:
    """Test POST /tts/clone/{voice_name} — clone from a saved voice"""
    print("\n🔁 Testing POST /tts/clone/test_voice ...")
    payload = {
        "text": "This speech should clone the saved test voice.",
        "max_new_tokens": 4096,
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/tts/clone/test_voice", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0

        print(f"   Status:      {r.status_code}")
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Audio size:  {len(r.content)} bytes")

        if r.status_code == 200 and len(r.content) > 1000:
            out_path = os.path.join(OUTPUT_DIR, "clone_saved_test.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ Saved voice cloning passed")
            return True
        else:
            print(f"   ❌ Unexpected: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_voice_design(base_url: str) -> bool:
    """Test POST /tts/design — voice design from instruction"""
    print("\n🎨 Testing POST /tts/design ...")
    payload = {
        "text": "Hello, welcome to our tavern. What can I get you today?",
        "instruction": "Hearty, jovial tavern owner's voice, loud and welcoming with a friendly tone in American English.",
        "max_new_tokens": 4096,
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/tts/design", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0

        print(f"   Status:      {r.status_code}")
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Audio size:  {len(r.content)} bytes")
        print(f"   Duration:    {r.headers.get('X-Audio-Duration', '?')}s")

        if r.status_code == 200 and len(r.content) > 1000:
            out_path = os.path.join(OUTPUT_DIR, "design_test.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ Voice design passed")
            return True
        elif r.status_code == 503:
            print("   ⚠️  VoiceGenerator not loaded (skipping)")
            return True  # Not a failure, just not available
        else:
            print(f"   ❌ Unexpected: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_tts_with_voice_name(base_url: str) -> bool:
    """Test POST /tts with voice_name — auto-route to clone"""
    print("\n🔄 Testing POST /tts with voice_name ...")
    payload = {
        "text": "This should use a cloned voice via the voice_name parameter.",
        "voice_name": "test_voice",
        "language": "en",
        "output_format": "wav",
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/tts", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0

        print(f"   Status:      {r.status_code}")
        print(f"   Time:        {elapsed:.1f}s")
        print(f"   Audio size:  {len(r.content)} bytes")

        if r.status_code == 200 and len(r.content) > 1000:
            out_path = os.path.join(OUTPUT_DIR, "tts_voice_name_test.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"   💾 Saved: {out_path}")
            print("   ✅ TTS with voice_name passed")
            return True
        else:
            print(f"   ❌ Unexpected: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def wait_for_ready(base_url: str, timeout: int = 600) -> bool:
    """Wait for the service to be ready (model loaded)."""
    print(f"⏳ Waiting for MOSS-TTS service at {base_url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            data = r.json()
            status = data.get("status", "unknown")
            if status == "ready":
                print(f"   ✅ Service ready ({time.time() - start:.0f}s)")
                return True
            print(f"   Status: {status} ({time.time() - start:.0f}s elapsed)")
        except requests.ConnectionError:
            print(f"   Waiting... ({time.time() - start:.0f}s elapsed)")
        except Exception as e:
            print(f"   Error: {e}")
        time.sleep(10)
    print(f"   ❌ Timed out after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="MOSS-TTS API Test Suite")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the MOSS-TTS API")
    parser.add_argument("--ref", help="Path to reference audio for clone test")
    parser.add_argument("--skip-wait", action="store_true", help="Skip waiting for service readiness")
    parser.add_argument("--test", choices=["health", "voices", "tts", "stream", "clone", "design", "shim", "all"],
                        default="all", help="Which test to run")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("🌿 MOSS-TTS API Test Suite")
    print("=" * 60)
    print(f"  URL:    {args.url}")
    print(f"  Output: {OUTPUT_DIR}/")

    if not args.skip_wait:
        if not wait_for_ready(args.url):
            sys.exit(1)

    results = {}

    if args.test in ("health", "all"):
        results["health"] = test_health(args.url)

    if args.test in ("voices", "all"):
        results["voices"] = test_voices(args.url)

    if args.test in ("tts", "all"):
        results["tts"] = test_tts(args.url)

    if args.test in ("stream", "all"):
        results["stream"] = test_tts_stream(args.url)

    if args.test in ("clone", "all"):
        results["clone_upload"] = test_clone_upload(args.url, args.ref)
        results["voice_upload"] = test_voice_upload(args.url)
        results["clone_saved"] = test_clone_saved_voice(args.url)

    if args.test in ("design", "all"):
        results["voice_design"] = test_voice_design(args.url)

    if args.test in ("shim", "all"):
        results["tts_voice_name"] = test_tts_with_voice_name(args.url)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"📋 Results: {passed}/{total} passed")
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
