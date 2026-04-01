#!/usr/bin/env python3
"""
Qwen3-TTS Streaming → STT Round-Trip Test

Tests:
  1. Hits /tts/stream with a cloned voice
  2. Reassembles framed binary chunks into a single WAV file
  3. Transcribes with the STT API (or local Whisper fallback)
  4. Computes WER / similarity and reports TTFB

Usage:
    python3 scripts/test_qwen_streaming_stt.py
    python3 scripts/test_qwen_streaming_stt.py --voice Vivian --text "Hello world test"
    python3 scripts/test_qwen_streaming_stt.py --custom-voice  # use Vivian (custom_voice mode)
"""

import argparse
import io
import json
import struct
import sys
import time
import wave
from pathlib import Path

import requests

# ─── Defaults ─────────────────────────────────────────────────────────────────
QWEN_URL     = "http://localhost:8012"
STT_URL      = "http://192.168.1.196:8603/v1/audio/transcriptions"
STT_KEY      = "stt-api-key"
STT_MODEL    = "medium"

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, this is a streaming text to speech test.",
    "In a world without walls and fences, who needs windows and gates?",
    "The rain in Spain falls mainly on the plain.",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def assemble_chunks(raw_stream: bytes) -> tuple[bytes, dict]:
    """Parse framed binary stream into raw PCM + metadata list.
    
    Frame format: [4B audio_len LE][4B meta_len LE][audio bytes][JSON bytes]
    """
    offset = 0
    pcm_chunks = []
    meta_list  = []
    sample_rate = 24000  # Qwen default

    while offset < len(raw_stream):
        if offset + 8 > len(raw_stream):
            break
        audio_len, meta_len = struct.unpack_from("<II", raw_stream, offset)
        offset += 8

        if offset + audio_len + meta_len > len(raw_stream):
            print(f"  [WARN] truncated frame at offset {offset}", file=sys.stderr)
            break

        audio_bytes = raw_stream[offset: offset + audio_len]
        offset += audio_len
        meta_bytes  = raw_stream[offset: offset + meta_len]
        offset += meta_len

        try:
            meta = json.loads(meta_bytes)
            meta_list.append(meta)
            if "sample_rate" in meta:
                sample_rate = meta["sample_rate"]
        except Exception:
            meta = {}

        if audio_len > 0:
            # audio_bytes is a complete WAV mini-file (44-byte header + PCM)
            # Strip the 44-byte WAV header to get raw int16 PCM
            if audio_bytes[:4] == b"RIFF":
                pcm_chunks.append(audio_bytes[44:])
            else:
                pcm_chunks.append(audio_bytes)

    raw_pcm = b"".join(pcm_chunks)
    return raw_pcm, sample_rate, meta_list


def pcm_to_wav(raw_pcm: bytes, sample_rate: int, channels: int = 1,
               sample_width: int = 2) -> bytes:
    """Wrap raw int16 PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def transcribe_wav(wav_bytes: bytes) -> str:
    """Send WAV bytes to STT API, fall back to local Whisper."""
    try:
        resp = requests.post(
            STT_URL,
            headers={"Authorization": f"Bearer {STT_KEY}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": STT_MODEL, "language": "en"},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        print(f"  [STT-API] error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
    except requests.exceptions.ConnectionError:
        print(f"  [STT-API] not reachable at {STT_URL}", file=sys.stderr)

    # Local whisper fallback
    print("  Falling back to local Whisper…")
    try:
        import whisper
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper", "-q"])
        import whisper

    try:
        tmp = Path("/tmp/qwen_test_audio.wav")
        tmp.write_bytes(wav_bytes)
        model = whisper.load_model("base")
        result = model.transcribe(str(tmp))
        tmp.unlink(missing_ok=True)
        return result["text"].strip()
    except Exception as e:
        print(f"  [Whisper] error: {e}", file=sys.stderr)
        return "[Transcription failed]"


def word_similarity(original: str, transcribed: str) -> float:
    orig_words = set(original.lower().split())
    trans_words = set(transcribed.lower().split())
    if not orig_words:
        return 0.0
    return len(orig_words & trans_words) / len(orig_words)


def word_error_rate(ref: str, hyp: str) -> float:
    """Simple WER: edit-distance / len(ref_words)."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    return dp[m] / max(n, 1)


# ─── Core test ────────────────────────────────────────────────────────────────

def run_stream_test(text: str, voice: str, mode: str, base_url: str, save_audio: bool) -> dict:
    """Run one streaming TTS → STT test. Returns result dict."""

    payload = {
        "text": text,
        "voice_name": voice,
        "language": "en",
        "output_format": "wav",
    }

    print(f"\n  Text   : {text}")
    print(f"  Voice  : {voice} ({mode} mode)")

    t_start = time.perf_counter()

    try:
        resp = requests.post(
            f"{base_url}/tts/stream",
            json=payload,
            stream=True,
            timeout=120,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    # Stream, tracking TTFB on first chunk
    raw_chunks = []
    ttfb_ms = None
    bytes_received = 0

    for chunk in resp.iter_content(chunk_size=16384):
        if chunk:
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t_start) * 1000
            raw_chunks.append(chunk)
            bytes_received += len(chunk)

    t_done = time.perf_counter()
    total_ms = (t_done - t_start) * 1000

    raw_stream = b"".join(raw_chunks)
    print(f"  TTFB   : {ttfb_ms:.0f}ms")
    print(f"  Total  : {total_ms:.0f}ms  ({bytes_received/1024:.1f} KB)")

    # Assemble chunks
    raw_pcm, sample_rate, meta_list = assemble_chunks(raw_stream)
    audio_dur_s = len(raw_pcm) / 2 / sample_rate  # int16 = 2 bytes/sample
    streaming_mode = meta_list[0].get("streaming", "unknown") if meta_list else "unknown"
    print(f"  Chunks : {len(meta_list)}  Audio: {audio_dur_s:.2f}s  SR: {sample_rate}  mode: {streaming_mode}")

    if len(raw_pcm) == 0:
        return {"ok": False, "error": "Empty audio received"}

    wav_bytes = pcm_to_wav(raw_pcm, sample_rate)

    if save_audio:
        out_root = Path("qwen_samples")
        out_root.mkdir(parents=True, exist_ok=True)
        # Use simple slug
        slug = "".join(c if c.isalnum() else "_" for c in text[:20]).strip("_")
        out_path = out_root / f"qwen_{voice}_{slug}.wav"
        out_path.write_bytes(wav_bytes)
        print(f"  Saved  : {out_path}")

    # STT
    print("  Transcribing…")
    t_stt = time.perf_counter()
    transcribed = transcribe_wav(wav_bytes)
    stt_ms = (time.perf_counter() - t_stt) * 1000

    sim = word_similarity(text, transcribed)
    wer = word_error_rate(text, transcribed)

    print(f"  STT    : \"{transcribed}\"  ({stt_ms:.0f}ms)")
    print(f"  WER    : {wer*100:.1f}%   Similarity: {sim*100:.1f}%")

    result = dict(
        ok=True,
        text=text,
        voice=voice,
        ttfb_ms=round(ttfb_ms or 0, 1),
        total_ms=round(total_ms, 1),
        audio_dur_s=round(audio_dur_s, 2),
        chunk_count=len(meta_list),
        streaming_mode=streaming_mode,
        transcribed=transcribed,
        wer=round(wer, 3),
        similarity=round(sim, 3),
        pass_=sim >= 0.8,
    )
    return result


# ─── Health check ─────────────────────────────────────────────────────────────

def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    print(f"Waiting for server at {base_url}…", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                print(" ready ✓")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" TIMEOUT ✗")
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Streaming → STT Round-trip Test")
    parser.add_argument("--base-url",  default=QWEN_URL)
    parser.add_argument("--voice",     default="Vivian",
                        help="Voice name (Vivian, Ryan, etc.)")
    parser.add_argument("--text",      default=None,
                        help="Single test sentence (default: run all built-in sentences)")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save reassembled WAV to /tmp/")
    parser.add_argument("--no-wait",   action="store_true",
                        help="Skip server readiness poll")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print("=" * 62)
    print("  Qwen3-TTS  Streaming → STT  Round-Trip Test")
    print("=" * 62)

    if not args.no_wait:
        if not wait_for_server(base_url):
            sys.exit(1)

    # Check backend
    try:
        hresp = requests.get(f"{base_url}/health", timeout=5).json()
        print(f"Backend: {hresp.get('backend', '?')}  Model: {hresp.get('model', '?')}")
    except Exception as e:
        print(f"[WARN] could not read health: {e}")

    sentences = [args.text] if args.text else TEST_SENTENCES
    voice = args.voice

    all_results = []
    for text in sentences:
        result = run_stream_test(
            text=text,
            voice=voice,
            mode="auto",
            base_url=base_url,
            save_audio=args.save_audio,
        )
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    passed = sum(1 for r in all_results if r.get("pass_"))
    failed = len(all_results) - passed
    errors = sum(1 for r in all_results if not r.get("ok"))

    ttfbs   = [r["ttfb_ms"]   for r in all_results if r.get("ok")]
    wers    = [r["wer"]       for r in all_results if r.get("ok")]
    sims    = [r["similarity"] for r in all_results if r.get("ok")]
    modes   = list({r.get("streaming_mode") for r in all_results if r.get("ok")})

    print(f"  Sentences  : {len(all_results)}")
    print(f"  Passed (≥80% sim): {passed}   Failed: {failed}   Errors: {errors}")
    if ttfbs:
        print(f"  TTFB avg   : {sum(ttfbs)/len(ttfbs):.0f}ms   min: {min(ttfbs):.0f}ms   max: {max(ttfbs):.0f}ms")
    if wers:
        print(f"  WER avg    : {sum(wers)/len(wers)*100:.1f}%")
    if sims:
        print(f"  Similarity : {sum(sims)/len(sims)*100:.1f}%")
    if modes:
        mode_str = ", ".join(str(m) for m in modes)
        print(f"  Stream mode: {mode_str}")
    print("=" * 62)

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
