#!/usr/bin/env python3
"""Batch-generate MOSS voice-clone samples for every reference in data/voices."""

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

TEXTS = [
    "Hello! This is the first clone sample. MOSS-TTS is copying this reference voice.",
    "The rain fell softly on the old city streets. Nobody spoke for a long moment, and then the voice returned, calm and clear as before.",
]

EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
FRAME_RATE_HZ = 12.5
# Tuned via scripts/openmoss_param_sweep.py + STT on :8603 (May 2026)
OPENMOSS_SAMPLING = {
    "text_temperature": 0.75,
    "text_top_p": 0.6,
    "text_top_k": 30,
    "audio_temperature": 0.6,
    "audio_top_p": 0.6,
    "audio_top_k": 30,
    "audio_repetition_penalty": 1.05,
}

# English TTS ~12–14 chars/sec; 1s audio ≈ 12.5 codec frames at 12.5 Hz.
CHARS_PER_SEC = float(os.environ.get("OPENMOSS_CHARS_PER_SEC", "13.0"))

# Per-process cache: avoid ffmpeg+base64 on every utterance (same ref for whole run).
_ref_cache_key: tuple[str, int] | None = None
_ref_wav_bytes: bytes | None = None
_ref_b64: str | None = None
_ref_duration_s: float | None = None
_prepared_ref_wav: Path | None = None
_http_sessions: dict[str, requests.Session] = {}


def openmoss_token_budget_chars(text: str) -> tuple[int, int]:
    """Derive target audio frames + generation cap from text length."""
    slack = float(os.environ.get("OPENMOSS_DUR_SLACK", "1.12"))
    max_sec = float(os.environ.get("OPENMOSS_MAX_SEC", "20.0"))
    extra = int(os.environ.get("OPENMOSS_MAX_EXTRA", "24"))

    est_sec = max(1.2, len(text) / CHARS_PER_SEC) * slack
    est_sec = min(est_sec, max_sec)
    max_codec = int(os.environ.get("OPENMOSS_MAX_CODEC_TOKENS", "400"))
    tokens = max(48, min(max_codec, int(round(est_sec * FRAME_RATE_HZ))))
    return tokens, tokens + extra
OPENMOSS_MODEL = Path(
    os.environ.get(
        "OPENMOSS_MODEL",
        "/home/alec/git/speaker/openmoss/weights/moss-tts-v15-q8_0.gguf",
    )
)
OPENMOSS_CLI = Path("/home/alec/git/speaker/openmoss/build/moss-tts-cli")
LLAMA_LIB = Path("/home/alec/git/llama-nexus/llama.cpp/build/bin")
OPENMOSS_GPU = int(os.environ.get("OPENMOSS_MAIN_GPU", "1"))
REF_MAX_SECONDS = 12


def wav_duration_seconds(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size < 1000:
        return None
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _ref_cache_key_for(ref: Path) -> tuple[str, int]:
    resolved = ref.resolve()
    return str(resolved), resolved.stat().st_mtime_ns


def preload_reference(ref: Path) -> None:
    """Encode reference once per process (call at worker startup)."""
    global _ref_cache_key, _ref_wav_bytes, _ref_b64, _ref_duration_s, _prepared_ref_wav
    key = _ref_cache_key_for(ref)
    if _ref_cache_key == key and _ref_b64 is not None:
        return
    wav = reference_wav_bytes(ref)
    _ref_cache_key = key
    _ref_wav_bytes = wav
    _ref_b64 = base64.b64encode(wav).decode()
    _ref_duration_s = wav_duration_seconds(ref)
    if _prepared_ref_wav is not None and _prepared_ref_wav.is_file():
        _prepared_ref_wav.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="moss_ref_") as tmp:
        _prepared_ref_wav = Path(tmp.name)
    _prepared_ref_wav.write_bytes(wav)


def reference_wav_b64(ref: Path) -> str:
    preload_reference(ref)
    assert _ref_b64 is not None
    return _ref_b64


def openmoss_token_budget(text: str, reference_wav: Path | None) -> tuple[int, int]:
    """Derive tokens + max_new_tokens from text length (reference only caps upward runs)."""
    tokens, max_new = openmoss_token_budget_chars(text)
    if reference_wav is not None:
        if _ref_cache_key == _ref_cache_key_for(reference_wav):
            dur = _ref_duration_s
        else:
            dur = wav_duration_seconds(reference_wav)
        if dur:
            ref_tokens = max(48, min(400, int(round(dur * FRAME_RATE_HZ))))
            tokens = min(tokens, ref_tokens)
            max_new = min(max_new, ref_tokens + int(os.environ.get("OPENMOSS_MAX_EXTRA", "24")))
    return tokens, max_new


DOCKER_CONTAINER_BY_PORT: dict[int, str] = {
    8014: "speaker-openmoss-tts-1",
    8015: "speaker-openmoss-tts-0-1",
    8017: "speaker-openmoss-tts-1-1",
}


def docker_container_for_api(api: str) -> str | None:
    """Map host openmoss shim port → docker container for raw :8081 API."""
    override = os.environ.get("OPENMOSS_DOCKER_CONTAINER")
    if override:
        return override
    return DOCKER_CONTAINER_BY_PORT.get(port_from_api(api))


def generate_openmoss_docker(
    container: str,
    payload: dict,
    out: Path,
) -> tuple[bool, str]:
    """POST JSON to raw moss-tts-server (:8081) inside a docker container."""
    script = """
import json, sys, urllib.request
payload = json.loads(sys.stdin.read())
req = urllib.request.Request(
    "http://127.0.0.1:8081/tts",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req, timeout=600)
sys.stdout.buffer.write(resp.read())
"""
    proc = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", script],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="ignore")[:200]
        return False, detail or "docker raw openmoss failed"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(proc.stdout)
    if out.stat().st_size < 1024:
        out.unlink(missing_ok=True)
        return False, "output too small"
    return True, "raw ok"


def http_session_for_api(api: str) -> requests.Session:
    base = api.rsplit("/", 1)[0]
    if base not in _http_sessions:
        _http_sessions[base] = requests.Session()
    return _http_sessions[base]


def reference_wav_bytes(ref: Path) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(ref),
                "-t", str(REF_MAX_SECONDS),
                "-ac", "1", "-ar", "24000",
                str(tmp_path),
            ],
            check=True,
            timeout=120,
        )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_moss(ref: Path, text: str, out: Path, api: str) -> tuple[bool, str]:
    cmd = [
        "curl", "-s", "-X", "POST", api,
        "-F", f"reference=@{ref}",
        "-F", f"text={text}",
        "-F", "language=en",
        "-o", str(out),
        "-w", "%{http_code}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    code = proc.stdout.strip()[-3:] if proc.stdout.strip() else "000"
    if code == "200" and out.exists() and out.stat().st_size > 1000:
        return True, f"HTTP {code}"
    detail = out.read_text(errors="ignore")[:120] if out.exists() else proc.stderr[:120]
    if out.exists():
        out.unlink(missing_ok=True)
    return False, f"HTTP {code} {detail}"


def port_from_api(api: str) -> int:
    from urllib.parse import urlparse

    parsed = urlparse(api)
    return parsed.port or 8014


def health_url_from_api(api: str) -> str:
    return api.rsplit("/", 1)[0] + "/health"


def kill_server_on_port(port: int) -> None:
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        capture_output=True,
        timeout=10,
        check=False,
    )
    time.sleep(1)


def openmoss_health_ok(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=2)
        body = resp.text.strip()
        if body == "ok":
            return True
        if resp.ok and "healthy" in body.lower():
            return True
    except requests.RequestException:
        pass
    return False


def ensure_openmoss_server(api: str | None = None, force_restart: bool = False) -> None:
    port = port_from_api(api) if api else int(os.environ.get("OPENMOSS_PORT", "8014"))
    health_url = health_url_from_api(api) if api else f"http://127.0.0.1:{port}/health"
    if not force_restart and openmoss_health_ok(health_url):
        return

    kill_server_on_port(port)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMA_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    env["OPENMOSS_MAIN_GPU"] = str(OPENMOSS_GPU)
    env["OPENMOSS_PORT"] = str(port)
    env.setdefault("OPENMOSS_MODEL_VERSION", "v15")
    # Aux codec/embeds need ~5 GiB VRAM; default to CPU when GPU0 is often full.
    env.setdefault("OPENMOSS_AUX_CPU", "1")
    log_path = Path(f"/tmp/openmoss-server-{port}.log")
    with log_path.open("a") as log:
        log.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"model={env.get('OPENMOSS_MODEL_VERSION')} gpu={OPENMOSS_GPU} port={port} ---\n")
    root = Path(os.environ.get("SPEAKER_ROOT", "/home/alec/git/speaker"))
    start_script = root / "training/moss-realtime/scripts/legacy/start-openmoss.sh"
    env["SPEAKER_ROOT"] = str(root)
    with log_path.open("ab") as log:
        subprocess.Popen(
            [str(start_script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(root),
        )
    for _ in range(120):
        if openmoss_health_ok(health_url):
            return
        time.sleep(2)
    tail = log_path.read_text(errors="ignore")[-800:] if log_path.is_file() else ""
    raise RuntimeError(f"openmoss not ready at {health_url}\n{tail}")


def _merge_sampling(overrides: dict | None) -> dict:
    out = dict(OPENMOSS_SAMPLING)
    if overrides:
        out.update(overrides)
    return out


def generate_openmoss(
    ref: Path,
    text: str,
    out: Path,
    api: str,
    tokens: int,
    max_new: int,
    instruction: str | None = None,
    sampling: dict | None = None,
) -> tuple[bool, str]:
    """Synthesize via openmoss JSON /tts API (raw server, not Speaker shim)."""
    payload: dict = {
        "text": text,
        "language": "en",
        "max_new_tokens": max(256, int(max_new)),
        "sampling": _merge_sampling(sampling),
        "reference_wav_b64": reference_wav_b64(ref),
    }
    if os.environ.get("OPENMOSS_NO_TOKENS", "").lower() not in ("1", "true", "yes"):
        payload["tokens"] = int(tokens)
    if instruction:
        payload["instruction"] = instruction

    container = docker_container_for_api(api)
    if container:
        return generate_openmoss_docker(container, payload, out)

    try:
        session = http_session_for_api(api)
        resp = session.post(api, json=payload, timeout=600)
    except requests.RequestException as exc:
        return False, str(exc)
    if resp.status_code != 200:
        detail = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
        return False, detail
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    if out.stat().st_size < 1024:
        out.unlink(missing_ok=True)
        return False, "output too small"
    return True, f"HTTP {resp.status_code}"


def generate_openmoss_cli(
    ref: Path,
    text: str,
    out: Path,
    tokens: int,
    max_new: int,
    instruction: str | None = None,
) -> tuple[bool, str]:
    """Synthesize via moss-tts-cli (single-GPU fallback)."""
    preload_reference(ref)
    if not OPENMOSS_CLI.is_file():
        return False, f"missing CLI: {OPENMOSS_CLI}"
    cmd = [
        str(OPENMOSS_CLI),
        "--model", str(OPENMOSS_MODEL),
        "--text", text,
        "--language", "en",
        "--tokens", str(tokens),
        "--max-new-tokens", str(max_new),
        "--output", str(out),
    ]
    ref_path = _prepared_ref_wav
    if ref_path is not None and ref_path.is_file():
        cmd.extend(["--reference", str(ref_path)])
    if instruction:
        cmd.extend(["--instruction", instruction])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LLAMA_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(OPENMOSS_GPU)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "cli failed")[:200]
    if out.is_file() and out.stat().st_size > 1024:
        return True, "cli ok"
    return False, "cli produced no wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch MOSS voice-clone samples")
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8014/tts")
    parser.add_argument("--out-dir", type=Path, default=Path("moss_samples"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_openmoss_server(args.api)
    preload_reference(args.ref)
    for i, text in enumerate(TEXTS):
        out = args.out_dir / f"sample_{i:02d}.wav"
        tokens, max_new = openmoss_token_budget(text, args.ref)
        ok, detail = generate_openmoss(args.ref, text, out, args.api, tokens, max_new)
        print(f"{out.name}: {ok} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())