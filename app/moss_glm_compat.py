"""
GLM-TTS API compatibility layer for MOSS-TTS.

Ports voice-management, language listing, and API-info endpoints from app/main.py
so clients can switch from GLM to MOSS without changing request/response shapes.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.models import (
    APIInfo,
    LanguageListResponse,
    VoiceDeleteResponse,
    VoiceUploadResponse,
)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

# MOSS-TTS-v1.5 supported languages (code -> display name)
MOSS_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian (Farsi)",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

QWEN_LANGUAGE_NAMES: list[str] = sorted(set(MOSS_LANGUAGES.values()) | {"Auto"})

OPENAI_VOICE_NAMES: frozenset[str] = frozenset(
    {
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer",
        "ash",
        "ballad",
        "coral",
        "sage",
    }
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _active_model_id() -> str:
    if os.environ.get("MOSS_ENABLE_REALTIME", "false").lower() == "true":
        return os.environ.get("MOSS_RT_MODEL_ID", "OpenMOSS-Team/MOSS-TTS-Realtime")
    return os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS-v1.5")


def resolve_openai_voice(
    voice: str,
    available_voices: list[str],
    *,
    native_voice: bool = False,
) -> Optional[str]:
    """Map OpenAI / Speaker voice names to a reference voice (or None for native)."""
    if voice in available_voices:
        return voice
    if voice.lower() in OPENAI_VOICE_NAMES:
        if available_voices:
            return available_voices[0]
        if native_voice:
            return None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No voices available",
        )
    if native_voice:
        return None
    preview = ", ".join(available_voices[:10])
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Voice '{voice}' not found. Available: {preview}",
    )


def convert_audio_format(
    audio_data: np.ndarray,
    sample_rate: int,
    target_format: str,
) -> tuple[bytes, str]:
    """Convert audio to the requested OpenAI-compatible format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    content_type = content_types.get(target_format, "audio/mpeg")

    if target_format == "wav":
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        return wav_buffer.read(), content_type

    if target_format == "flac":
        flac_buffer = io.BytesIO()
        sf.write(flac_buffer, audio_data, sample_rate, format="FLAC")
        flac_buffer.seek(0)
        return flac_buffer.read(), content_type

    if target_format == "pcm":
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes(), content_type

    wav_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            sf.write(wav_file.name, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            wav_path = wav_file.name

        output_path = wav_path.replace(".wav", f".{target_format}")
        if target_format == "mp3":
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path,
            ]
        elif target_format == "opus":
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libopus",
                "-b:a",
                "128k",
                output_path,
            ]
        elif target_format == "aac":
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "aac",
                "-b:a",
                "192k",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path.replace(f".{target_format}", ".mp3"),
            ]
            output_path = output_path.replace(f".{target_format}", ".mp3")
            content_type = "audio/mpeg"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            wav_buffer.seek(0)
            return wav_buffer.read(), "audio/wav"

        with open(output_path, "rb") as f:
            return f.read(), content_type
    finally:
        for path in (wav_path, output_path):
            if path and os.path.exists(path):
                os.unlink(path)


def _voices_dir() -> Path:
    return Path(os.environ.get("VOICES_DIR", "/app/data/voices"))


def _list_voice_names() -> list[str]:
    voices_dir = _voices_dir()
    if not voices_dir.exists():
        return []
    return sorted(
        d.name
        for d in voices_dir.iterdir()
        if d.is_dir()
        and any(f.suffix.lower() in AUDIO_EXTENSIONS for f in d.iterdir())
    )


def _validate_voice_name(voice_name: str) -> None:
    if not voice_name.replace("_", "").isalnum():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice name must contain only letters, numbers, and underscores",
        )


def _voice_dir(voice_name: str) -> Path:
    return _voices_dir() / voice_name


def _require_voice(voice_name: str) -> Path:
    voice_dir = _voice_dir(voice_name)
    if not voice_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{voice_name}' not found",
        )
    return voice_dir


def _resolve_language_name(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    code = language.lower().split("-")[0]
    if code in MOSS_LANGUAGES:
        return MOSS_LANGUAGES[code]
    # Accept full names directly (e.g. "French")
    for name in MOSS_LANGUAGES.values():
        if name.lower() == language.lower():
            return name
    return language


@router.get("/", response_model=APIInfo)
async def read_root():
    model_id = _active_model_id()
    return APIInfo(
        message="Welcome to the Speaker TTS API",
        version="1.0.0",
        model=model_id,
        status="operational",
        documentation="/docs",
        health_check="/health",
    )


@router.get("/languages", response_model=LanguageListResponse)
async def get_supported_languages():
    return LanguageListResponse(languages=MOSS_LANGUAGES, total_count=len(MOSS_LANGUAGES))


@router.get("/api/v1/qwen/languages")
async def list_qwen_languages():
    return {"languages": QWEN_LANGUAGE_NAMES, "total_count": len(QWEN_LANGUAGE_NAMES)}


@router.get("/voices/{voice_name}/details")
async def get_voice_details(voice_name: str):
    voice_dir = _require_voice(voice_name)

    files = []
    total_size = 0
    for filename in os.listdir(voice_dir):
        if filename.lower().endswith((".wav", ".mp3")):
            filepath = voice_dir / filename
            file_stat = filepath.stat()

            duration = None
            try:
                audio_data, sample_rate = sf.read(filepath)
                duration = len(audio_data) / sample_rate
            except Exception:
                pass

            files.append(
                {
                    "name": filename,
                    "path": str(filepath),
                    "size": file_stat.st_size,
                    "duration": duration,
                    "modified": file_stat.st_mtime,
                }
            )
            total_size += file_stat.st_size

    return {
        "voice_name": voice_name,
        "files": files,
        "total_files": len(files),
        "total_size": total_size,
    }


@router.get("/voices/{voice_name}/export")
async def export_voice(voice_name: str):
    voice_dir = _require_voice(voice_name)

    voice_files = []
    for filename in sorted(os.listdir(voice_dir)):
        filepath = voice_dir / filename
        if filepath.is_file():
            voice_files.append((filename, filepath))

    if not voice_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice '{voice_name}' has no files to export",
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, filepath in voice_files:
            zf.write(filepath, filename)

    zip_bytes = zip_buffer.getvalue()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={voice_name}.zip",
            "Content-Length": str(len(zip_bytes)),
        },
    )


@router.post("/voices/import")
async def import_voice(
    voice_name: str,
    file: UploadFile = File(...),
):
    _validate_voice_name(voice_name)

    voice_dir = _voice_dir(voice_name)
    if voice_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Voice '{voice_name}' already exists. "
                "Delete it first or choose a different name."
            ),
        )

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are supported for voice import",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        zip_buffer = io.BytesIO(content)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in member:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid path in ZIP: {member}",
                    )

            audio_files = [
                m
                for m in zf.namelist()
                if not m.endswith("/")
                and os.path.splitext(m)[1].lower() in (".wav", ".mp3")
            ]
            if not audio_files:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="ZIP contains no .wav or .mp3 files",
                )

            voice_dir.mkdir(parents=True, exist_ok=True)
            imported_files = []
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                basename = os.path.basename(member)
                if not basename:
                    continue
                target_path = voice_dir / basename
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                imported_files.append(basename)

        return {
            "message": f"Voice '{voice_name}' imported successfully",
            "voice_name": voice_name,
            "files_imported": len(imported_files),
        }
    except HTTPException:
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise
    except Exception as e:
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import voice: {e}",
        ) from e


@router.get("/voices/{voice_name}/files/{filename}")
async def download_voice_file(voice_name: str, filename: str):
    voice_dir = _require_voice(voice_name)
    filepath = voice_dir / filename

    if not os.path.abspath(filepath).startswith(os.path.abspath(voice_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in voice '{voice_name}'",
        )

    return FileResponse(filepath, media_type="application/octet-stream", filename=filename)


@router.delete("/voices/{voice_name}/files/{filename}")
async def delete_voice_file(voice_name: str, filename: str):
    voice_dir = _require_voice(voice_name)
    filepath = voice_dir / filename

    if not os.path.abspath(filepath).startswith(os.path.abspath(voice_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in voice '{voice_name}'",
        )

    os.remove(filepath)

    remaining_files = [
        f
        for f in os.listdir(voice_dir)
        if f.lower().endswith((".wav", ".mp3"))
    ]
    if not remaining_files:
        shutil.rmtree(voice_dir)

    return {"message": f"File '{filename}' deleted successfully"}


@router.post("/voices/combine")
async def combine_voice_files(request: Request):
    body = await request.json()
    voice_name = body.get("voice_name")
    source_voices = body.get("source_voices") or []

    if not voice_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="voice_name is required",
        )
    _validate_voice_name(voice_name)

    if len(source_voices) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 source voices are required to combine",
        )

    target_dir = _voice_dir(voice_name)
    if target_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice '{voice_name}' already exists",
        )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_index = 1
        copied_files = []

        for source_voice in source_voices:
            source_dir = _voice_dir(source_voice)
            if not source_dir.exists():
                continue

            for filename in os.listdir(source_dir):
                if filename.lower().endswith((".wav", ".mp3")):
                    source_path = source_dir / filename
                    file_extension = os.path.splitext(filename)[1]
                    target_filename = f"{voice_name}_{str(file_index).zfill(2)}{file_extension}"
                    target_path = target_dir / target_filename
                    shutil.copy2(source_path, target_path)
                    copied_files.append(target_filename)
                    file_index += 1

        if not copied_files:
            shutil.rmtree(target_dir)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid audio files found in source voices",
            )

        return {
            "message": f"Successfully combined {len(source_voices)} voices into '{voice_name}'",
            "voice_name": voice_name,
            "files_copied": len(copied_files),
            "source_voices": source_voices,
        }
    except HTTPException:
        raise
    except Exception as e:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to combine voices: {e}",
        ) from e


@router.delete("/voices/{voice_name}", response_model=VoiceDeleteResponse)
async def delete_voice(voice_name: str):
    voice_dir = _require_voice(voice_name)
    shutil.rmtree(voice_dir)
    return VoiceDeleteResponse(
        message=f"Voice '{voice_name}' deleted successfully",
        deleted_voice=voice_name,
    )


def save_voice_upload(
    voice_name: str,
    filename: str,
    content: bytes,
    transcription: Optional[str] = None,
) -> VoiceUploadResponse:
    """GLM-compatible voice upload with indexed filenames."""
    _validate_voice_name(voice_name)

    if not filename or not (
        filename.lower().endswith(".wav") or filename.lower().endswith(".mp3")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav and .mp3 files are supported",
        )

    voice_dir = _voice_dir(voice_name)
    voice_dir.mkdir(parents=True, exist_ok=True)

    file_extension = os.path.splitext(filename)[1]
    voice_index = 1
    voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"
    while (voice_dir / voice_file_name).exists():
        voice_index += 1
        voice_file_name = f"{voice_name}_{str(voice_index).zfill(2)}{file_extension}"

    local_save_path = voice_dir / voice_file_name
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    local_save_path.write_bytes(content)

    if transcription and transcription.strip():
        transcription_path = local_save_path.with_suffix(".txt")
        transcription_path.write_text(transcription.strip(), encoding="utf-8")

    return VoiceUploadResponse(
        message=f"Voice '{voice_name}' uploaded successfully",
        voice_name=voice_name,
        file_name=voice_file_name,
        file_size=len(content),
    )
