"""
WebSocket endpoint for real-time audiobook generation.
Streams per-segment progress to connected clients with cancel support.
Also relays visual generation events from the background worker.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Set

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audiobook_models import (
    AudiobookProject, SegmentStatus,
    load_project, save_project, get_project_dir,
)
from .generation_queue import add_visual_listener, remove_visual_listener

logger = logging.getLogger("speaker.audiobook.ws")

router = APIRouter(tags=["Audiobook WebSocket"])

# Will be set by main.py on startup
tts_service = None

# Track active generation tasks per project
_active_generations: dict[str, asyncio.Task] = {}
_cancel_flags: dict[str, asyncio.Event] = {}


def set_tts_service(service):
    """Called from main.py to inject the TTS service."""
    global tts_service
    tts_service = service


# ============================================================
# Message helpers
# ============================================================

async def _send(ws: WebSocket, msg_type: str, **data):
    """Send a typed JSON message to the WebSocket client."""
    try:
        await ws.send_json({"type": msg_type, **data})
    except Exception:
        pass  # Client may have disconnected


# ============================================================
# Generation logic
# ============================================================

async def _generate_segment_async(text: str, voice: str, language: str = "en"):
    """Run synchronous TTS generation in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(
        tts_service.generate_speech,
        text=text,
        voice_name=voice,
        language=language,
    )


async def _run_generation(
    ws: WebSocket,
    project: AudiobookProject,
    segments_to_generate: list,
    cancel_event: asyncio.Event,
):
    """
    Generate audio for a list of segments, streaming progress via WebSocket.
    
    segments_to_generate: list of (chapter_idx, segment_idx, segment) tuples
    """
    total = len(segments_to_generate)
    generated = 0
    errors = 0
    skipped = 0

    await _send(ws, "generation_started", total=total)

    for i, (ch_idx, seg_idx, seg) in enumerate(segments_to_generate):
        # Check cancel flag
        if cancel_event.is_set():
            await _send(ws, "cancelled", generated=generated, errors=errors, remaining=total - i)
            save_project(project)
            return

        # Skip already done
        if seg.status == SegmentStatus.DONE:
            skipped += 1
            continue

        voice = seg.voice_name or project.narrator_voice
        if not voice:
            seg.status = SegmentStatus.ERROR
            seg.error_message = "No voice assigned"
            errors += 1
            await _send(ws, "segment_error", segment_id=seg.id, error="No voice assigned",
                        progress={"done": generated, "errors": errors, "total": total})
            continue

        # Notify start
        seg.status = SegmentStatus.GENERATING
        await _send(ws, "segment_start", segment_id=seg.id, index=i,
                     text_preview=seg.text[:80], voice=voice, total=total)

        try:
            audio, sample_rate = await _generate_segment_async(seg.text, voice)

            if audio is None or len(audio) == 0:
                raise RuntimeError("TTS returned empty audio")

            # Save audio
            audio_dir = os.path.join(get_project_dir(project.id), "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{seg.id}.wav")
            sf.write(audio_path, audio, sample_rate, format="WAV", subtype="PCM_16")

            seg.status = SegmentStatus.DONE
            seg.audio_path = audio_path
            seg.duration = len(audio) / sample_rate
            seg.error_message = None
            generated += 1

            await _send(ws, "segment_done",
                        segment_id=seg.id,
                        duration=round(seg.duration, 2),
                        progress={"done": generated, "errors": errors, "total": total})

            # Save after each segment so progress persists
            save_project(project)

        except Exception as e:
            seg.status = SegmentStatus.ERROR
            seg.error_message = str(e)
            errors += 1
            logger.error(f"Failed segment '{seg.id}': {e}")

            await _send(ws, "segment_error",
                        segment_id=seg.id,
                        error=str(e),
                        progress={"done": generated, "errors": errors, "total": total})

            save_project(project)

    # Done
    save_project(project)
    await _send(ws, "complete", generated=generated, errors=errors, skipped=skipped, total=total)


def _collect_segments(project: AudiobookProject, chapter_idx: Optional[int] = None):
    """Collect (ch_idx, seg_idx, segment) tuples for generation."""
    segments = []
    for ch_idx_i, ch in enumerate(project.chapters):
        if chapter_idx is not None and ch_idx_i != chapter_idx:
            continue
        for seg_idx, seg in enumerate(ch.segments):
            if seg.status != SegmentStatus.DONE:
                segments.append((ch_idx_i, seg_idx, seg))
    return segments


# ============================================================
# WebSocket endpoint
# ============================================================

@router.websocket("/audiobook/ws/{project_id}")
async def audiobook_generation_ws(ws: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time audiobook generation.
    
    Inbound messages:
        {"action": "generate_all"}
        {"action": "generate_chapter", "chapter_idx": 0}
        {"action": "cancel"}
    
    Outbound messages (TTS):
        {"type": "connected", "project_id": "..."}
        {"type": "generation_started", "total": 47}
        {"type": "segment_start", "segment_id": "abc", ...}
        {"type": "segment_done", "segment_id": "abc", "duration": 3.2, "progress": {...}}
        {"type": "segment_error", "segment_id": "abc", "error": "...", "progress": {...}}
        {"type": "complete", "generated": 45, "errors": 2, ...}
        {"type": "cancelled", ...}
        {"type": "error", "message": "..."}
    
    Outbound messages (Visual — relayed from generation queue event bus):
        {"type": "visual_start", "segment_id": "...", "mode": "..."}
        {"type": "visual_prompt_generating", "segment_id": "..."}
        {"type": "visual_prompt_done", "segment_id": "...", "prompt": "..."}
        {"type": "visual_rendering", "segment_id": "...", "mode": "..."}
        {"type": "visual_progress", "segment_id": "...", "step": 5, "total_steps": 20}
        {"type": "visual_done", "segment_id": "...", "visual_type": "...", "visual_path": "..."}
        {"type": "visual_error", "segment_id": "...", "error": "..."}
    """
    await ws.accept()

    # Verify project exists
    project = load_project(project_id)
    if not project:
        await _send(ws, "error", message=f"Project '{project_id}' not found")
        await ws.close()
        return

    await _send(ws, "connected", project_id=project_id, project_name=project.name)

    # Cancel event for this project
    cancel_event = asyncio.Event()
    _cancel_flags[project_id] = cancel_event

    # --- Visual event relay ---
    # Register a listener that forwards visual events for this project
    async def _on_visual_event(event: dict):
        if event.get("project_id") == project_id:
            # Forward to client (strip project_id, already known)
            msg = {k: v for k, v in event.items() if k != "project_id"}
            try:
                await ws.send_json(msg)
            except Exception:
                pass

    add_visual_listener(_on_visual_event)

    try:
        while True:
            # Wait for client messages
            try:
                raw = await ws.receive_text()
                msg = json.loads(raw)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await _send(ws, "error", message="Invalid JSON")
                continue

            action = msg.get("action", "")

            if action == "cancel":
                cancel_event.set()
                # If there's an active task, it will check the flag
                if project_id in _active_generations:
                    task = _active_generations[project_id]
                    if not task.done():
                        await _send(ws, "cancel_acknowledged")
                        # Wait briefly for the task to notice
                        try:
                            await asyncio.wait_for(task, timeout=10)
                        except asyncio.TimeoutError:
                            task.cancel()
                else:
                    await _send(ws, "error", message="No active generation to cancel")

            elif action in ("generate_all", "generate_chapter"):
                if not tts_service:
                    await _send(ws, "error", message="TTS service not available")
                    continue

                # Cancel any existing generation for this project
                if project_id in _active_generations:
                    old_task = _active_generations[project_id]
                    if not old_task.done():
                        cancel_event.set()
                        try:
                            await asyncio.wait_for(old_task, timeout=10)
                        except asyncio.TimeoutError:
                            old_task.cancel()

                # Fresh cancel event
                cancel_event = asyncio.Event()
                _cancel_flags[project_id] = cancel_event

                # Reload project to get latest state
                project = load_project(project_id)
                if not project:
                    await _send(ws, "error", message="Project not found")
                    continue

                # Collect segments
                chapter_idx = msg.get("chapter_idx") if action == "generate_chapter" else None
                segments = _collect_segments(project, chapter_idx)

                if not segments:
                    await _send(ws, "complete", generated=0, errors=0, skipped=0, total=0,
                                message="No pending segments")
                    continue

                # Validate voices
                available_voices = tts_service.get_voices()
                for _, _, seg in segments:
                    voice = seg.voice_name or project.narrator_voice
                    if voice and voice not in available_voices:
                        await _send(ws, "error",
                                    message=f"Voice '{voice}' not found. Available: {', '.join(available_voices)}")
                        continue

                # Start generation as background task
                gen_task = asyncio.create_task(
                    _run_generation(ws, project, segments, cancel_event)
                )
                _active_generations[project_id] = gen_task

            elif action == "ping":
                await _send(ws, "pong")

            else:
                await _send(ws, "error", message=f"Unknown action: {action}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project '{project_id}'")
    except Exception as e:
        logger.error(f"WebSocket error for project '{project_id}': {e}")
    finally:
        # Cleanup: unregister visual listener
        remove_visual_listener(_on_visual_event)
        cancel_event.set()
        if project_id in _active_generations:
            task = _active_generations.pop(project_id, None)
            if task and not task.done():
                task.cancel()
        _cancel_flags.pop(project_id, None)

