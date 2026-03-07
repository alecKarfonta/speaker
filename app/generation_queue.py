"""
generation_queue.py — Three asyncio queues for sequential job processing.

Queues:
  tts_queue    – audio segment generation (blocking GPU/CPU)
  prompt_queue – scene prompt generation via LLM
  visual_queue – ComfyUI visual generation (blocking GPU)

Each queue has exactly one worker coroutine, guaranteeing that jobs within
each queue are processed one at a time and never collide.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("speaker.queues")


# ---------------------------------------------------------------------------
# Job dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TTSJob:
    project_id: str
    segment_id: str


@dataclass
class PromptJob:
    project_id: str
    segment_id: str


@dataclass
class VisualJob:
    project_id: str
    segment_id: str
    mode: Optional[str] = None
    frames: Optional[int] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    animation: Optional[str] = None
    ref_character: Optional[str] = None  # explicit portrait override
    enable_audio: bool = False
    two_stage: bool = False


# ---------------------------------------------------------------------------
# Queue state
# ---------------------------------------------------------------------------

tts_queue: asyncio.Queue = asyncio.Queue()
prompt_queue: asyncio.Queue = asyncio.Queue()
visual_queue: asyncio.Queue = asyncio.Queue()

# Currently-processing job per queue (for status display)
_active: Dict[str, Optional[dict]] = {
    "tts": None,
    "prompt": None,
    "visual": None,
}

# History of recent completed jobs (capped at 50)
_history: List[dict] = []

_workers_started = False


# ---------------------------------------------------------------------------
# Worker implementations
# ---------------------------------------------------------------------------

async def _tts_worker():
    """Process audio generation jobs one at a time."""
    from .audiobook_models import load_project, save_project
    from .audiobook_models import SegmentStatus

    logger.info("TTS worker started")
    while True:
        job: TTSJob = await tts_queue.get()
        _active["tts"] = {"project_id": job.project_id, "segment_id": job.segment_id, "type": "tts"}
        try:
            from . import audiobook_router as _router
            tts = _router.tts_service
            if not tts:
                logger.error("TTS worker: no TTS service available")
                tts_queue.task_done()
                _active["tts"] = None
                continue

            project = load_project(job.project_id)
            if not project:
                tts_queue.task_done()
                _active["tts"] = None
                continue

            # Find segment
            segment = None
            for ch in project.chapters:
                for seg in ch.segments:
                    if seg.id == job.segment_id:
                        segment = seg
                        break

            if not segment:
                tts_queue.task_done()
                _active["tts"] = None
                continue

            voice = segment.voice_name or project.narrator_voice
            if not voice:
                logger.warning(f"TTS worker: no voice for segment {job.segment_id}")
                tts_queue.task_done()
                _active["tts"] = None
                continue

            segment.status = SegmentStatus.GENERATING
            save_project(project)

            import os
            from .audiobook_models import get_project_dir
            audio_dir = os.path.join(get_project_dir(job.project_id), "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{job.segment_id}.wav")

            try:
                import soundfile as sf
                audio_data, sample_rate = await asyncio.to_thread(
                    tts.generate_speech, segment.text, voice
                )
                sf.write(audio_path, audio_data, sample_rate, format="WAV", subtype="PCM_16")

                duration = len(audio_data) / sample_rate
                segment.audio_path = audio_path
                segment.duration = duration
                segment.status = SegmentStatus.DONE
                logger.info(f"TTS worker: generated audio for segment {job.segment_id} ({duration:.1f}s)")
                _history.append({"type": "tts", "project_id": job.project_id, "segment_id": job.segment_id, "status": "done"})
            except Exception as e:
                segment.status = SegmentStatus.ERROR
                segment.error_message = str(e)
                logger.error(f"TTS worker: failed {job.segment_id}: {e}")
                _history.append({"type": "tts", "project_id": job.project_id, "segment_id": job.segment_id, "status": "error", "error": str(e)})

            save_project(project)
        except Exception as e:
            logger.error(f"TTS worker outer error: {e}")
        finally:
            tts_queue.task_done()
            _active["tts"] = None
            if len(_history) > 50:
                _history.pop(0)


async def _prompt_worker():
    """Process scene prompt generation jobs one at a time."""
    from .audiobook_models import load_project, save_project

    logger.info("Prompt worker started")
    while True:
        job: PromptJob = await prompt_queue.get()
        _active["prompt"] = {"project_id": job.project_id, "segment_id": job.segment_id, "type": "prompt"}
        try:
            from .audiobook_llm import generate_scene_prompt, generate_visual_style

            project = load_project(job.project_id)
            if not project:
                prompt_queue.task_done()
                _active["prompt"] = None
                continue

            # Auto-generate visual style if missing
            if not project.visual_style and project.raw_text:
                style = generate_visual_style(project.raw_text)
                if style:
                    project.visual_style = style
                    save_project(project)

            all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]
            segment, chapter, seg_idx = None, None, None
            for i, (ch, seg) in enumerate(all_segments):
                if seg.id == job.segment_id:
                    segment, chapter, seg_idx = seg, ch, i
                    break

            if not segment:
                prompt_queue.task_done()
                _active["prompt"] = None
                continue

            prev_scene = all_segments[seg_idx - 1][1].scene_prompt if seg_idx > 0 else ""
            next_text = all_segments[seg_idx + 1][1].text if seg_idx < len(all_segments) - 1 else ""

            scene = await asyncio.to_thread(
                generate_scene_prompt,
                text=segment.text,
                chapter_title=chapter.title if chapter else "",
                character=segment.character or "Narrator",
                emotion=segment.emotion or "neutral",
                visual_style=project.visual_style,
                prev_scene=prev_scene or "",
                next_text=next_text or "",
            )
            if scene:
                segment.scene_prompt = scene
                if segment.visual_status == "done":
                    segment.visual_status = "pending"
                save_project(project)
                logger.info(f"Prompt worker: generated scene prompt for {job.segment_id}")
                _history.append({"type": "prompt", "project_id": job.project_id, "segment_id": job.segment_id, "status": "done"})
            else:
                logger.error(f"Prompt worker: LLM returned empty for {job.segment_id}")
                _history.append({"type": "prompt", "project_id": job.project_id, "segment_id": job.segment_id, "status": "error"})
        except Exception as e:
            logger.error(f"Prompt worker error: {e}")
        finally:
            prompt_queue.task_done()
            _active["prompt"] = None
            if len(_history) > 50:
                _history.pop(0)


async def _visual_worker():
    """Process visual generation jobs one at a time."""
    from .audiobook_models import load_project, save_project, get_project_dir

    logger.info("Visual worker started")
    while True:
        job: VisualJob = await visual_queue.get()
        _active["visual"] = {"project_id": job.project_id, "segment_id": job.segment_id, "type": "visual"}
        try:
            from .comfyui_service import generate_visual as comfyui_generate, COMFYUI_VISUAL_MODE
            from .audiobook_llm import generate_scene_prompt, generate_visual_style
            from .audiobook_router import _find_character_portrait, _find_character_portrait_by_name

            project = load_project(job.project_id)
            if not project:
                visual_queue.task_done()
                _active["visual"] = None
                continue

            # Auto-generate visual style if missing
            if not project.visual_style and project.raw_text:
                style = generate_visual_style(project.raw_text)
                if style:
                    project.visual_style = style
                    save_project(project)

            all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]
            segment, chapter, seg_idx = None, None, None
            for i, (ch, seg) in enumerate(all_segments):
                if seg.id == job.segment_id:
                    segment, chapter, seg_idx = seg, ch, i
                    break

            if not segment:
                visual_queue.task_done()
                _active["visual"] = None
                continue

            # Save animation style
            if job.animation is not None:
                segment.animation_style = job.animation
                save_project(project)

            # Generate scene prompt if missing
            if not segment.scene_prompt:
                prev_scene = all_segments[seg_idx - 1][1].scene_prompt if seg_idx > 0 else ""
                next_text = all_segments[seg_idx + 1][1].text if seg_idx < len(all_segments) - 1 else ""
                scene = await asyncio.to_thread(
                    generate_scene_prompt,
                    text=segment.text,
                    chapter_title=chapter.title if chapter else "",
                    character=segment.character or "Narrator",
                    emotion=segment.emotion or "neutral",
                    visual_style=project.visual_style,
                    prev_scene=prev_scene or "",
                    next_text=next_text or "",
                )
                if not scene:
                    segment.visual_status = "error"
                    save_project(project)
                    visual_queue.task_done()
                    _active["visual"] = None
                    continue
                segment.scene_prompt = scene
                save_project(project)

            # Mark as generating
            segment.visual_status = "generating"
            save_project(project)

            visual_dir = os.path.join(get_project_dir(job.project_id), "visuals")
            os.makedirs(visual_dir, exist_ok=True)

            # Explicit portrait override from user selection, else auto-detect
            if job.ref_character:
                ref_image = _find_character_portrait_by_name(project, job.ref_character)
            else:
                ref_image = _find_character_portrait(project, segment)

            try:
                path, vtype = await comfyui_generate(
                    prompt=segment.scene_prompt,
                    output_dir=visual_dir,
                    prefix=f"seg_{segment.id}",
                    mode=job.mode or COMFYUI_VISUAL_MODE,
                    duration=segment.duration,
                    ref_image=ref_image,
                    frames=job.frames,
                    fps=job.fps,
                    width=job.width,
                    height=job.height,
                    enable_audio=job.enable_audio,
                    two_stage=job.two_stage,
                )
                segment.visual_path = path
                segment.visual_type = vtype
                segment.visual_status = "done"
                logger.info(f"Visual worker: generated {vtype} for segment {job.segment_id}")
                _history.append({"type": "visual", "project_id": job.project_id, "segment_id": job.segment_id, "status": "done"})
            except Exception as e:
                segment.visual_status = "error"
                logger.error(f"Visual worker: failed {job.segment_id}: {e}")
                _history.append({"type": "visual", "project_id": job.project_id, "segment_id": job.segment_id, "status": "error", "error": str(e)})

            save_project(project)
        except Exception as e:
            logger.error(f"Visual worker outer error: {e}")
        finally:
            visual_queue.task_done()
            _active["visual"] = None
            if len(_history) > 50:
                _history.pop(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_tts(project_id: str, segment_id: str) -> int:
    """Add a TTS job to the queue. Returns queue depth after enqueue."""
    tts_queue.put_nowait(TTSJob(project_id=project_id, segment_id=segment_id))
    pos = tts_queue.qsize()
    logger.debug(f"Enqueued TTS job {segment_id} (depth: {pos})")
    return pos


def enqueue_prompt(project_id: str, segment_id: str) -> int:
    """Add a scene-prompt job to the queue. Returns queue depth after enqueue."""
    prompt_queue.put_nowait(PromptJob(project_id=project_id, segment_id=segment_id))
    pos = prompt_queue.qsize()
    logger.debug(f"Enqueued prompt job {segment_id} (depth: {pos})")
    return pos


def enqueue_visual(
    project_id: str,
    segment_id: str,
    mode: str = None,
    frames: int = None,
    fps: int = None,
    width: int = None,
    height: int = None,
    animation: str = None,
    ref_character: str = None,
    enable_audio: bool = False,
    two_stage: bool = False,
) -> int:
    """Add a visual generation job to the queue. Returns queue depth after enqueue."""
    visual_queue.put_nowait(VisualJob(
        project_id=project_id,
        segment_id=segment_id,
        mode=mode,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        animation=animation,
        ref_character=ref_character,
        enable_audio=enable_audio,
        two_stage=two_stage,
    ))
    pos = visual_queue.qsize()
    logger.debug(f"Enqueued visual job {segment_id} (depth: {pos})")
    return pos


def get_queue_status() -> dict:
    """Return snapshots of all three queue depths and active jobs."""
    return {
        "tts": {
            "queued": tts_queue.qsize(),
            "active": _active["tts"],
        },
        "prompt": {
            "queued": prompt_queue.qsize(),
            "active": _active["prompt"],
        },
        "visual": {
            "queued": visual_queue.qsize(),
            "active": _active["visual"],
        },
        "recent": _history[-10:],
    }


def start_workers(loop: asyncio.AbstractEventLoop = None):
    """Schedule the three worker coroutines. Safe to call multiple times."""
    global _workers_started
    if _workers_started:
        return
    _workers_started = True

    asyncio.ensure_future(_tts_worker())
    asyncio.ensure_future(_prompt_worker())
    asyncio.ensure_future(_visual_worker())
    logger.info("Generation queue workers started (TTS, Prompt, Visual)")
