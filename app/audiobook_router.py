from pydantic import BaseModel
"""
Audiobook API router — all /audiobook/* endpoints.
Handles project CRUD, segment generation, and audio export.
"""

import asyncio
import io
import os
import re
import logging
import numpy as np
import soundfile as sf
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, UploadFile, status, Response

# Per-project export locks — prevents concurrent exports from corrupting clips
_export_locks: Dict[str, asyncio.Lock] = {}
from datetime import datetime

from .audiobook_models import (
    AudiobookProject, Chapter, Segment, SegmentStatus, VisualAsset,
    CreateProjectRequest, UpdateCharacterMapRequest, UpdateSegmentRequest,
    SplitSegmentRequest, ReparseRequest, ProjectSummary, ProjectDetailResponse,
    VisualAssetResponse,
    save_project, load_project, list_projects, delete_project_from_disk,
    project_to_detail_response, get_project_dir, _resolve_visual,
)
from .audiobook_parser import (
    parse_book_text, detect_characters, assign_segment_voices,
)
from .audiobook_llm import analyze_characters as llm_analyze_characters, check_llm_available, generate_narrator_voice_prompt as llm_generate_narrator_voice_prompt
from .audiobook_parsers import parse_uploaded_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger("speaker.audiobook")

router = APIRouter(prefix="/audiobook", tags=["Audiobook"])

# Will be set by main.py on startup
tts_service = None


def set_tts_service(service):
    """Called from main.py to inject the TTS service."""
    global tts_service
    tts_service = service


@router.on_event("startup")
async def _start_generation_workers():
    """Start background queue workers when FastAPI starts."""
    from .generation_queue import start_workers
    start_workers()
    logger.info("Generation queue workers registered at startup")


def _get_project_or_404(project_id: str) -> AudiobookProject:
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return project


def _find_segment(project: AudiobookProject, segment_id: str):
    """Find a segment by ID across all chapters. Returns (chapter_index, segment_index, segment)."""
    for ch_idx, ch in enumerate(project.chapters):
        for seg_idx, seg in enumerate(ch.segments):
            if seg.id == segment_id:
                return ch_idx, seg_idx, seg
    raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found")


# --- Project CRUD ---

@router.get("/projects", response_model=List[ProjectSummary])
async def list_audiobook_projects():
    """List all audiobook projects."""
    return list_projects()


@router.post("/projects", response_model=ProjectDetailResponse, status_code=201)
async def create_audiobook_project(request: CreateProjectRequest):
    """Create a new audiobook project from raw text."""
    # Parse text into chapters
    chapters = parse_book_text(request.text, request.chapter_pattern)
    
    # Detect characters
    characters = detect_characters(request.text)
    
    # Get available voices for default narrator
    narrator = request.narrator_voice or ""
    if not narrator and tts_service:
        voices = tts_service.get_voices()
        if voices:
            narrator = voices[0]
    
    # Assign narrator voice to all segments initially
    for ch in chapters:
        for seg in ch.segments:
            seg.voice_name = narrator
    
    project = AudiobookProject(
        name=request.name,
        raw_text=request.text,
        chapter_pattern=request.chapter_pattern,
        chapters=chapters,
        narrator_voice=narrator,
        detected_characters=characters,
    )
    
    save_project(project)
    logger.info(f"Created audiobook project '{project.name}' with {len(chapters)} chapters, {project.total_segments} segments")
    return project_to_detail_response(project)


@router.post("/projects/import", response_model=ProjectDetailResponse, status_code=201)
async def import_audiobook_project(
    file: UploadFile,
    narrator_voice: str = "",
):
    """
    Import a book from an uploaded file (EPUB, PDF, or TXT).
    Extracts text and chapter structure automatically.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        parsed = parse_uploaded_file(file.filename, file_bytes)
    except Exception as e:
        logger.error(f"Failed to parse '{file.filename}': {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")

    if not parsed.chapters:
        raise HTTPException(status_code=422, detail="No text content found in file")

    # Convert parsed chapters to our Chapter model with segments
    chapters = []
    for i, pch in enumerate(parsed.chapters):
        from .audiobook_parser import split_chapter_into_segments
        segments = split_chapter_into_segments(pch.text)
        ch = Chapter(index=i, title=pch.title)
        ch.segments = segments
        chapters.append(ch)

    # Detect characters from the full text
    characters = detect_characters(parsed.raw_text)

    # Get narrator voice
    narrator = narrator_voice
    if not narrator and tts_service:
        voices = tts_service.get_voices()
        if voices:
            narrator = voices[0]

    # Assign narrator to all segments
    for ch in chapters:
        for seg in ch.segments:
            seg.voice_name = narrator

    project = AudiobookProject(
        name=parsed.title,
        raw_text=parsed.raw_text,
        chapter_pattern="imported",
        chapters=chapters,
        narrator_voice=narrator,
        detected_characters=characters,
    )

    save_project(project)
    total_segs = sum(len(ch.segments) for ch in chapters)
    logger.info(
        f"Imported '{parsed.title}' from {file.filename}: "
        f"{len(chapters)} chapters, {total_segs} segments, {len(parsed.raw_text)} chars"
    )
    return project_to_detail_response(project)


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_audiobook_project(project_id: str):
    """Get full project details."""
    project = _get_project_or_404(project_id)
    return project_to_detail_response(project)


@router.delete("/projects/{project_id}")
async def delete_audiobook_project(project_id: str):
    """Delete a project and all its audio files."""
    _get_project_or_404(project_id)  # Verify exists
    delete_project_from_disk(project_id)
    logger.info(f"Deleted audiobook project '{project_id}'")
    return {"message": f"Project '{project_id}' deleted"}


# --- Re-parse ---

@router.post("/projects/{project_id}/parse", response_model=ProjectDetailResponse)
async def reparse_project(project_id: str, request: ReparseRequest):
    """Re-parse the project text with a new chapter pattern."""
    project = _get_project_or_404(project_id)
    
    chapters = parse_book_text(project.raw_text, request.chapter_pattern)
    
    # Re-assign voices
    for ch in chapters:
        assign_segment_voices(ch.segments, project.character_voice_map, project.narrator_voice)
    
    project.chapters = chapters
    project.chapter_pattern = request.chapter_pattern
    project.detected_characters = detect_characters(project.raw_text)
    
    save_project(project)
    logger.info(f"Re-parsed project '{project_id}' with pattern '{request.chapter_pattern}'")
    return project_to_detail_response(project)


# --- Character mapping ---

@router.get("/projects/{project_id}/characters")
async def get_characters(project_id: str):
    """Get detected characters and their voice assignments."""
    project = _get_project_or_404(project_id)
    return {
        "detected_characters": project.detected_characters,
        "character_voice_map": project.character_voice_map,
        "character_descriptions": project.character_descriptions,
        "narrator_voice": project.narrator_voice,
    }


@router.put("/projects/{project_id}/characters", response_model=ProjectDetailResponse)
async def update_character_map(project_id: str, request: UpdateCharacterMapRequest):
    """Update character-to-voice mapping and re-assign segment voices."""
    project = _get_project_or_404(project_id)
    
    project.character_voice_map = request.character_voice_map
    if request.narrator_voice is not None:
        project.narrator_voice = request.narrator_voice
    
    # Re-assign voices across all segments
    for ch in project.chapters:
        assign_segment_voices(ch.segments, project.character_voice_map, project.narrator_voice)
    
    save_project(project)
    logger.info(f"Updated character map for project '{project_id}'")
    return project_to_detail_response(project)


# --- AI Character Analysis ---

@router.get("/ai-status")
async def ai_status():
    """Check if the LLM endpoint is reachable for AI analysis."""
    available = check_llm_available()
    return {"available": available}


@router.post("/projects/{project_id}/analyze", response_model=ProjectDetailResponse)
async def analyze_project_characters(project_id: str):
    """Use AI to detect characters, suggest voice assignments, and update the project."""
    project = _get_project_or_404(project_id)
    
    # Get available voices
    available_voices = tts_service.get_voices() if tts_service else []
    
    # Call LLM
    result = llm_analyze_characters(project.raw_text, available_voices)
    
    if not result:
        raise HTTPException(
            status_code=503,
            detail="AI analysis failed — LLM endpoint may be unreachable. Falling back to regex detection."
        )
    
    # Update project with AI results
    project.detected_characters = result.get("detected_characters", project.detected_characters)
    project.character_voice_map = result.get("character_voice_map", project.character_voice_map)
    project.character_descriptions = result.get("character_descriptions", {})
    
    visual_profiles = result.get("character_visual_profiles", {})
    
    # Update CharacterRef objects
    existing_chars = {c.name: c for c in project.characters}
    new_characters = []
    
    from app.audiobook_models import CharacterRef
    for name in project.detected_characters:
        ref = existing_chars.get(name, CharacterRef(name=name))
        ref.description = project.character_descriptions.get(name, "")
        if name in visual_profiles:
            ref.visual_profile = visual_profiles[name]
        new_characters.append(ref)
        
    project.characters = new_characters
    
    # Auto-design voices if the TTS backend supports it
    if hasattr(tts_service, "design_voice") and project.character_descriptions:
        logger.info(f"Auto-designing voices for characters using Moss VoiceGenerator")
        import re
        
        narrator_desc = result.get("narrator_description", "")
        if narrator_desc:
            voice_id = "gen_narrator"
            try:
                tts_service.design_voice(voice_id, narrator_desc)
                result["narrator_voice"] = voice_id
                available_voices.append(voice_id)
            except Exception as e:
                logger.error(f"Failed to design voice for narrator: {e}")

        # Store the narrator voice prompt on the project
        if narrator_desc:
            project.narrator_voice_prompt = narrator_desc
                
        for char_name, desc in project.character_descriptions.items():
            current_voice = project.character_voice_map.get(char_name)
            
            # Design a new voice if no valid existing voice was matched
            if not current_voice or current_voice not in available_voices:
                voice_id = re.sub(r'[^a-zA-Z0-9]', '', char_name.lower())
                if not voice_id:
                    voice_id = f"voice_{hash(char_name)}"
                voice_id = f"gen_{voice_id}" 
                
                try:
                    tts_service.design_voice(voice_id, desc)
                    project.character_voice_map[char_name] = voice_id
                except Exception as e:
                    logger.error(f"Failed to design voice for '{char_name}': {e}")
    
    narrator = result.get("narrator_voice", "")
    if narrator:
        project.narrator_voice = narrator
    
    # Re-assign voices across all segments using the new map
    for ch in project.chapters:
        assign_segment_voices(ch.segments, project.character_voice_map, project.narrator_voice)
    
    save_project(project)
    logger.info(f"AI analysis complete for project '{project_id}': {len(project.detected_characters)} characters detected")
    return project_to_detail_response(project)


# --- Segment operations ---

@router.put("/projects/{project_id}/segments/{segment_id}", response_model=ProjectDetailResponse)
async def update_segment(project_id: str, segment_id: str, request: UpdateSegmentRequest):
    """Edit a segment's text, voice, or scene prompt."""
    project = _get_project_or_404(project_id)
    _, _, seg = _find_segment(project, segment_id)
    
    if request.text is not None:
        seg.text = request.text
        seg.status = SegmentStatus.PENDING  # Reset status when text changes
        seg.audio_path = None
        seg.duration = None
    
    if request.voice_name is not None:
        seg.voice_name = request.voice_name

    if request.scene_prompt is not None:
        seg.scene_prompt = request.scene_prompt
        # If visual was already generated, allow re-generation with new prompt
        if seg.visual_status == "done":
            seg.visual_status = "pending"
    
    save_project(project)
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/segments/{segment_id}/split", response_model=ProjectDetailResponse)
async def split_segment(project_id: str, segment_id: str, request: SplitSegmentRequest = SplitSegmentRequest()):
    """
    Split a segment into two at the given character position or nearest sentence boundary.
    The original segment is replaced by two new segments with the same voice/character.
    """
    import re as _re
    project = _get_project_or_404(project_id)
    ch_idx, seg_idx, seg = _find_segment(project, segment_id)

    text = seg.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Segment too short to split")

    split_pos = request.split_at
    if split_pos is None or split_pos <= 0 or split_pos >= len(text):
        # Auto-split: find nearest sentence boundary to midpoint
        midpoint = len(text) // 2
        # Look for sentence-ending punctuation near the midpoint
        best_pos = None
        best_dist = len(text)
        for m in _re.finditer(r'[.!?]\s+', text):
            end = m.end()
            dist = abs(end - midpoint)
            if dist < best_dist:
                best_dist = dist
                best_pos = end
        split_pos = best_pos if best_pos else midpoint

    part1 = text[:split_pos].strip()
    part2 = text[split_pos:].strip()

    if not part1 or not part2:
        raise HTTPException(status_code=400, detail="Split would create an empty segment")

    # Create two new segments inheriting voice/character
    seg_a = Segment(text=part1, voice_name=seg.voice_name, character=seg.character, emotion=seg.emotion)
    seg_b = Segment(text=part2, voice_name=seg.voice_name, character=seg.character, emotion=seg.emotion)

    # Delete old audio file if it exists
    if seg.audio_path and os.path.isfile(seg.audio_path):
        os.remove(seg.audio_path)

    # Replace the original segment with the two new ones
    chapter = project.chapters[ch_idx]
    chapter.segments[seg_idx:seg_idx + 1] = [seg_a, seg_b]

    save_project(project)
    logger.info(f"Split segment '{segment_id}' into '{seg_a.id}' ({len(part1)} chars) + '{seg_b.id}' ({len(part2)} chars)")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/segments/{segment_id}/merge", response_model=ProjectDetailResponse)
async def merge_segment(project_id: str, segment_id: str):
    """
    Merge a segment with the next segment in the same chapter.
    Combines text, resets to PENDING, and deletes audio for both.
    """
    project = _get_project_or_404(project_id)
    ch_idx, seg_idx, seg = _find_segment(project, segment_id)
    chapter = project.chapters[ch_idx]

    if seg_idx >= len(chapter.segments) - 1:
        raise HTTPException(status_code=400, detail="No next segment to merge with")

    next_seg = chapter.segments[seg_idx + 1]

    # Combine text
    merged_text = seg.text.rstrip() + " " + next_seg.text.lstrip()

    # Delete audio files
    for s in [seg, next_seg]:
        if s.audio_path and os.path.isfile(s.audio_path):
            os.remove(s.audio_path)

    # Create merged segment (keep first segment's voice/character)
    merged = Segment(
        text=merged_text,
        voice_name=seg.voice_name,
        character=seg.character,
        emotion=seg.emotion,
    )

    # Replace the two segments with the merged one
    chapter.segments[seg_idx:seg_idx + 2] = [merged]

    save_project(project)
    logger.info(f"Merged segments '{segment_id}' + '{next_seg.id}' into '{merged.id}' ({len(merged_text)} chars)")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/retry-failed", response_model=ProjectDetailResponse)
async def retry_failed_segments(project_id: str):
    """Reset all ERROR segments to PENDING so they can be regenerated."""
    project = _get_project_or_404(project_id)
    reset_count = 0
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.status == SegmentStatus.ERROR:
                seg.status = SegmentStatus.PENDING
                seg.error_message = None
                # Delete broken audio if any
                if seg.audio_path and os.path.isfile(seg.audio_path):
                    os.remove(seg.audio_path)
                seg.audio_path = None
                seg.duration = None
                reset_count += 1

    save_project(project)
    logger.info(f"Reset {reset_count} failed segments in project '{project_id}'")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/segments/{segment_id}/generate")
async def generate_segment(project_id: str, segment_id: str):
    """Enqueue audio generation for a single segment."""
    from .generation_queue import enqueue_tts
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    project = _get_project_or_404(project_id)
    _, _, seg = _find_segment(project, segment_id)
    seg.status = SegmentStatus.GENERATING  # show queued immediately
    save_project(project)
    depth = enqueue_tts(project_id, segment_id)
    return {"status": "queued", "queue_depth": depth, "segment_id": segment_id}


def _split_text_into_chunks(text: str, max_len: int) -> list:
    """Split text at sentence boundaries to stay under max_len per chunk."""
    import re
    # Split at sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = ""
    for sent in sentences:
        if len(sent) > max_len:
            # Sentence itself is too long — split at clause boundaries
            if current:
                chunks.append(current.strip())
                current = ""
            sub_parts = re.split(r'(?<=[,;:])\s+', sent)
            for part in sub_parts:
                if len(current) + len(part) + 1 > max_len:
                    if current:
                        chunks.append(current.strip())
                    current = part
                else:
                    current = f"{current} {part}".strip() if current else part
        elif len(current) + len(sent) + 1 > max_len:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip() if current else sent
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks if chunks else [text[:max_len]]


@router.get("/projects/{project_id}/segments/{segment_id}/audio")
async def get_segment_audio(project_id: str, segment_id: str):
    """Stream the audio for a generated segment."""
    project = _get_project_or_404(project_id)
    _, _, seg = _find_segment(project, segment_id)
    
    if not seg.audio_path or not os.path.exists(seg.audio_path):
        raise HTTPException(status_code=404, detail="Audio not generated yet")
    
    with open(seg.audio_path, "rb") as f:
        audio_bytes = f.read()
    
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename={segment_id}.wav"},
    )


# --- Chapter operations ---

@router.post("/projects/{project_id}/chapters/{chapter_idx}/generate")
async def generate_chapter(project_id: str, chapter_idx: int):
    """Generate all pending segments in a chapter."""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    project = _get_project_or_404(project_id)
    
    if chapter_idx < 0 or chapter_idx >= len(project.chapters):
        raise HTTPException(status_code=404, detail=f"Chapter {chapter_idx} not found")
    
    chapter = project.chapters[chapter_idx]
    results = {"generated": 0, "errors": 0, "skipped": 0}
    
    for seg in chapter.segments:
        if seg.status == SegmentStatus.DONE:
            results["skipped"] += 1
            continue
        
        voice = seg.voice_name or project.narrator_voice
        if not voice:
            seg.status = SegmentStatus.ERROR
            seg.error_message = "No voice assigned"
            results["errors"] += 1
            continue
        
        seg.status = SegmentStatus.GENERATING
        save_project(project)
        
        try:
            audio, sample_rate = tts_service.generate_speech(
                text=seg.text, voice_name=voice, language="en"
            )
            
            if audio is None or len(audio) == 0:
                raise RuntimeError("TTS returned empty audio")
            
            audio_dir = os.path.join(get_project_dir(project_id), "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{seg.id}.wav")
            sf.write(audio_path, audio, sample_rate, format="WAV", subtype="PCM_16")
            
            seg.status = SegmentStatus.DONE
            seg.audio_path = audio_path
            seg.duration = len(audio) / sample_rate
            seg.error_message = None
            results["generated"] += 1
            
        except Exception as e:
            seg.status = SegmentStatus.ERROR
            seg.error_message = str(e)
            results["errors"] += 1
            logger.error(f"Failed segment '{seg.id}': {e}")
    
    save_project(project)
    return results


@router.get("/projects/{project_id}/chapters/{chapter_idx}/export")
async def export_chapter(project_id: str, chapter_idx: int):
    """Export concatenated chapter audio."""
    project = _get_project_or_404(project_id)
    
    if chapter_idx < 0 or chapter_idx >= len(project.chapters):
        raise HTTPException(status_code=404, detail=f"Chapter {chapter_idx} not found")
    
    chapter = project.chapters[chapter_idx]
    audio_segments = []
    sample_rate = None
    
    for seg in chapter.segments:
        if seg.audio_path and os.path.exists(seg.audio_path):
            data, sr = sf.read(seg.audio_path)
            audio_segments.append(data)
            sample_rate = sr
    
    if not audio_segments:
        raise HTTPException(status_code=400, detail="No audio segments generated in this chapter")
    
    # Add 0.5s silence between segments
    silence = np.zeros(int(sample_rate * 0.5))
    combined = []
    for i, seg_audio in enumerate(audio_segments):
        combined.append(seg_audio)
        if i < len(audio_segments) - 1:
            combined.append(silence)
    
    full_audio = np.concatenate(combined)
    
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, full_audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    
    chapter_name = chapter.title.replace(" ", "_")[:50] or f"chapter_{chapter_idx}"
    
    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={chapter_name}.wav"},
    )


# --- Full book operations ---

@router.post("/projects/{project_id}/generate-all")
async def generate_all(project_id: str):
    """Enqueue all pending segments for audio generation."""
    from .generation_queue import enqueue_tts
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    project = _get_project_or_404(project_id)
    enqueued = 0
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.status == SegmentStatus.DONE:
                continue
            voice = seg.voice_name or project.narrator_voice
            if not voice:
                continue
            seg.status = SegmentStatus.GENERATING
            enqueue_tts(project_id, seg.id)
            enqueued += 1
    save_project(project)
    return {"status": "queued", "enqueued": enqueued}


@router.get("/projects/{project_id}/export")
async def export_full_book(project_id: str):
    """Export the full audiobook as a single WAV file."""
    project = _get_project_or_404(project_id)
    
    all_audio = []
    sample_rate = None
    
    for ch in project.chapters:
        chapter_audio = []
        for seg in ch.segments:
            if seg.audio_path and os.path.exists(seg.audio_path):
                data, sr = sf.read(seg.audio_path)
                chapter_audio.append(data)
                sample_rate = sr
        
        if chapter_audio:
            # Silence between segments (0.5s)
            seg_silence = np.zeros(int(sample_rate * 0.5))
            chapter_combined = []
            for i, seg_audio in enumerate(chapter_audio):
                chapter_combined.append(seg_audio)
                if i < len(chapter_audio) - 1:
                    chapter_combined.append(seg_silence)
            all_audio.append(np.concatenate(chapter_combined))
    
    if not all_audio:
        raise HTTPException(status_code=400, detail="No audio segments generated")
    
    # 2s silence between chapters
    chapter_silence = np.zeros(int(sample_rate * 2.0))
    full_combined = []
    for i, ch_audio in enumerate(all_audio):
        full_combined.append(ch_audio)
        if i < len(all_audio) - 1:
            full_combined.append(chapter_silence)
    
    full_audio = np.concatenate(full_combined)
    
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, full_audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    
    filename = project.name.replace(" ", "_")[:50] or "audiobook"
    
    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}.wav"},
    )


@router.get("/projects/{project_id}/download-all")
async def download_all_assets(project_id: str):
    """
    Download all project assets (audio WAVs, visuals images/videos, portraits) as a ZIP file.
    Organized into audio/, visuals/, portraits/ subfolders.
    """
    import zipfile

    project = _get_project_or_404(project_id)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project.name or project_id)[:50]

    # Build in-memory zip
    buf = io.BytesIO()
    added = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # Audio WAVs — audio/{chapter_title}/{segment_id}.wav
        for ch in project.chapters:
            ch_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', ch.title or f"chapter_{project.chapters.index(ch)}")[:50]
            for seg in ch.segments:
                if seg.audio_path and os.path.exists(seg.audio_path):
                    zf.write(seg.audio_path, f"audio/{ch_safe}/{seg.id}.wav")
                    added += 1

        # Visuals — visuals/{segment_id}.ext
        for ch in project.chapters:
            for seg in ch.segments:
                if seg.visual_path and os.path.exists(seg.visual_path):
                    ext = os.path.splitext(seg.visual_path)[1]
                    zf.write(seg.visual_path, f"visuals/{seg.id}{ext}")
                    added += 1

        # Portraits — portraits/{character_name}.ext
        for char_ref in project.characters:
            if char_ref.portrait_path and os.path.exists(char_ref.portrait_path):
                ext = os.path.splitext(char_ref.portrait_path)[1]
                safe_char = re.sub(r'[^a-zA-Z0-9_-]', '_', char_ref.name)[:50]
                zf.write(char_ref.portrait_path, f"portraits/{safe_char}{ext}")
                added += 1
            for i, vpath in enumerate(char_ref.portrait_variants or []):
                if vpath and os.path.exists(vpath):
                    ext = os.path.splitext(vpath)[1]
                    safe_char = re.sub(r'[^a-zA-Z0-9_-]', '_', char_ref.name)[:50]
                    zf.write(vpath, f"portraits/{safe_char}_v{i+1}{ext}")
                    added += 1

        # Assembled video if it exists
        video_path = os.path.join(get_project_dir(project_id), "video", "full_book.mp4")
        if os.path.exists(video_path):
            zf.write(video_path, "full_book.mp4")
            added += 1

    if added == 0:
        raise HTTPException(status_code=400, detail="No assets found for this project. Generate audio and/or visuals first.")

    buf.seek(0)
    logger.info(f"Serving download-all zip for project {project_id}: {added} files")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_assets.zip"},
    )


# ============================================================
# Visual Generation Endpoints
# ============================================================


@router.post("/projects/{project_id}/reset-stuck-visuals", response_model=ProjectDetailResponse)
async def reset_stuck_visuals(project_id: str):
    """Reset any segments stuck in 'generating' visual status back to retryable state."""
    project = _get_project_or_404(project_id)
    reset_count = 0
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_status == "generating":
                seg.visual_status = "none" if not seg.scene_prompt else "pending"
                reset_count += 1
    if reset_count:
        save_project(project)
        logger.info(f"Reset {reset_count} stuck visual generation(s) for project {project_id}")
    return project_to_detail_response(project)


@router.get("/projects/{project_id}/segments/{segment_id}/visual")
async def get_segment_visual(project_id: str, segment_id: str):
    """Serve the generated visual image for a segment."""
    from fastapi.responses import FileResponse

    project = _get_project_or_404(project_id)
    _, _, seg = _find_segment(project, segment_id)

    if not seg.visual_path or not os.path.exists(seg.visual_path):
        raise HTTPException(status_code=404, detail="Visual not generated yet")

    ext = os.path.splitext(seg.visual_path)[1].lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    media_type = media_types.get(ext, "image/png")

    return FileResponse(
        path=seg.visual_path,
        media_type=media_type,
        filename=f"{segment_id}{ext}",
    )

@router.post("/projects/{project_id}/segments/{segment_id}/generate-visual", response_model=ProjectDetailResponse)
async def generate_segment_visual(
    project_id: str,
    segment_id: str,
    mode: str = None,
    frames: int = None,
    fps: int = None,
    width: int = None,
    height: int = None,
    animation: str = None,
    ref_character: str = None,
):
    """Enqueue visual generation for a single segment. Auto-creates a VisualAsset if needed."""
    from .generation_queue import enqueue_visual
    project = _get_project_or_404(project_id)
    _, _, segment = _find_segment(project, segment_id)

    # Auto-create a VisualAsset if segment doesn't have one
    if not segment.visual_id:
        asset = VisualAsset(
            label=(segment.scene_prompt or "")[:40].strip() or f"Visual for seg {segment.id}",
            scene_prompt=segment.scene_prompt,
            visual_path=segment.visual_path,
            visual_type=segment.visual_type,
            visual_mode=segment.visual_mode,
            visual_status=segment.visual_status,
            animation_style=segment.animation_style,
            video_fill_mode=segment.video_fill_mode,
        )
        project.visuals.append(asset)
        segment.visual_id = asset.id

    va = _resolve_visual(project, segment.visual_id)
    # Persist animation style choice immediately
    if animation is not None and va:
        va.animation_style = animation
    if va:
        va.visual_status = "queued"
    save_project(project)
    depth = enqueue_visual(
        project_id=project_id,
        segment_id=segment_id,
        mode=mode,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        animation=animation,
        ref_character=ref_character,
    )
    logger.info(f"Enqueued visual generation for {segment_id} (depth {depth})")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/segments/{segment_id}/generate-scene-prompt", response_model=ProjectDetailResponse)
async def generate_segment_scene_prompt(project_id: str, segment_id: str):
    """Enqueue scene prompt generation for a single segment."""
    from .generation_queue import enqueue_prompt
    project = _get_project_or_404(project_id)
    _, _, segment = _find_segment(project, segment_id)
    depth = enqueue_prompt(project_id, segment_id)
    logger.info(f"Enqueued prompt generation for {segment_id} (depth {depth})")
    return project_to_detail_response(project)


@router.patch("/projects/{project_id}/segments/{segment_id}/video-fill-mode", response_model=ProjectDetailResponse)
async def set_video_fill_mode(project_id: str, segment_id: str, mode: str):
    """Set the video_fill_mode for a segment. Options: loop | hold | fade."""
    if mode not in ("loop", "hold", "fade"):
        raise HTTPException(status_code=400, detail="mode must be loop, hold, or fade")
    project = _get_project_or_404(project_id)
    _, _, segment = _find_segment(project, segment_id)
    segment.video_fill_mode = mode
    save_project(project)
    return project_to_detail_response(project)



    from .audiobook_llm import generate_scene_prompt, generate_visual_style

    project = _get_project_or_404(project_id)

    # Auto-generate visual style if missing
    if not project.visual_style and project.raw_text:
        style = generate_visual_style(project.raw_text)
        if style:
            project.visual_style = style
            save_project(project)

    # Find segment + context
    all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]
    segment, chapter, seg_idx = None, None, None
    for i, (ch, seg) in enumerate(all_segments):
        if seg.id == segment_id:
            segment, chapter, seg_idx = seg, ch, i
            break

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    prev_scene = all_segments[seg_idx - 1][1].scene_prompt if seg_idx > 0 else ""
    next_text = all_segments[seg_idx + 1][1].text if seg_idx < len(all_segments) - 1 else ""

    scene = generate_scene_prompt(
        text=segment.text,
        chapter_title=chapter.title if chapter else "",
        character=segment.character or "Narrator",
        emotion=segment.emotion or "neutral",
        visual_style=project.visual_style,
        prev_scene=prev_scene or "",
        next_text=next_text or "",
    )
    if not scene:
        raise HTTPException(status_code=500, detail="LLM failed to generate scene prompt")

    segment.scene_prompt = scene
    # Reset visual status so user knows a new visual is needed
    if segment.visual_status == "done":
        segment.visual_status = "pending"
    save_project(project)
    logger.info(f"Regenerated scene prompt for segment {segment_id}")
    return project_to_detail_response(project)



@router.post("/projects/{project_id}/generate-visuals", response_model=ProjectDetailResponse)
async def generate_all_visuals(project_id: str, mode: str = None):
    """Enqueue visual generation for all segments that don't already have one."""
    from .generation_queue import enqueue_visual
    project = _get_project_or_404(project_id)
    enqueued = 0
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_status == "done":
                continue
            seg.visual_status = "queued"
            enqueue_visual(
                project_id=project_id,
                segment_id=seg.id,
                mode=mode,
            )
            enqueued += 1
    save_project(project)
    logger.info(f"Enqueued {enqueued} visual generation jobs for project {project_id}")
    return project_to_detail_response(project)


# --- Visual Asset CRUD ---

def _find_visual_or_404(project: AudiobookProject, visual_id: str) -> VisualAsset:
    """Find a VisualAsset by id or raise 404."""
    for va in project.visuals:
        if va.id == visual_id:
            return va
    raise HTTPException(status_code=404, detail=f"Visual asset '{visual_id}' not found")


@router.get("/projects/{project_id}/visuals")
async def list_visual_assets(project_id: str):
    """List all visual assets for a project."""
    project = _get_project_or_404(project_id)
    # Compute assignment counts
    counts: Dict[str, int] = {}
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_id:
                counts[seg.visual_id] = counts.get(seg.visual_id, 0) + 1
    return [
        VisualAssetResponse(
            id=va.id, label=va.label, scene_prompt=va.scene_prompt,
            has_visual=va.visual_path is not None and os.path.exists(va.visual_path) if va.visual_path else False,
            visual_type=va.visual_type, visual_mode=va.visual_mode,
            visual_status=va.visual_status, animation_style=va.animation_style,
            video_fill_mode=va.video_fill_mode, ref_character=va.ref_character,
            gen_frames=va.gen_frames, gen_fps=va.gen_fps,
            gen_width=va.gen_width, gen_height=va.gen_height,
            gen_enable_audio=va.gen_enable_audio, gen_two_stage=va.gen_two_stage,
            created_at=va.created_at, assigned_segments=counts.get(va.id, 0),
        )
        for va in project.visuals
    ]


class CreateVisualRequest(BaseModel):
    label: str = ""
    scene_prompt: Optional[str] = None


@router.post("/projects/{project_id}/visuals", response_model=ProjectDetailResponse, status_code=201)
async def create_visual_asset(project_id: str, request: CreateVisualRequest):
    """Create a new empty visual asset."""
    project = _get_project_or_404(project_id)
    asset = VisualAsset(
        label=request.label or (request.scene_prompt or "")[:40].strip() or "New Visual",
        scene_prompt=request.scene_prompt,
    )
    project.visuals.append(asset)
    save_project(project)
    logger.info(f"Created visual asset '{asset.id}' in project {project_id}")
    return project_to_detail_response(project)


class UpdateVisualRequest(BaseModel):
    label: Optional[str] = None
    scene_prompt: Optional[str] = None
    animation_style: Optional[str] = None
    video_fill_mode: Optional[str] = None
    ref_character: Optional[str] = None


@router.put("/projects/{project_id}/visuals/{visual_id}", response_model=ProjectDetailResponse)
async def update_visual_asset(project_id: str, visual_id: str, request: UpdateVisualRequest):
    """Update a visual asset's metadata."""
    project = _get_project_or_404(project_id)
    va = _find_visual_or_404(project, visual_id)
    if request.label is not None:
        va.label = request.label
    if request.scene_prompt is not None:
        va.scene_prompt = request.scene_prompt
        # Reset visual status so user knows a new visual is needed
        if va.visual_status == "done":
            va.visual_status = "pending"
    if request.animation_style is not None:
        va.animation_style = request.animation_style
    if request.video_fill_mode is not None:
        if request.video_fill_mode not in ("loop", "hold", "fade"):
            raise HTTPException(status_code=400, detail="video_fill_mode must be loop, hold, or fade")
        va.video_fill_mode = request.video_fill_mode
    if request.ref_character is not None:
        va.ref_character = request.ref_character
    save_project(project)
    logger.info(f"Updated visual asset '{visual_id}' in project {project_id}")
    return project_to_detail_response(project)


@router.delete("/projects/{project_id}/visuals/{visual_id}", response_model=ProjectDetailResponse)
async def delete_visual_asset(project_id: str, visual_id: str):
    """Delete a visual asset and unlink it from all segments."""
    project = _get_project_or_404(project_id)
    va = _find_visual_or_404(project, visual_id)
    # Unlink from all segments
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_id == visual_id:
                seg.visual_id = None
    # Remove asset
    project.visuals = [v for v in project.visuals if v.id != visual_id]
    save_project(project)
    logger.info(f"Deleted visual asset '{visual_id}' from project {project_id}")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/visuals/{visual_id}/generate", response_model=ProjectDetailResponse)
async def generate_visual_asset(
    project_id: str,
    visual_id: str,
    mode: str = None,
    frames: int = None,
    fps: int = None,
    width: int = None,
    height: int = None,
    animation: str = None,
    ref_character: str = None,
    enable_audio: bool = False,
    two_stage: bool = False,
):
    """Enqueue visual generation for a visual asset.
    Uses the first assigned segment for context (text, character)."""
    from .generation_queue import enqueue_visual
    project = _get_project_or_404(project_id)
    va = _find_visual_or_404(project, visual_id)

    # Find first segment assigned to this visual for context
    context_segment_id = None
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_id == visual_id:
                context_segment_id = seg.id
                break
        if context_segment_id:
            break

    if not context_segment_id:
        raise HTTPException(
            status_code=400,
            detail="Visual must be assigned to at least one segment before generating (needs text context)"
        )

    if animation is not None:
        va.animation_style = animation
    if ref_character is not None:
        va.ref_character = ref_character
    # Persist generation settings on the visual asset
    if mode is not None:
        va.visual_mode = mode
    if frames is not None:
        va.gen_frames = frames
    if fps is not None:
        va.gen_fps = fps
    if width is not None:
        va.gen_width = width
    if height is not None:
        va.gen_height = height
    va.gen_enable_audio = enable_audio
    va.gen_two_stage = two_stage
    va.visual_status = "queued"
    save_project(project)

    depth = enqueue_visual(
        project_id=project_id,
        segment_id=context_segment_id,
        mode=mode,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        animation=animation,
        ref_character=ref_character,
        enable_audio=enable_audio,
        two_stage=two_stage,
    )
    logger.info(f"Enqueued visual generation for asset '{visual_id}' via segment {context_segment_id} (depth {depth})")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/visuals/{visual_id}/generate-prompt", response_model=ProjectDetailResponse)
async def generate_visual_scene_prompt(project_id: str, visual_id: str):
    """Generate a scene prompt for a visual asset using LLM.
    Uses the first assigned segment for text context."""
    from .audiobook_llm import generate_scene_prompt, generate_visual_style

    project = _get_project_or_404(project_id)
    va = _find_visual_or_404(project, visual_id)

    # Auto-generate visual style if missing
    if not project.visual_style and project.raw_text:
        style = generate_visual_style(project.raw_text)
        if style:
            project.visual_style = style
            save_project(project)

    # Find first assigned segment for context
    context_seg = None
    all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]
    seg_idx = None
    for i, (ch, seg) in enumerate(all_segments):
        if seg.visual_id == visual_id:
            context_seg = seg
            seg_idx = i
            break

    if not context_seg:
        raise HTTPException(
            status_code=400,
            detail="Visual must be assigned to at least one segment before generating a scene prompt"
        )

    prev_scene = all_segments[seg_idx - 1][1].scene_prompt if seg_idx > 0 else ""
    next_text = all_segments[seg_idx + 1][1].text if seg_idx < len(all_segments) - 1 else ""
    ch = all_segments[seg_idx][0]

    scene = generate_scene_prompt(
        text=context_seg.text,
        chapter_title=ch.title if ch else "",
        character=context_seg.character or "Narrator",
        emotion=context_seg.emotion or "neutral",
        visual_style=project.visual_style,
        prev_scene=prev_scene or "",
        next_text=next_text or "",
    )
    if not scene:
        raise HTTPException(status_code=500, detail="LLM failed to generate scene prompt")

    va.scene_prompt = scene
    if not va.label or va.label == "New Visual":
        va.label = scene[:40].strip()
    if va.visual_status == "done":
        va.visual_status = "pending"
    save_project(project)
    logger.info(f"Generated scene prompt for visual asset '{visual_id}'")
    return project_to_detail_response(project)


class AssignVisualRequest(BaseModel):
    visual_id: Optional[str] = None  # None to unassign


@router.post("/projects/{project_id}/segments/{segment_id}/assign-visual", response_model=ProjectDetailResponse)
async def assign_visual_to_segment(project_id: str, segment_id: str, request: AssignVisualRequest):
    """Assign (or unassign) a visual asset to a segment."""
    project = _get_project_or_404(project_id)
    _, _, segment = _find_segment(project, segment_id)

    if request.visual_id is not None:
        # Verify the visual exists
        _find_visual_or_404(project, request.visual_id)

    segment.visual_id = request.visual_id
    save_project(project)
    logger.info(f"{'Assigned' if request.visual_id else 'Unassigned'} visual for segment {segment_id}")
    return project_to_detail_response(project)


@router.get("/projects/{project_id}/visuals/{visual_id}/file")
async def get_visual_asset_file(project_id: str, visual_id: str):
    """Serve the generated visual file for a visual asset."""
    from fastapi.responses import FileResponse
    project = _get_project_or_404(project_id)
    va = _find_visual_or_404(project, visual_id)

    if not va.visual_path or not os.path.exists(va.visual_path):
        raise HTTPException(status_code=404, detail="Visual not generated yet")

    ext = os.path.splitext(va.visual_path)[1].lower()
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".mp4": "video/mp4", ".gif": "image/gif",
    }.get(ext, "application/octet-stream")

    return FileResponse(path=va.visual_path, media_type=media_type, filename=f"{visual_id}{ext}")


@router.post("/projects/{project_id}/generate-style", response_model=ProjectDetailResponse)
async def generate_project_style(project_id: str):
    """Auto-generate a visual style from the project's text."""
    from .audiobook_llm import generate_visual_style

    project = _get_project_or_404(project_id)
    style = generate_visual_style(project.raw_text)
    if not style:
        raise HTTPException(status_code=500, detail="Failed to generate visual style")
    project.visual_style = style
    save_project(project)
    logger.info(f"Generated visual style for {project_id}: {style}")
    return project_to_detail_response(project)


class UpdateStyleRequest(BaseModel):
    visual_style: str


@router.put("/projects/{project_id}/visual-style", response_model=ProjectDetailResponse)
async def update_visual_style(project_id: str, req: UpdateStyleRequest):
    """Manually set the project's visual style."""
    project = _get_project_or_404(project_id)
    project.visual_style = req.visual_style
    save_project(project)
    return project_to_detail_response(project)


def _find_character_portrait(project, segment) -> Optional[str]:
    """Find the ComfyUI portrait filename for the character speaking in a segment."""
    if not segment.character or not project.characters:
        return None
    char_name = segment.character.lower()
    for char_ref in project.characters:
        if char_ref.name.lower() == char_name and char_ref.portrait_comfyui:
            return char_ref.portrait_comfyui
    return None


def _find_character_portrait_by_name(project, character_name: str) -> Optional[str]:
    """Find the ComfyUI portrait filename for a named character."""
    if not character_name or not project.characters:
        return None
    target = character_name.lower()
    for char_ref in project.characters:
        if char_ref.name.lower() == target and char_ref.portrait_comfyui:
            return char_ref.portrait_comfyui
    return None


@router.post("/projects/{project_id}/extract-characters", response_model=ProjectDetailResponse)
async def extract_characters_endpoint(project_id: str):
    """Use LLM to extract character appearances from book text."""
    from .audiobook_llm import extract_characters
    from .audiobook_models import CharacterRef

    project = _get_project_or_404(project_id)
    if not project.raw_text:
        raise HTTPException(status_code=400, detail="Project has no text")

    # Group segments into chunks of ~8000 characters to process the entire book
    chunks = []
    current_chunk = ""
    for ch in project.chapters:
        for seg in ch.segments:
            if len(current_chunk) + len(seg.text) > 8000:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = seg.text
            else:
                current_chunk += "\n\n" + seg.text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        # Fallback if no segments
        chunks = [project.raw_text]

    all_characters_data = []
    max_retries = 3
    for attempt in range(max_retries):
        for chunk in chunks:
            chars = extract_characters(chunk)
            if chars:
                all_characters_data.extend(chars)
        if all_characters_data:
            break
        logger.warning(f"Character extraction attempt {attempt + 1}/{max_retries} returned empty, retrying...")

    if not all_characters_data:
        raise HTTPException(status_code=500, detail="LLM failed to extract characters after multiple attempts")

    # Merge with existing characters (preserve portraits)
    existing = {c.name.lower(): c for c in project.characters}
    new_chars_dict = {}

    # First add existing so they are preserved
    for name, c in existing.items():
        new_chars_dict[name] = c

    for char_data in all_characters_data:
        name = char_data["name"]
        name_lower = name.lower()
        if name_lower in new_chars_dict:
            # Update fields with new extraction data
            new_chars_dict[name_lower].description = char_data["description"]
            if char_data.get("portrait_prompt"):
                new_chars_dict[name_lower].portrait_prompt = char_data["portrait_prompt"]
            if char_data.get("voice_prompt"):
                new_chars_dict[name_lower].voice_prompt = char_data["voice_prompt"]
        else:
            new_chars_dict[name_lower] = CharacterRef(
                name=name,
                description=char_data["description"],
                portrait_prompt=char_data.get("portrait_prompt"),
                voice_prompt=char_data.get("voice_prompt"),
            )

    project.characters = list(new_chars_dict.values())
    save_project(project)
    logger.info(f"Extracted {len(project.characters)} characters for project {project_id}")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/generate-portraits", response_model=ProjectDetailResponse)
async def generate_portraits(project_id: str):
    """Generate hero portrait images for all characters without portraits."""
    from .audiobook_llm import generate_portrait_prompt
    from .comfyui_service import generate_image, upload_image

    project = _get_project_or_404(project_id)
    if not project.characters:
        raise HTTPException(status_code=400, detail="No characters. Run extract-characters first.")

    portrait_dir = os.path.join(get_project_dir(project_id), "portraits")
    os.makedirs(portrait_dir, exist_ok=True)

    generated = 0
    for char_ref in project.characters:
        if char_ref.portrait_path and os.path.exists(char_ref.portrait_path):
            continue  # Already has portrait

        # Use existing portrait prompt if available, otherwise generate via LLM
        portrait_prompt = char_ref.portrait_prompt
        if not portrait_prompt:
            portrait_prompt = generate_portrait_prompt(char_ref.name, char_ref.description)
            if portrait_prompt:
                char_ref.portrait_prompt = portrait_prompt
                save_project(project)
        if not portrait_prompt:
            logger.warning(f"Failed to generate portrait prompt for {char_ref.name}")
            continue

        # Generate portrait image
        try:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', char_ref.name.lower())
            path = await generate_image(
                prompt=portrait_prompt,
                output_dir=portrait_dir,
                prefix=f"portrait_{safe_name}",
            )
            char_ref.portrait_path = path

            # Upload to ComfyUI for use in video generation
            comfyui_name = await upload_image(path)
            char_ref.portrait_comfyui = comfyui_name

            generated += 1
            logger.info(f"Generated portrait for {char_ref.name}: {path} (ComfyUI: {comfyui_name})")
        except Exception as e:
            logger.error(f"Portrait generation failed for {char_ref.name}: {e}")

    save_project(project)
    return project_to_detail_response(project)


@router.put("/projects/{project_id}/characters/{character_name}")
async def update_character(
    project_id: str, character_name: str,
    description: str = None,
    portrait_prompt: str = None,
    voice_prompt: str = None,
):
    """Update a character's description, portrait prompt, or voice prompt."""
    project = _get_project_or_404(project_id)

    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break

    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    if description is not None:
        char_ref.description = description
        # Clear portrait so it gets regenerated
        char_ref.portrait_path = None
        char_ref.portrait_comfyui = None
    
    if portrait_prompt is not None:
        char_ref.portrait_prompt = portrait_prompt
        # Clear portrait so it gets regenerated with new prompt
        char_ref.portrait_path = None
        char_ref.portrait_comfyui = None
    
    if voice_prompt is not None:
        char_ref.voice_prompt = voice_prompt

    save_project(project)
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/characters/{character_name}/generate-portrait-variant")
async def generate_portrait_variant(project_id: str, character_name: str):
    """Generate a new portrait variant for a character and return updated project."""
    from .audiobook_llm import generate_portrait_prompt
    from .comfyui_service import generate_image, upload_image

    project = _get_project_or_404(project_id)
    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break
    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    if not char_ref.description:
        raise HTTPException(status_code=400, detail="Character has no description. Run character extraction first.")

    # Use stored portrait_prompt if available, otherwise generate via LLM
    portrait_prompt = char_ref.portrait_prompt
    if not portrait_prompt:
        portrait_prompt = generate_portrait_prompt(char_ref.name, char_ref.description)
    if not portrait_prompt:
        raise HTTPException(status_code=500, detail="Failed to generate portrait prompt")

    portrait_dir = os.path.join(get_project_dir(project_id), "portraits")
    os.makedirs(portrait_dir, exist_ok=True)

    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', char_ref.name.lower())
    variant_idx = len(char_ref.portrait_variants) + 1
    try:
        path = await generate_image(
            prompt=portrait_prompt,
            output_dir=portrait_dir,
            prefix=f"portrait_{safe_name}_v{variant_idx}",
            width=768,
            height=1024,
        )
        char_ref.portrait_variants.append(path)

        # If this is the first portrait, also set it as the primary
        if not char_ref.portrait_path or not os.path.exists(char_ref.portrait_path):
            char_ref.portrait_path = path
            comfyui_name = await upload_image(path)
            char_ref.portrait_comfyui = comfyui_name

        save_project(project)
        logger.info(f"Generated portrait variant {variant_idx} for {char_ref.name}")
        return project_to_detail_response(project)
    except Exception as e:
        logger.error(f"Portrait variant generation failed for {char_ref.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}/characters/{character_name}/select-portrait")
async def select_portrait(project_id: str, character_name: str, variant_index: int = 0):
    """Select a specific portrait variant as the primary portrait for a character."""
    from .comfyui_service import upload_image

    project = _get_project_or_404(project_id)
    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break
    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    if variant_index < 0 or variant_index >= len(char_ref.portrait_variants):
        raise HTTPException(status_code=400, detail=f"Invalid variant index {variant_index}")

    selected_path = char_ref.portrait_variants[variant_index]
    if not os.path.exists(selected_path):
        raise HTTPException(status_code=404, detail="Portrait file not found on disk")

    char_ref.portrait_path = selected_path
    try:
        comfyui_name = await upload_image(selected_path)
        char_ref.portrait_comfyui = comfyui_name
    except Exception as e:
        logger.warning(f"Failed to upload portrait to ComfyUI: {e}")

    save_project(project)
    return project_to_detail_response(project)


@router.get("/projects/{project_id}/characters/{character_name}/portrait")
async def get_character_portrait(project_id: str, character_name: str, variant: int = -1):
    """Serve a character's portrait image. Use variant=-1 for primary, or 0..N for variants."""
    from fastapi.responses import FileResponse

    project = _get_project_or_404(project_id)
    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break
    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    if variant >= 0:
        if variant >= len(char_ref.portrait_variants):
            raise HTTPException(status_code=404, detail="Variant not found")
        path = char_ref.portrait_variants[variant]
    else:
        path = char_ref.portrait_path

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No portrait available")

    return FileResponse(path, media_type="image/png")


@router.post("/projects/{project_id}/characters/{character_name}/preview-voice")
async def preview_character_voice(project_id: str, character_name: str):
    """Generate a short TTS voice preview for a character."""
    project = _get_project_or_404(project_id)

    # Find the character's voice
    voice_name = project.character_voice_map.get(character_name)
    if not voice_name:
        # Try case-insensitive
        for k, v in project.character_voice_map.items():
            if k.lower() == character_name.lower():
                voice_name = v
                break
    if not voice_name:
        voice_name = project.narrator_voice or ""

    if not voice_name:
        raise HTTPException(status_code=400, detail="No voice assigned to this character")

    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service unavailable")

    # Use a short preview sentence
    preview_text = f"Hello, my name is {character_name}. Pleased to make your acquaintance."
    try:
        audio_data, sample_rate = await asyncio.to_thread(
            tts_service.generate_speech, preview_text, voice_name
        )
        # Convert numpy audio to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
        wav_bytes = wav_buffer.getvalue()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"inline; filename=preview_{character_name}.wav"},
        )
    except Exception as e:
        logger.error(f"Voice preview failed for {character_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/characters/{character_name}/design-voice", response_model=ProjectDetailResponse)
async def design_character_voice(project_id: str, character_name: str):
    """Design a unique TTS voice for a character using their voice_prompt via MOSS VoiceGenerator."""
    project = _get_project_or_404(project_id)

    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break

    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    voice_description = char_ref.voice_prompt
    if not voice_description:
        raise HTTPException(status_code=400, detail="No voice prompt set for this character. Run character extraction or set a voice prompt first.")

    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service unavailable")

    if not hasattr(tts_service, 'design_voice'):
        raise HTTPException(status_code=503, detail="Voice design not supported by current TTS backend")

    try:
        # Create a sanitized voice name from the character name
        import re as _re
        voice_name = _re.sub(r'[^a-zA-Z0-9_-]', '_', character_name).strip('_')
        voice_name = f"char_{voice_name}"

        logger.info(f"Designing voice '{voice_name}' for character '{character_name}' with prompt: {voice_description[:100]}...")

        # Use MOSS VoiceGenerator to design the voice
        voice_path = await asyncio.to_thread(tts_service.design_voice, voice_name, voice_description)

        # Assign the designed voice to the character in the voice map
        project.character_voice_map[character_name] = voice_name
        save_project(project)

        logger.info(f"Designed voice '{voice_name}' for character '{character_name}' saved to {voice_path}")
        return project_to_detail_response(project)
    except Exception as e:
        logger.error(f"Voice design failed for {character_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateNarratorVoicePromptRequest(BaseModel):
    narrator_voice_prompt: str


@router.post("/projects/{project_id}/generate-narrator-voice", response_model=ProjectDetailResponse)
async def generate_narrator_voice(project_id: str):
    """Generate a narrator voice prompt from book text via LLM, then design the voice via MOSS."""
    project = _get_project_or_404(project_id)

    if not project.raw_text:
        raise HTTPException(status_code=400, detail="No book text available")

    # Step 1: Generate narrator voice description from book text
    voice_desc = llm_generate_narrator_voice_prompt(project.raw_text)
    if not voice_desc:
        raise HTTPException(status_code=503, detail="LLM failed to generate narrator voice description")

    project.narrator_voice_prompt = voice_desc
    save_project(project)
    logger.info(f"Generated narrator voice prompt for project '{project_id}': {voice_desc[:100]}...")

    # Step 2: Design voice via MOSS if available
    if tts_service and hasattr(tts_service, 'design_voice'):
        try:
            voice_name = "narrator_voice"
            await asyncio.to_thread(tts_service.design_voice, voice_name, voice_desc)
            project.narrator_voice = voice_name

            # Re-assign narrator voice to all non-character segments
            for ch in project.chapters:
                assign_segment_voices(ch.segments, project.character_voice_map, project.narrator_voice)

            save_project(project)
            logger.info(f"Designed narrator voice '{voice_name}' for project '{project_id}'")
        except Exception as e:
            logger.error(f"Failed to design narrator voice: {e}")
            # Still return — the prompt was saved even if voice design failed

    return project_to_detail_response(project)


@router.put("/projects/{project_id}/narrator-voice-prompt", response_model=ProjectDetailResponse)
async def update_narrator_voice_prompt(project_id: str, request: UpdateNarratorVoicePromptRequest):
    """Update the narrator voice prompt text (manual edit)."""
    project = _get_project_or_404(project_id)
    project.narrator_voice_prompt = request.narrator_voice_prompt
    save_project(project)
    logger.info(f"Updated narrator voice prompt for project '{project_id}'")
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/design-narrator-voice", response_model=ProjectDetailResponse)
async def design_narrator_voice(project_id: str):
    """Redesign the narrator voice from the current narrator_voice_prompt via MOSS."""
    project = _get_project_or_404(project_id)

    if not project.narrator_voice_prompt:
        raise HTTPException(status_code=400, detail="No narrator voice prompt set. Generate one first or set it manually.")

    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service unavailable")

    if not hasattr(tts_service, 'design_voice'):
        raise HTTPException(status_code=503, detail="Voice design not supported by current TTS backend")

    try:
        voice_name = "narrator_voice"
        logger.info(f"Redesigning narrator voice with prompt: {project.narrator_voice_prompt[:100]}...")
        await asyncio.to_thread(tts_service.design_voice, voice_name, project.narrator_voice_prompt)

        project.narrator_voice = voice_name

        # Re-assign narrator voice to all non-character segments
        for ch in project.chapters:
            assign_segment_voices(ch.segments, project.character_voice_map, project.narrator_voice)

        save_project(project)
        logger.info(f"Redesigned narrator voice '{voice_name}' for project '{project_id}'")
        return project_to_detail_response(project)
    except Exception as e:
        logger.error(f"Narrator voice design failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/export-video")
async def export_video(project_id: str):
    """Start video assembly as a background task. Returns immediately with status='queued'."""
    from .video_assembler import build_project_video

    project = _get_project_or_404(project_id)

    # Collect all segments that have both audio and visuals
    segments_data = []
    for ch in project.chapters:
        for seg in ch.segments:
            if not seg.visual_path or not os.path.exists(seg.visual_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment '{seg.id}' is missing a visual. Generate all visuals first.",
                )
            if not seg.audio_path or not os.path.exists(seg.audio_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment '{seg.id}' is missing audio. Generate all audio first.",
                )
            segments_data.append((seg.visual_path, seg.audio_path, seg.visual_type or "image", seg.animation_style, seg.video_fill_mode))

    if not segments_data:
        raise HTTPException(status_code=400, detail="No segments with audio and visuals found")

    # Prevent concurrent exports for the same project
    if project_id not in _export_locks:
        _export_locks[project_id] = asyncio.Lock()
    lock = _export_locks[project_id]
    if lock.locked():
        raise HTTPException(status_code=409, detail="Export already in progress for this project")

    video_dir = os.path.join(get_project_dir(project_id), "video")

    # Update status immediately
    project.video_status = "generating"
    save_project(project)

    async def _run_assembly():
        async with lock:
            import shutil
            # Wipe old clips so stale files from previous/aborted runs are never reused
            clips_dir = os.path.join(video_dir, "clips")
            if os.path.exists(clips_dir):
                shutil.rmtree(clips_dir)
            proj = load_project(project_id)
            try:
                path = await asyncio.to_thread(build_project_video, segments_data, video_dir)
                proj.video_status = "done"
                proj.video_path = path
                logger.info(f"Video assembly complete: {path}")
            except Exception as e:
                proj.video_status = "error"
                proj.video_error = str(e)
                logger.error(f"Video assembly failed: {e}")
            finally:
                save_project(proj)

    asyncio.ensure_future(_run_assembly())
    return {"status": "queued", "message": "Video assembly started in background"}


@router.get("/projects/{project_id}/export-status")
async def export_status(project_id: str):
    """Poll video assembly status: queued | generating | done | error."""
    project = _get_project_or_404(project_id)
    status = getattr(project, "video_status", "idle")
    error = getattr(project, "video_error", None)
    video_path = getattr(project, "video_path", None)

    video_ready = False
    if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        video_ready = True

    return {
        "status": status,
        "video_ready": video_ready,
        "error": error if status == "error" else None,
    }



@router.get("/projects/{project_id}/video")
async def download_video(project_id: str):
    """Download the assembled video file."""
    from fastapi.responses import FileResponse

    project = _get_project_or_404(project_id)
    video_path = os.path.join(get_project_dir(project_id), "video", "full_book.mp4")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found. Export the video first.")

    filename = project.name.replace(" ", "_")[:50] or "audiobook"
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{filename}.mp4",
    )


@router.get("/comfyui/health")
async def comfyui_health():
    """Check if ComfyUI is reachable."""
    from .comfyui_service import health_check, COMFYUI_API_URL
    is_healthy = await health_check()
    return {
        "healthy": is_healthy,
        "url": COMFYUI_API_URL,
    }


@router.get("/queue-status")
async def queue_status():
    """Return the current depth and active job for each generation queue."""
    from .generation_queue import get_queue_status
    return get_queue_status()

