from pydantic import BaseModel
"""
Audiobook API router — all /audiobook/* endpoints.
Handles project CRUD, segment generation, and audio export.
"""

import io
import os
import re
import logging
import numpy as np
import soundfile as sf
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, status, Response
from datetime import datetime

from .audiobook_models import (
    AudiobookProject, Chapter, Segment, SegmentStatus,
    CreateProjectRequest, UpdateCharacterMapRequest, UpdateSegmentRequest,
    SplitSegmentRequest, ReparseRequest, ProjectSummary, ProjectDetailResponse,
    save_project, load_project, list_projects, delete_project_from_disk,
    project_to_detail_response, get_project_dir,
)
from .audiobook_parser import (
    parse_book_text, detect_characters, assign_segment_voices,
)
from .audiobook_llm import analyze_characters as llm_analyze_characters, check_llm_available
from .audiobook_parsers import parse_uploaded_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger("speaker.audiobook")

router = APIRouter(prefix="/audiobook", tags=["Audiobook"])

# Will be set by main.py on startup
tts_service = None


def set_tts_service(service):
    """Called from main.py to inject the TTS service."""
    global tts_service
    tts_service = service


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
    """Edit a segment's text or voice."""
    project = _get_project_or_404(project_id)
    _, _, seg = _find_segment(project, segment_id)
    
    if request.text is not None:
        seg.text = request.text
        seg.status = SegmentStatus.PENDING  # Reset status when text changes
        seg.audio_path = None
        seg.duration = None
    
    if request.voice_name is not None:
        seg.voice_name = request.voice_name
    
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
    """Generate (or regenerate) audio for a single segment."""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    project = _get_project_or_404(project_id)
    ch_idx, seg_idx, seg = _find_segment(project, segment_id)
    
    voice = seg.voice_name or project.narrator_voice
    if not voice:
        raise HTTPException(status_code=400, detail="No voice assigned to segment and no narrator voice set")
    
    # Validate voice exists
    available_voices = tts_service.get_voices()
    if voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available: {', '.join(available_voices)}"
        )
    
    seg.status = SegmentStatus.GENERATING
    save_project(project)
    
    try:
        # Auto-chunk if text exceeds TTS max length
        from .config import settings
        max_len = settings.max_text_length
        text = seg.text.strip()
        
        if len(text) <= max_len:
            chunks = [text]
        else:
            chunks = _split_text_into_chunks(text, max_len)
            logger.info(f"Auto-split segment '{segment_id}' ({len(text)} chars) into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        audio_parts = []
        sample_rate = None
        for i, chunk in enumerate(chunks):
            chunk_audio, sr = tts_service.generate_speech(
                text=chunk,
                voice_name=voice,
                language="en",
            )
            if chunk_audio is None or len(chunk_audio) == 0:
                raise RuntimeError(f"TTS returned empty audio for chunk {i+1}/{len(chunks)}")
            audio_parts.append(chunk_audio)
            sample_rate = sr
        
        # Concatenate chunks with a brief silence gap between them
        if len(audio_parts) == 1:
            audio = audio_parts[0]
        else:
            gap = np.zeros(int(sample_rate * 0.3), dtype=audio_parts[0].dtype)  # 300ms gap
            combined = []
            for i, part in enumerate(audio_parts):
                combined.append(part)
                if i < len(audio_parts) - 1:
                    combined.append(gap)
            audio = np.concatenate(combined)
        
        # Save audio to disk
        project_dir = get_project_dir(project_id)
        audio_dir = os.path.join(project_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_path = os.path.join(audio_dir, f"{segment_id}.wav")
        sf.write(audio_path, audio, sample_rate, format="WAV", subtype="PCM_16")
        
        seg.status = SegmentStatus.DONE
        seg.audio_path = audio_path
        seg.duration = len(audio) / sample_rate
        seg.error_message = None
        
        save_project(project)
        logger.info(f"Generated segment '{segment_id}' ({seg.duration:.1f}s, {len(chunks)} chunk(s))")
        
        return {
            "status": "done",
            "duration": seg.duration,
            "segment_id": segment_id,
        }
    
    except Exception as e:
        seg.status = SegmentStatus.ERROR
        seg.error_message = str(e)
        save_project(project)
        logger.error(f"Failed segment '{segment_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


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
    """Generate all pending segments across all chapters."""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    project = _get_project_or_404(project_id)
    results = {"generated": 0, "errors": 0, "skipped": 0}
    
    for ch in project.chapters:
        for seg in ch.segments:
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


# ============================================================
# Visual Generation Endpoints
# ============================================================

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
async def generate_segment_visual(project_id: str, segment_id: str, mode: str = None):
    """Generate a visual (image or video) for a single segment using ComfyUI."""
    from .comfyui_service import generate_visual as comfyui_generate, COMFYUI_VISUAL_MODE
    from .audiobook_llm import generate_scene_prompt, generate_visual_style

    project = _get_project_or_404(project_id)

    # Auto-generate visual style if missing
    if not project.visual_style and project.raw_text:
        style = generate_visual_style(project.raw_text)
        if style:
            project.visual_style = style
            save_project(project)
            logger.info(f"Auto-generated visual style: {style}")

    # Find the segment and its neighbours
    segment, chapter = None, None
    all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]
    seg_idx = None
    for i, (ch, seg) in enumerate(all_segments):
        if seg.id == segment_id:
            segment = seg
            chapter = ch
            seg_idx = i
            break

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Get context from adjacent segments
    prev_scene = all_segments[seg_idx - 1][1].scene_prompt if seg_idx > 0 else ""
    next_text = all_segments[seg_idx + 1][1].text if seg_idx < len(all_segments) - 1 else ""

    # Step 1: Generate scene prompt via LLM if we don't have one
    if not segment.scene_prompt:
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
        save_project(project)

    # Step 2: Generate visual via ComfyUI
    segment.visual_status = "generating"
    save_project(project)

    try:
        visual_dir = os.path.join(get_project_dir(project_id), "visuals")
        # Find character portrait reference for this segment
        ref_image = _find_character_portrait(project, segment)
        path, vtype = await comfyui_generate(
            prompt=segment.scene_prompt,
            output_dir=visual_dir,
            prefix=f"seg_{segment.id}",
            mode=mode or COMFYUI_VISUAL_MODE,
            duration=segment.duration,
            ref_image=ref_image,
        )
        segment.visual_path = path
        segment.visual_type = vtype
        segment.visual_status = "done"
        logger.info(f"Generated {vtype} for segment {segment_id}: {path}")
    except Exception as e:
        segment.visual_status = "error"
        logger.error(f"Visual generation failed for segment {segment_id}: {e}")
        save_project(project)
        raise HTTPException(status_code=500, detail=f"ComfyUI generation failed: {str(e)}")

    save_project(project)
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/generate-visuals", response_model=ProjectDetailResponse)
async def generate_all_visuals(project_id: str, mode: str = None):
    """Generate visuals for all segments that don't already have one."""
    from .comfyui_service import generate_visual as comfyui_generate, COMFYUI_VISUAL_MODE
    from .audiobook_llm import generate_scene_prompt, generate_visual_style

    project = _get_project_or_404(project_id)
    visual_dir = os.path.join(get_project_dir(project_id), "visuals")
    visual_mode = mode or COMFYUI_VISUAL_MODE

    # Auto-generate visual style if missing
    if not project.visual_style and project.raw_text:
        style = generate_visual_style(project.raw_text)
        if style:
            project.visual_style = style
            save_project(project)
            logger.info(f"Auto-generated visual style: {style}")

    # Build flat list of all segments for context access
    all_segments = [(ch, seg) for ch in project.chapters for seg in ch.segments]

    generated = 0
    errors = 0

    for i, (ch, seg) in enumerate(all_segments):
        if seg.visual_status == "done":
            continue  # Skip already-generated

        # Context from adjacent segments
        prev_scene = all_segments[i - 1][1].scene_prompt if i > 0 else ""
        next_text = all_segments[i + 1][1].text if i < len(all_segments) - 1 else ""

        # Generate scene prompt with style + context
        if not seg.scene_prompt:
            scene = generate_scene_prompt(
                text=seg.text,
                chapter_title=ch.title,
                character=seg.character or "Narrator",
                emotion=seg.emotion or "neutral",
                visual_style=project.visual_style,
                prev_scene=prev_scene or "",
                next_text=next_text or "",
            )
            if not scene:
                seg.visual_status = "error"
                errors += 1
                continue
            seg.scene_prompt = scene

        # Generate visual
        seg.visual_status = "generating"
        save_project(project)

        try:
            ref_image = _find_character_portrait(project, seg)
            path, vtype = await comfyui_generate(
                prompt=seg.scene_prompt,
                output_dir=visual_dir,
                prefix=f"seg_{seg.id}",
                mode=visual_mode,
                duration=seg.duration,
                ref_image=ref_image,
            )
            seg.visual_path = path
            seg.visual_type = vtype
            seg.visual_status = "done"
            generated += 1
        except Exception as e:
            seg.visual_status = "error"
            errors += 1
            logger.error(f"Visual generation failed for segment {seg.id}: {e}")

        save_project(project)

    logger.info(f"Batch visual generation: {generated} generated, {errors} errors")
    return project_to_detail_response(project)


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
    for chunk in chunks:
        chars = extract_characters(chunk)
        if chars:
            all_characters_data.extend(chars)

    if not all_characters_data:
        raise HTTPException(status_code=500, detail="LLM failed to extract characters")

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
            # Update description if it is longer/more detailed than existing, 
            # or just overwrite (current logic was overwrite)
            new_chars_dict[name_lower].description = char_data["description"]
        else:
            new_chars_dict[name_lower] = CharacterRef(
                name=name,
                description=char_data["description"],
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

        # Generate portrait prompt from description
        portrait_prompt = generate_portrait_prompt(char_ref.name, char_ref.description)
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
async def update_character(project_id: str, character_name: str, description: str = None):
    """Update a character's description. Set description to regenerate portrait next time."""
    project = _get_project_or_404(project_id)

    char_ref = None
    for c in project.characters:
        if c.name.lower() == character_name.lower():
            char_ref = c
            break

    if not char_ref:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    if description:
        char_ref.description = description
        # Clear portrait so it gets regenerated
        char_ref.portrait_path = None
        char_ref.portrait_comfyui = None

    save_project(project)
    return project_to_detail_response(project)


@router.post("/projects/{project_id}/export-video")
async def export_video(project_id: str):
    """Assemble all segment visuals + audio into a final MP4 video."""
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
            segments_data.append((seg.visual_path, seg.audio_path, seg.visual_type or "image"))

    if not segments_data:
        raise HTTPException(status_code=400, detail="No segments with audio and visuals found")

    video_dir = os.path.join(get_project_dir(project_id), "video")

    try:
        video_path = build_project_video(segments_data, video_dir)
    except Exception as e:
        logger.error(f"Video assembly failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video assembly failed: {str(e)}")

    return {"status": "ok", "video_path": video_path, "message": "Video assembled successfully"}


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
