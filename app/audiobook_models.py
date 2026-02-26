"""
Data models for the audiobook generator system.
Handles project persistence, chapter/segment management, and character-voice mapping.
"""

import json
import uuid
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SegmentStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    ERROR = "error"


class Segment(BaseModel):
    """Individual TTS unit — a chunk of text to be spoken by one voice."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    voice_name: Optional[str] = None
    character: Optional[str] = None  # None = narrator
    emotion: Optional[str] = None  # AI-detected emotion tag
    status: SegmentStatus = SegmentStatus.PENDING
    audio_path: Optional[str] = None
    duration: Optional[float] = None  # seconds
    error_message: Optional[str] = None
    # Visual generation fields
    scene_prompt: Optional[str] = None
    visual_path: Optional[str] = None
    visual_type: Optional[str] = None  # "image" or "video"
    visual_status: str = "none"  # none | pending | generating | done | error


class Chapter(BaseModel):
    """A chapter containing ordered segments."""
    index: int
    title: str = ""
    segments: List[Segment] = []

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def done_segments(self) -> int:
        return sum(1 for s in self.segments if s.status == SegmentStatus.DONE)

    @property
    def progress(self) -> float:
        if not self.segments:
            return 0.0
        return self.done_segments / self.total_segments


class CharacterVoiceMapping(BaseModel):
    """Maps character names to voice names."""
    character: str
    voice_name: str


class CharacterRef(BaseModel):
    """Character reference with appearance description and portrait for visual consistency."""
    name: str
    description: str = ""  # Detailed physical appearance description
    portrait_path: Optional[str] = None  # Local path to hero portrait image
    portrait_comfyui: Optional[str] = None  # Filename on ComfyUI input directory


class AudiobookProject(BaseModel):
    """Top-level audiobook project."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    raw_text: str = ""
    chapter_pattern: str = "auto"  # regex or "auto"
    chapters: List[Chapter] = []
    character_voice_map: Dict[str, str] = {}  # character_name -> voice_name
    narrator_voice: str = ""
    detected_characters: List[str] = []
    character_descriptions: Dict[str, str] = {}  # AI-generated character descriptions
    visual_style: str = ""  # Persistent visual style prompt for scene continuity
    characters: List[CharacterRef] = []  # Character references with portraits

    @property
    def total_segments(self) -> int:
        return sum(ch.total_segments for ch in self.chapters)

    @property
    def done_segments(self) -> int:
        return sum(ch.done_segments for ch in self.chapters)

    @property
    def progress(self) -> float:
        total = self.total_segments
        if total == 0:
            return 0.0
        return self.done_segments / total


# --- Request / Response models for the API ---

class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    chapter_pattern: str = Field(default="auto")
    narrator_voice: Optional[str] = None


class UpdateCharacterMapRequest(BaseModel):
    character_voice_map: Dict[str, str]
    narrator_voice: Optional[str] = None


class UpdateSegmentRequest(BaseModel):
    text: Optional[str] = None
    voice_name: Optional[str] = None


class ReparseRequest(BaseModel):
    chapter_pattern: str = "auto"


class SplitSegmentRequest(BaseModel):
    split_at: Optional[int] = None  # Character index; None = auto-split at nearest sentence boundary


class ProjectSummary(BaseModel):
    """Lightweight project info for listing."""
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    total_chapters: int
    total_segments: int
    done_segments: int
    progress: float
    detected_characters: List[str]


class SegmentResponse(BaseModel):
    id: str
    text: str
    voice_name: Optional[str]
    character: Optional[str]
    emotion: Optional[str]
    status: SegmentStatus
    duration: Optional[float]
    error_message: Optional[str]
    has_audio: bool
    scene_prompt: Optional[str] = None
    has_visual: bool = False
    visual_type: Optional[str] = None
    visual_status: str = "none"


class ChapterResponse(BaseModel):
    index: int
    title: str
    segments: List[SegmentResponse]
    total_segments: int
    done_segments: int
    progress: float


class ProjectDetailResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    chapters: List[ChapterResponse]
    character_voice_map: Dict[str, str]
    narrator_voice: str
    detected_characters: List[str]
    character_descriptions: Dict[str, str]
    total_segments: int
    done_segments: int
    progress: float
    # Stats
    total_duration: float = 0.0  # total audio seconds
    error_segments: int = 0
    total_characters: int = 0  # total text chars across all segments
    visual_ready: int = 0  # segments with visuals generated
    visual_style: str = ""  # project visual style for continuity
    characters: List[dict] = []  # character references with portraits


# --- Persistence helpers ---

AUDIOBOOKS_DIR = os.environ.get("AUDIOBOOKS_DIR", "data/audiobooks")


def get_project_dir(project_id: str) -> str:
    return os.path.join(AUDIOBOOKS_DIR, project_id)


def get_project_json_path(project_id: str) -> str:
    return os.path.join(get_project_dir(project_id), "project.json")


def save_project(project: AudiobookProject) -> None:
    """Save project state to disk."""
    project.updated_at = datetime.utcnow()
    project_dir = get_project_dir(project.id)
    os.makedirs(project_dir, exist_ok=True)
    with open(get_project_json_path(project.id), "w") as f:
        f.write(project.model_dump_json(indent=2))


def load_project(project_id: str) -> Optional[AudiobookProject]:
    """Load project from disk."""
    path = get_project_json_path(project_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return AudiobookProject.model_validate_json(f.read())


def list_projects() -> List[ProjectSummary]:
    """List all saved projects."""
    summaries = []
    if not os.path.exists(AUDIOBOOKS_DIR):
        return summaries
    for entry in os.listdir(AUDIOBOOKS_DIR):
        project_dir = os.path.join(AUDIOBOOKS_DIR, entry)
        json_path = os.path.join(project_dir, "project.json")
        if os.path.isdir(project_dir) and os.path.exists(json_path):
            try:
                project = load_project(entry)
                if project:
                    summaries.append(ProjectSummary(
                        id=project.id,
                        name=project.name,
                        created_at=project.created_at,
                        updated_at=project.updated_at,
                        total_chapters=len(project.chapters),
                        total_segments=project.total_segments,
                        done_segments=project.done_segments,
                        progress=project.progress,
                        detected_characters=project.detected_characters,
                    ))
            except Exception:
                continue
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def delete_project_from_disk(project_id: str) -> bool:
    """Delete a project directory entirely."""
    import shutil
    project_dir = get_project_dir(project_id)
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
        return True
    return False


def project_to_detail_response(project: AudiobookProject) -> ProjectDetailResponse:
    """Convert a project to its API detail response."""
    chapters = []
    for ch in project.chapters:
        segments = [
            SegmentResponse(
                id=seg.id,
                text=seg.text,
                voice_name=seg.voice_name,
                character=seg.character,
                emotion=seg.emotion,
                status=seg.status,
                duration=seg.duration,
                error_message=seg.error_message,
                scene_prompt=seg.scene_prompt,
                has_visual=seg.visual_path is not None and os.path.exists(seg.visual_path) if seg.visual_path else False,
                visual_type=seg.visual_type,
                visual_status=seg.visual_status,
                has_audio=seg.audio_path is not None and os.path.exists(seg.audio_path) if seg.audio_path else False,
            )
            for seg in ch.segments
        ]
        chapters.append(ChapterResponse(
            index=ch.index,
            title=ch.title,
            segments=segments,
            total_segments=ch.total_segments,
            done_segments=ch.done_segments,
            progress=ch.progress,
        ))
    # Compute stats
    all_segs = [seg for ch in project.chapters for seg in ch.segments]
    total_duration = sum(s.duration or 0.0 for s in all_segs)
    error_count = sum(1 for s in all_segs if s.status == SegmentStatus.ERROR)
    total_chars = sum(len(s.text) for s in all_segs)
    vis_ready = sum(1 for s in all_segs if s.visual_status == "done")

    return ProjectDetailResponse(
        id=project.id,
        name=project.name,
        created_at=project.created_at,
        updated_at=project.updated_at,
        chapters=chapters,
        character_voice_map=project.character_voice_map,
        narrator_voice=project.narrator_voice,
        character_descriptions=project.character_descriptions,
        detected_characters=project.detected_characters,
        total_segments=project.total_segments,
        done_segments=project.done_segments,
        progress=project.progress,
        total_duration=total_duration,
        error_segments=error_count,
        total_characters=total_chars,
        visual_ready=vis_ready,
        visual_style=project.visual_style,
        characters=[c.model_dump() for c in project.characters],
    )
