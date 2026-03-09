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


class VisualAsset(BaseModel):
    """Independent visual asset that can be shared across multiple segments."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""  # User-friendly name (auto-populated from scene_prompt if empty)
    scene_prompt: Optional[str] = None
    visual_path: Optional[str] = None
    visual_type: Optional[str] = None  # "image" or "video"
    visual_mode: Optional[str] = None  # image | scene_image | video | ref_video | scene_video
    visual_status: str = "none"  # none | pending | queued | generating | done | error
    animation_style: Optional[str] = None  # zoom_in | zoom_out | pan_left | pan_right | pan_up | static | random | None
    video_fill_mode: str = "hold"  # loop | hold | fade
    ref_character: Optional[str] = None  # Character name for portrait reference
    # Generation settings (persisted so UI can repopulate after refresh)
    gen_frames: Optional[int] = None
    gen_fps: Optional[int] = None
    gen_width: Optional[int] = None
    gen_height: Optional[int] = None
    gen_enable_audio: bool = False
    gen_two_stage: bool = False
    gen_candidates: int = 1  # how many seed variants to generate
    candidate_paths: List[str] = []  # paths to all candidate outputs
    selected_candidate: int = -1  # index of chosen candidate (-1 = auto-first)
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    # Visual — linked to a VisualAsset
    visual_id: Optional[str] = None  # FK to VisualAsset.id
    # Legacy visual fields (kept for migration, populated from VisualAsset on serialization)
    scene_prompt: Optional[str] = None
    visual_path: Optional[str] = None
    visual_type: Optional[str] = None  # "image" or "video"
    visual_mode: Optional[str] = None  # image | image_ref | faceid_image | video | faceid_video
    visual_status: str = "none"  # none | pending | generating | done | error
    animation_style: Optional[str] = None  # zoom_in | zoom_out | pan_left | pan_right | pan_up | static | random | None
    video_fill_mode: str = "hold"  # loop | hold (last frame → fade) | fade (immediate fade to black)


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
    portrait_prompt: Optional[str] = None  # Image generation prompt for portrait (SD/ComfyUI style)
    voice_prompt: Optional[str] = None  # Voice description for MOSS voice generator
    visual_profile: Optional[Dict[str, str]] = None  # face, skin, build, clothing
    portrait_path: Optional[str] = None  # Local path to hero portrait image
    portrait_variants: List[str] = []  # Paths to all portrait variant images
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
    narrator_voice_prompt: str = ""  # Voice description for narrator (used by MOSS voice designer)
    detected_characters: List[str] = []
    character_descriptions: Dict[str, str] = {}  # AI-generated character descriptions
    visual_style: str = ""  # Persistent visual style prompt for scene continuity
    characters: List[CharacterRef] = []  # Character references with portraits
    visuals: List[VisualAsset] = []  # Independent visual assets
    # Video export status
    video_status: str = "idle"  # idle | generating | done | error
    video_path: str = ""
    video_error: str = ""

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
    scene_prompt: Optional[str] = None  # Editable visual generation prompt


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


class VisualAssetResponse(BaseModel):
    """API response for a visual asset."""
    id: str
    label: str = ""
    scene_prompt: Optional[str] = None
    has_visual: bool = False
    visual_type: Optional[str] = None
    visual_path: Optional[str] = None
    visual_mode: Optional[str] = None
    visual_status: str = "none"
    animation_style: Optional[str] = None
    video_fill_mode: str = "hold"
    ref_character: Optional[str] = None
    gen_frames: Optional[int] = None
    gen_fps: Optional[int] = None
    gen_width: Optional[int] = None
    gen_height: Optional[int] = None
    gen_enable_audio: bool = False
    gen_two_stage: bool = False
    gen_candidates: int = 1
    candidate_paths: List[str] = []
    candidate_count: int = 0
    selected_candidate: int = -1
    created_at: Optional[datetime] = None
    assigned_segments: int = 0  # how many segments use this visual


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
    visual_path: Optional[str] = None
    visual_mode: Optional[str] = None
    visual_status: str = "none"
    animation_style: Optional[str] = None
    video_fill_mode: str = "hold"
    visual_id: Optional[str] = None  # linked VisualAsset id


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
    narrator_voice_prompt: str = ""  # editable narrator voice description
    visuals: List[VisualAssetResponse] = []  # independent visual assets


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


def _migrate_inline_visuals(project: AudiobookProject) -> bool:
    """Migrate legacy inline visual fields on segments to VisualAsset entities.
    Returns True if any migration occurred."""
    dirty = False
    # Build lookup of existing visual_ids for dedup
    existing_ids = {v.id for v in project.visuals}
    for ch in project.chapters:
        for seg in ch.segments:
            # Only migrate if segment has visual data but no visual_id link
            if seg.visual_path and not seg.visual_id:
                asset = VisualAsset(
                    label=(seg.scene_prompt or "")[:40].strip() or f"Visual for seg {seg.id}",
                    scene_prompt=seg.scene_prompt,
                    visual_path=seg.visual_path,
                    visual_type=seg.visual_type,
                    visual_mode=seg.visual_mode,
                    visual_status=seg.visual_status,
                    animation_style=seg.animation_style,
                    video_fill_mode=seg.video_fill_mode,
                )
                if asset.id not in existing_ids:
                    project.visuals.append(asset)
                    existing_ids.add(asset.id)
                seg.visual_id = asset.id
                dirty = True
    return dirty


def _sync_visual_assets(project: AudiobookProject) -> bool:
    """Sync VisualAsset records from segment data when the VA is missing data.
    Only populates VA fields that are empty — never overwrites existing VA paths,
    since shared VAs would be clobbered by stale segment data on load."""
    dirty = False
    va_lookup = {v.id: v for v in project.visuals}
    for ch in project.chapters:
        for seg in ch.segments:
            if not seg.visual_id or seg.visual_id not in va_lookup:
                continue
            va = va_lookup[seg.visual_id]
            # Only populate VA path if it has NONE — never overwrite existing
            if seg.visual_path and not va.visual_path:
                va.visual_path = seg.visual_path
                va.visual_type = seg.visual_type
                if seg.visual_status == "done":
                    va.visual_status = "done"
                dirty = True
            # If VA is stuck queued/generating but segment is done, fix VA status
            elif va.visual_status in ("queued", "generating") and seg.visual_status == "done":
                va.visual_status = "done"
                dirty = True
    return dirty


def load_project(project_id: str) -> Optional[AudiobookProject]:
    """Load project from disk. Recovers stuck 'generating' states from interrupted requests."""
    path = get_project_json_path(project_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        project = AudiobookProject.model_validate_json(f.read())

    # Recover stuck states — if the server restarted mid-generation,
    # segments may be permanently stuck in "generating".
    dirty = False
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_status == "generating":
                seg.visual_status = "none" if not seg.scene_prompt else "pending"
                dirty = True
            if seg.status == SegmentStatus.GENERATING:
                seg.status = SegmentStatus.PENDING
                dirty = True

    # Recover stuck visual assets
    for va in project.visuals:
        if va.visual_status in ("generating", "queued"):
            va.visual_status = "none" if not va.scene_prompt else "pending"
            dirty = True

    # Migrate legacy inline visuals to VisualAsset entities
    if _migrate_inline_visuals(project):
        dirty = True

    # Sync stale VisualAsset data from segments
    if _sync_visual_assets(project):
        dirty = True

    if dirty:
        save_project(project)

    return project


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
    """Soft-delete a project by renaming its directory (recoverable)."""
    import time
    project_dir = get_project_dir(project_id)
    if os.path.exists(project_dir):
        trash_dir = project_dir + f"_deleted_{int(time.time())}"
        os.rename(project_dir, trash_dir)
        return True
    return False


def _resolve_visual(project: AudiobookProject, visual_id: Optional[str]) -> Optional[VisualAsset]:
    """Find a VisualAsset by id in the project."""
    if not visual_id:
        return None
    for va in project.visuals:
        if va.id == visual_id:
            return va
    return None


def project_to_detail_response(project: AudiobookProject) -> ProjectDetailResponse:
    """Convert a project to its API detail response."""
    # Pre-compute segment count per visual for assignment display
    visual_seg_counts: Dict[str, int] = {}
    for ch in project.chapters:
        for seg in ch.segments:
            if seg.visual_id:
                visual_seg_counts[seg.visual_id] = visual_seg_counts.get(seg.visual_id, 0) + 1

    chapters = []
    for ch in project.chapters:
        segments = []
        for seg in ch.segments:
            # Resolve visual data from the linked VisualAsset (or fallback to legacy inline)
            va = _resolve_visual(project, seg.visual_id)
            if va:
                vpath = va.visual_path
                has_vis = vpath is not None and os.path.exists(vpath) if vpath else False
                seg_resp = SegmentResponse(
                    id=seg.id, text=seg.text, voice_name=seg.voice_name,
                    character=seg.character, emotion=seg.emotion, status=seg.status,
                    duration=seg.duration, error_message=seg.error_message,
                    scene_prompt=va.scene_prompt,
                    has_visual=has_vis,
                    visual_path=vpath,
                    visual_type=va.visual_type, visual_mode=va.visual_mode,
                    visual_status=va.visual_status,
                    animation_style=va.animation_style, video_fill_mode=va.video_fill_mode,
                    has_audio=seg.audio_path is not None and os.path.exists(seg.audio_path) if seg.audio_path else False,
                    visual_id=seg.visual_id,
                )
            else:
                seg_resp = SegmentResponse(
                    id=seg.id, text=seg.text, voice_name=seg.voice_name,
                    character=seg.character, emotion=seg.emotion, status=seg.status,
                    duration=seg.duration, error_message=seg.error_message,
                    scene_prompt=seg.scene_prompt,
                    has_visual=seg.visual_path is not None and os.path.exists(seg.visual_path) if seg.visual_path else False,
                    visual_path=seg.visual_path,
                    visual_type=seg.visual_type, visual_mode=seg.visual_mode,
                    visual_status=seg.visual_status,
                    animation_style=seg.animation_style, video_fill_mode=seg.video_fill_mode,
                    has_audio=seg.audio_path is not None and os.path.exists(seg.audio_path) if seg.audio_path else False,
                    visual_id=seg.visual_id,
                )
            segments.append(seg_resp)
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
    # Count segments that have a visual done (via asset or legacy)
    vis_ready = 0
    for s in all_segs:
        va = _resolve_visual(project, s.visual_id)
        if va and va.visual_status == "done":
            vis_ready += 1
        elif not s.visual_id and s.visual_status == "done":
            vis_ready += 1

    # Build visual asset responses
    visual_responses = [
        VisualAssetResponse(
            id=va.id,
            label=va.label,
            scene_prompt=va.scene_prompt,
            has_visual=va.visual_path is not None and os.path.exists(va.visual_path) if va.visual_path else False,
            visual_path=va.visual_path,
            visual_type=va.visual_type,
            visual_mode=va.visual_mode,
            visual_status=va.visual_status,
            animation_style=va.animation_style,
            video_fill_mode=va.video_fill_mode,
            ref_character=va.ref_character,
            gen_frames=va.gen_frames,
            gen_fps=va.gen_fps,
            gen_width=va.gen_width,
            gen_height=va.gen_height,
            gen_enable_audio=va.gen_enable_audio,
            gen_two_stage=va.gen_two_stage,
            gen_candidates=va.gen_candidates,
            candidate_paths=va.candidate_paths,
            candidate_count=len(va.candidate_paths),
            selected_candidate=va.selected_candidate,
            created_at=va.created_at,
            assigned_segments=visual_seg_counts.get(va.id, 0),
        )
        for va in project.visuals
    ]

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
        narrator_voice_prompt=project.narrator_voice_prompt,
        visuals=visual_responses,
    )
