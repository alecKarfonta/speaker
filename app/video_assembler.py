"""
Video assembler — combines segment visuals + audio into a final MP4 using ffmpeg.
"""

import os
import subprocess
import logging
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run_ffmpeg(args: List[str], description: str = "ffmpeg") -> None:
    """Run an ffmpeg command and raise on failure."""
    cmd = ["ffmpeg", "-y"] + args
    logger.info(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"{description} failed: {result.stderr}")
        raise RuntimeError(f"{description} failed: {result.stderr[:500]}")


def create_segment_clip(
    visual_path: str,
    audio_path: str,
    output_path: str,
    visual_type: str = "image",
) -> str:
    """
    Create a single segment clip by combining a visual and audio file.

    For images: loops the still image for the duration of the audio.
    For videos: uses the video directly (may be shorter/longer than audio).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if visual_type == "image":
        # Ken Burns: gentle zoom + pan over the duration of the audio.
        # We probe audio duration first so we know how many frames to render.
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0
        fps = 25
        total_frames = int(duration * fps) + 1

        # Randomise direction: 0 = slow zoom-in, 1 = slow zoom-out
        import random
        direction = random.choice([0, 1])
        if direction == 0:
            # Zoom in from 1.0 → 1.15 over the clip
            zoom_expr = f"min(1+0.15*on/{total_frames},1.15)"
        else:
            # Zoom out from 1.15 → 1.0 over the clip
            zoom_expr = f"max(1.15-0.15*on/{total_frames},1.0)"

        # zoompan: z=zoom, d=total frames, s=output size, fps
        _run_ffmpeg([
            "-i", visual_path,
            "-i", audio_path,
            "-filter_complex",
            f"[0:v]scale=1920:1080,zoompan=z='{zoom_expr}':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={fps}[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ], f"segment clip (Ken Burns): {os.path.basename(output_path)}")
    else:
        # Video + audio → muxed video
        _run_ffmpeg([
            "-i", visual_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-shortest",
            output_path,
        ], f"segment clip (video): {os.path.basename(output_path)}")

    return output_path


def assemble_chapter(
    segment_clips: List[str],
    output_path: str,
) -> str:
    """Concatenate segment clips into a chapter video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in segment_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")
        concat_file = f.name

    try:
        _run_ffmpeg([
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ], f"chapter assembly: {os.path.basename(output_path)}")
    finally:
        os.unlink(concat_file)

    return output_path


def assemble_full_book(
    chapter_videos: List[str],
    output_path: str,
) -> str:
    """Concatenate chapter videos into a full book video."""
    return assemble_chapter(chapter_videos, output_path)  # Same concat logic


def build_project_video(
    segments: List[Tuple[str, str, str]],
    output_dir: str,
) -> str:
    """
    Build the full project video from segment data.

    Args:
        segments: List of (visual_path, audio_path, visual_type) tuples in order
        output_dir: Directory to write intermediate and final files

    Returns:
        Path to the final MP4
    """
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []
    for i, (visual_path, audio_path, visual_type) in enumerate(segments):
        clip_path = os.path.join(clips_dir, f"segment_{i:04d}.mp4")
        create_segment_clip(visual_path, audio_path, clip_path, visual_type)
        clip_paths.append(clip_path)

    final_path = os.path.join(output_dir, "full_book.mp4")
    assemble_chapter(clip_paths, final_path)

    logger.info(f"Built project video: {final_path} ({len(clip_paths)} segments)")
    return final_path
