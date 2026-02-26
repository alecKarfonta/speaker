"""
Video assembler — combines segment visuals + audio into a final MP4 using ffmpeg.

Handles both still images (Ken Burns effect) and video clips (LTX-2 WebP).
Adds crossfade transitions between segments for a polished result.
"""

import os
import subprocess
import logging
import tempfile
import random
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Crossfade duration in seconds between segments
CROSSFADE_DURATION = 0.5


def _run_ffmpeg(args: List[str], description: str = "ffmpeg") -> None:
    """Run an ffmpeg command and raise on failure."""
    cmd = ["ffmpeg", "-y"] + args
    logger.info(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"{description} failed: {result.stderr}")
        raise RuntimeError(f"{description} failed: {result.stderr[:500]}")


def _probe_duration(path: str) -> float:
    """Probe the duration of a media file (audio or video) in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 10.0  # Fallback


def create_segment_clip(
    visual_path: str,
    audio_path: str,
    output_path: str,
    visual_type: str = "image",
) -> str:
    """
    Create a single segment clip by combining a visual and audio file.

    For images: loops the still image with Ken Burns zoom for audio duration.
    For videos: loops/trims the video to match audio duration exactly.
    Both: output normalized to 1920x1080, with brief fade in/out.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_dur = _probe_duration(audio_path)

    if visual_type == "image":
        _create_image_clip(visual_path, audio_path, output_path, audio_dur)
    else:
        _create_video_clip(visual_path, audio_path, output_path, audio_dur)

    return output_path


def _create_image_clip(
    visual_path: str, audio_path: str, output_path: str, audio_dur: float
) -> None:
    """Ken Burns zoom/pan over a still image for the audio duration."""
    fps = 25
    total_frames = int(audio_dur * fps) + 1

    # Randomise direction: zoom-in or zoom-out
    direction = random.choice([0, 1])
    if direction == 0:
        zoom_expr = f"min(1+0.15*on/{total_frames},1.15)"
    else:
        zoom_expr = f"max(1.15-0.15*on/{total_frames},1.0)"

    _run_ffmpeg([
        "-i", visual_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:v]scale=1920:1080,zoompan=z='{zoom_expr}':d={total_frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={fps},"
        f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(0, audio_dur - 0.3):.2f}:d=0.3[v]",
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


def _create_video_clip(
    visual_path: str, audio_path: str, output_path: str, audio_dur: float
) -> None:
    """
    Video clip synced to audio duration.
    If video is shorter than audio, loop it. If longer, trim it.
    Output scaled to 1920x1080 with fade in/out.
    """
    video_dur = _probe_duration(visual_path)
    fade_d = 0.3

    # Build input args
    input_args = []
    if video_dur > 0 and video_dur < audio_dur:
        # Loop video to cover audio duration
        loops_needed = math.ceil(audio_dur / video_dur)
        input_args = ["-stream_loop", str(loops_needed - 1), "-i", visual_path]
    else:
        input_args = ["-i", visual_path]

    input_args += ["-i", audio_path]

    # Video filter: scale to 1920x1080, fade in/out
    vf = (
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        f"fade=t=in:st=0:d={fade_d},"
        f"fade=t=out:st={max(0, audio_dur - fade_d):.2f}:d={fade_d}"
    )

    _run_ffmpeg(
        input_args + [
            "-t", f"{audio_dur:.2f}",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        f"segment clip (video): {os.path.basename(output_path)}",
    )


def assemble_chapter(
    segment_clips: List[str],
    output_path: str,
    crossfade: float = CROSSFADE_DURATION,
) -> str:
    """
    Concatenate segment clips into a chapter video with crossfade transitions.
    Uses ffmpeg xfade (video) + acrossfade (audio) between adjacent clips.
    Falls back to simple concat for 1 clip or if crossfade is 0.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(segment_clips) == 0:
        raise RuntimeError("No clips to assemble")

    if len(segment_clips) == 1 or crossfade <= 0:
        return _concat_simple(segment_clips, output_path)

    return _concat_with_crossfade(segment_clips, output_path, crossfade)


def _concat_simple(clips: List[str], output_path: str) -> str:
    """Simple concat without transitions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clips:
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
        ], f"chapter assembly (simple concat): {os.path.basename(output_path)}")
    finally:
        os.unlink(concat_file)

    return output_path


def _concat_with_crossfade(
    clips: List[str], output_path: str, crossfade: float
) -> str:
    """
    Concatenate clips with xfade/acrossfade transitions.
    For N clips, builds a chain of N-1 xfade filters.
    """
    n = len(clips)

    # Probe durations of all clips
    durations = [_probe_duration(c) for c in clips]

    # Build ffmpeg inputs
    inputs = []
    for clip in clips:
        inputs += ["-i", clip]

    # Build xfade filter chain:
    # [0:v][1:v]xfade=transition=fade:duration=D:offset=O[v01];
    # [v01][2:v]xfade=...
    video_filters = []
    audio_filters = []

    # Calculate offsets: each xfade starts at (cumulative_duration - crossfade)
    # The crossfade eats into the end of clip_i and start of clip_{i+1}
    cumulative = durations[0]

    for i in range(1, n):
        offset = max(0, cumulative - crossfade)

        # Video xfade
        if i == 1:
            vin_a = "[0:v]"
        else:
            vin_a = f"[v{i-1}]"

        vin_b = f"[{i}:v]"
        vout = f"[v{i}]" if i < n - 1 else "[vout]"

        video_filters.append(
            f"{vin_a}{vin_b}xfade=transition=fade:duration={crossfade:.2f}"
            f":offset={offset:.2f}{vout}"
        )

        # Audio crossfade
        if i == 1:
            ain_a = "[0:a]"
        else:
            ain_a = f"[a{i-1}]"

        ain_b = f"[{i}:a]"
        aout = f"[a{i}]" if i < n - 1 else "[aout]"

        audio_filters.append(
            f"{ain_a}{ain_b}acrossfade=d={crossfade:.2f}:c1=tri:c2=tri{aout}"
        )

        # After xfade, effective cumulative shortens by crossfade
        cumulative = offset + durations[i]

    filter_complex = ";".join(video_filters + audio_filters)

    _run_ffmpeg(
        inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        f"chapter assembly (crossfade): {os.path.basename(output_path)}",
    )

    return output_path


def assemble_full_book(
    chapter_videos: List[str],
    output_path: str,
) -> str:
    """Concatenate chapter videos into a full book video."""
    return assemble_chapter(chapter_videos, output_path)


def build_project_video(
    segments: List[Tuple[str, str, str]],
    output_dir: str,
    crossfade: float = CROSSFADE_DURATION,
) -> str:
    """
    Build the full project video from segment data.

    Args:
        segments: List of (visual_path, audio_path, visual_type) tuples in order
        output_dir: Directory to write intermediate and final files
        crossfade: Duration of crossfade transition between segments (seconds)

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
    assemble_chapter(clip_paths, final_path, crossfade=crossfade)

    logger.info(f"Built project video: {final_path} ({len(clip_paths)} segments)")
    return final_path
