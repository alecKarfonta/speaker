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
        raise RuntimeError(f"{description} failed:\n{result.stderr[-2000:]}")


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
    animation_style: Optional[str] = None,
    video_fill_mode: str = "hold",
) -> str:
    """
    Create a single segment clip by combining a visual and audio file.

    For images: applies a Ken Burns / pan animation over the still image.
    For videos: handles duration mismatch via video_fill_mode (loop/hold/fade).
    Both: output normalized to 1920x1080, with brief fade in/out.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_dur = _probe_duration(audio_path)

    if visual_type == "image":
        _create_image_clip(visual_path, audio_path, output_path, audio_dur,
                           animation_style=animation_style)
    else:
        _create_video_clip(visual_path, audio_path, output_path, audio_dur,
                           video_fill_mode=video_fill_mode)

    return output_path


def _create_image_clip(
    visual_path: str, audio_path: str, output_path: str, audio_dur: float,
    animation_style: Optional[str] = None,
) -> None:
    """Apply a Ken Burns / pan animation over a still image for the audio duration.

    animation_style options:
        zoom_in    — slow zoom toward center (default random option A)
        zoom_out   — slow zoom back from center (default random option B)
        pan_left   — camera drifts left to right
        pan_right  — camera drifts right to left
        pan_up     — camera drifts upward
        static     — no camera movement, just the still image
        random     — randomly pick zoom_in or zoom_out (backward-compatible default)
        None       — same as random
    """
    fps = 25
    total_frames = int(audio_dur * fps) + 1

    style = animation_style or "random"

    if style == "random":
        style = random.choice(["zoom_in", "zoom_out"])

    if style == "zoom_in":
        zoom_expr = f"min(1+0.15*on/{total_frames},1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == "zoom_out":
        zoom_expr = f"max(1.15-0.15*on/{total_frames},1.0)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == "pan_left":
        # Slight zoom, pan from left edge toward right
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*on/{total_frames}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == "pan_right":
        # Slight zoom, pan from right edge toward left
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == "pan_up":
        # Slight zoom, drift upward
        zoom_expr = "1.05"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(ih-ih/zoom)*(1-on/{total_frames})"
    else:  # static
        zoom_expr = "1.0"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    _run_ffmpeg([
        "-i", visual_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:v]scale=1920:1080,zoompan=z='{zoom_expr}':d={total_frames}"
        f":x='{x_expr}':y='{y_expr}':s=1920x1080:fps={fps},"
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
    ], f"segment clip ({style}): {os.path.basename(output_path)}")


def _is_animated_webp(path: str) -> bool:
    """Check if a file is an animated WebP by scanning for ANIM/ANMF RIFF chunks."""
    try:
        with open(path, "rb") as f:
            header = f.read(64)
        return b"ANIM" in header or b"ANMF" in header
    except Exception:
        return False


def _convert_animated_webp_to_mp4(webp_path: str, output_path: str) -> None:
    """
    Convert an animated WebP to a silent MP4 using Pillow to extract frames.
    This bypasses ffmpeg's broken webp_pipe demuxer for ANIM/ANMF chunks.
    """
    from PIL import Image
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="webp_frames_")
    try:
        img = Image.open(webp_path)

        frame_count = 0
        frame_durations = []  # ms per frame

        while True:
            duration = img.info.get("duration", 100)  # ms, default 100ms
            frame_durations.append(duration)
            # Convert RGBA → RGB for x264 compatibility
            frame_rgb = img.convert("RGB")
            frame_path = os.path.join(tmp_dir, f"frame_{frame_count:05d}.png")
            frame_rgb.save(frame_path, "PNG")
            frame_count += 1
            try:
                img.seek(img.tell() + 1)
            except EOFError:
                break

        if frame_count == 0:
            raise RuntimeError("No frames extracted from animated WebP")

        # Compute average FPS from frame durations
        avg_ms = sum(frame_durations) / len(frame_durations)
        fps = round(1000.0 / max(avg_ms, 10))  # clamp to sane fps
        logger.info(f"Extracted {frame_count} frames at avg {fps} fps from {os.path.basename(webp_path)}")

        # Build MP4 from frames
        _run_ffmpeg([
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ], f"frames → MP4: {os.path.basename(webp_path)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



def _create_video_clip(
    visual_path: str, audio_path: str, output_path: str, audio_dur: float,
    video_fill_mode: str = "hold",
) -> None:
    """
    Video clip synced to audio duration.
    Handles animated WebP (ComfyUI output) by pre-converting to MP4.

    video_fill_mode controls what happens when visual is shorter than audio:
      hold  — freeze last frame then fade to black (default, cinematic)
      loop  — loop the video from the beginning
      fade  — video plays once, then fadeout begins immediately at video end
    """
    # ── Animated WebP pre-conversion ──────────────────────────────────────
    ext = os.path.splitext(visual_path)[1].lower()
    tmp_mp4 = None
    if ext == ".webp" or _is_animated_webp(visual_path):
        tmp_mp4 = visual_path.rsplit(".", 1)[0] + "_tmp.mp4"
        try:
            _convert_animated_webp_to_mp4(visual_path, tmp_mp4)
            visual_path = tmp_mp4
            logger.info(f"Converted animated WebP to tmp MP4: {tmp_mp4}")
        except Exception as e:
            logger.warning(f"Animated WebP conversion failed ({e}), falling back to still image")
            _create_image_clip(visual_path, audio_path, output_path, audio_dur)
            if tmp_mp4 and os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)
            return

    # ── Standard video clip assembly ─────────────────────────────────────
    try:
        video_dur = _probe_duration(visual_path)
        fade_d = 0.3
        fps = 25  # normalize output FPS

        needs_fill = video_dur > 0 and video_dur < audio_dur
        mode = video_fill_mode if needs_fill else "trim"

        if mode == "loop":
            loops_needed = math.ceil(audio_dur / video_dur)
            input_args = ["-stream_loop", str(loops_needed - 1), "-i", visual_path, "-i", audio_path]
            vf = (
                f"fps={fps},"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fade=t=in:st=0:d={fade_d},"
                f"fade=t=out:st={max(0, audio_dur - fade_d):.2f}:d={fade_d}"
            )
        elif mode == "fade":
            # Play video once; fade to black starting at video end
            fade_start = max(0, video_dur - fade_d)
            input_args = ["-i", visual_path, "-i", audio_path]
            vf = (
                f"fps={fps},"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"tpad=stop_mode=add:stop_duration={audio_dur - video_dur + fade_d:.2f}:color=black,"
                f"fade=t=in:st=0:d={fade_d},"
                f"fade=t=out:st={fade_start:.2f}:d={fade_d}"
            )
        elif mode == "hold":
            # Freeze last frame, fade out at end of audio
            pad_frames = math.ceil((audio_dur - video_dur) * fps) + 1
            input_args = ["-i", visual_path, "-i", audio_path]
            vf = (
                f"fps={fps},"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"tpad=stop_mode=clone:stop={pad_frames},"
                f"fade=t=in:st=0:d={fade_d},"
                f"fade=t=out:st={max(0, audio_dur - fade_d):.2f}:d={fade_d}"
            )
        else:  # trim (video >= audio or unknown)
            input_args = ["-i", visual_path, "-i", audio_path]
            vf = (
                f"fps={fps},"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fade=t=in:st=0:d={fade_d},"
                f"fade=t=out:st={max(0, audio_dur - fade_d):.2f}:d={fade_d}"
            )

        _run_ffmpeg(
            input_args + [
                "-t", f"{audio_dur:.2f}",
                "-filter_complex", f"[0:v]{vf}[vout]",
                "-map", "[vout]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            f"segment clip (video/{mode}): {os.path.basename(output_path)}",
        )
    finally:
        if tmp_mp4 and os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)





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
    All inputs are first normalized to fps=25 to ensure matching timebases.
    """
    n = len(clips)

    # Probe durations of all clips
    durations = [_probe_duration(c) for c in clips]

    # Build ffmpeg inputs
    inputs = []
    for clip in clips:
        inputs += ["-i", clip]

    # Normalize each input to 25fps + matching timebase before xfade.
    # Without this, clips encoded at different FPS (e.g. 8fps from animated WebP
    # vs 25fps from image Ken Burns) have different timebases and xfade fails.
    FPS = 25
    norm_filters = []
    for i in range(n):
        norm_filters.append(f"[{i}:v]fps={FPS},settb=1/{FPS}[nv{i}]")
        norm_filters.append(f"[{i}:a]asetpts=PTS-STARTPTS[na{i}]")

    # Build xfade chain on normalized streams
    video_filters = []
    audio_filters = []
    cumulative = durations[0]

    for i in range(1, n):
        offset = max(0, cumulative - crossfade)

        vin_a = f"[nv{i-1}]" if i == 1 else f"[v{i-1}]"
        vin_b = f"[nv{i}]"
        vout = f"[v{i}]" if i < n - 1 else "[vout]"

        video_filters.append(
            f"{vin_a}{vin_b}xfade=transition=fade:duration={crossfade:.2f}"
            f":offset={offset:.2f}{vout}"
        )

        ain_a = f"[na{i-1}]" if i == 1 else f"[a{i-1}]"
        ain_b = f"[na{i}]"
        aout = f"[a{i}]" if i < n - 1 else "[aout]"

        audio_filters.append(
            f"{ain_a}{ain_b}acrossfade=d={crossfade:.2f}:c1=tri:c2=tri{aout}"
        )

        cumulative = offset + durations[i]

    filter_complex = ";".join(norm_filters + video_filters + audio_filters)

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
            "-movflags", "+faststart",
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
    segments: List[Tuple[str, str, str, Optional[str]]],
    output_dir: str,
    crossfade: float = CROSSFADE_DURATION,
) -> str:
    """
    Build the full project video from segment data.

    Args:
        segments: List of (visual_path, audio_path, visual_type, animation_style, video_fill_mode) tuples.
        output_dir: Directory to write intermediate and final files
        crossfade: Duration of crossfade transition between segments (seconds)

    Returns:
        Path to the final MP4
    """
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []
    for i, seg_data in enumerate(segments):
        visual_path, audio_path, visual_type = seg_data[0], seg_data[1], seg_data[2]
        animation_style = seg_data[3] if len(seg_data) > 3 else None
        video_fill_mode = seg_data[4] if len(seg_data) > 4 else "hold"
        clip_path = os.path.join(clips_dir, f"segment_{i:04d}.mp4")
        create_segment_clip(visual_path, audio_path, clip_path, visual_type,
                            animation_style=animation_style,
                            video_fill_mode=video_fill_mode)
        clip_paths.append(clip_path)

    final_path = os.path.join(output_dir, "full_book.mp4")
    assemble_chapter(clip_paths, final_path, crossfade=crossfade)

    logger.info(f"Built project video: {final_path} ({len(clip_paths)} segments)")
    return final_path

