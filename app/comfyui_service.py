"""
ComfyUI API client for image and video generation.
Queues workflows via REST API, polls for completion, downloads results.
"""

import json
import os
import time
import uuid
import random
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
import httpx

logger = logging.getLogger(__name__)

COMFYUI_API_URL = os.environ.get("COMFYUI_API_URL", "http://192.168.1.83:8188")
COMFYUI_VISUAL_MODE = os.environ.get("COMFYUI_VISUAL_MODE", "image")  # "image" or "video"
COMFYUI_IMAGE_WIDTH = int(os.environ.get("COMFYUI_IMAGE_WIDTH", "1024"))
COMFYUI_IMAGE_HEIGHT = int(os.environ.get("COMFYUI_IMAGE_HEIGHT", "1024"))
COMFYUI_VIDEO_FRAMES = int(os.environ.get("COMFYUI_VIDEO_FRAMES", "65"))
COMFYUI_VIDEO_FPS = int(os.environ.get("COMFYUI_VIDEO_FPS", "10"))
MAX_VIDEO_FRAMES = 449  # Safe VRAM limit at 768x512 on 32GB GPU

WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "comfyui_workflows")


def _load_workflow(name: str) -> Dict[str, Any]:
    """Load a workflow JSON from the workflows directory."""
    path = os.path.join(WORKFLOWS_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def _inject_params(workflow: Dict[str, Any], prompt: str,
                   width: int, height: int, seed: int,
                   batch_size: int = 1,
                   output_width: int = None,
                   output_height: int = None,
                   video_length: int = None,
                   fps: int = None,
                   ref_image: str = None,
                   ref_image2: str = None) -> Dict[str, Any]:
    """
    Inject prompt, dimensions, seed, and batch_size into a workflow.
    Finds nodes by class_type and updates their inputs.
    output_width/output_height set the final ImageScale node dimensions.
    video_length sets the frame count for video latent nodes.
    ref_image: character portrait filename (uploaded to ComfyUI).
    ref_image2: background image filename (uploaded to ComfyUI).
    """
    for node_id, node in workflow.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        meta_title = node.get("_meta", {}).get("title", "").lower()

        # Set positive prompt on CLIPTextEncode nodes
        if ct == "CLIPTextEncode":
            if "positive" in meta_title or inputs.get("text") == "PLACEHOLDER_PROMPT":
                inputs["text"] = prompt

        # Set prompt on TextEncodeQwenImageEditPlus (multi-ref scene gen)
        if ct == "TextEncodeQwenImageEditPlus":
            if inputs.get("prompt") == "PLACEHOLDER_PROMPT":
                inputs["prompt"] = prompt

        # Set dimensions and batch_size on EmptyLatentImage
        if ct == "EmptyLatentImage":
            inputs["width"] = width
            inputs["height"] = height
            inputs["batch_size"] = batch_size

        # Set dimensions and frame length on EmptyLTXVLatentVideo
        if ct == "EmptyLTXVLatentVideo":
            inputs["width"] = width
            inputs["height"] = height
            if video_length is not None:
                inputs["length"] = video_length

        # Set seed on KSampler / KSamplerAdvanced
        if "KSampler" in ct:
            if "seed" in inputs:
                inputs["seed"] = seed

        # Set seed on RandomNoise (used in advanced sampler workflows)
        if ct == "RandomNoise":
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed

        # Set output resolution on ImageScale (final crop/resize)
        if ct == "ImageScale" and (output_width or output_height):
            if output_width:
                inputs["width"] = output_width
            if output_height:
                inputs["height"] = output_height

        # Set FPS on LTXVConditioning and SaveAnimatedWEBP
        if fps is not None:
            if ct == "LTXVConditioning" and "frame_rate" in inputs:
                inputs["frame_rate"] = float(fps)
            if ct == "SaveAnimatedWEBP" and "fps" in inputs:
                inputs["fps"] = float(fps)

        # Set reference image on LoadImage nodes
        if ref_image is not None and ct == "LoadImage":
            # First LoadImage: character portrait
            if inputs.get("image") == "PLACEHOLDER_REF_IMAGE":
                inputs["image"] = ref_image
            # Second LoadImage: background
            elif ref_image2 is not None and inputs.get("image") == "PLACEHOLDER_BG_IMAGE":
                inputs["image"] = ref_image2

        # Set dimensions and frame length on LTXVImgToVideo
        if ct == "LTXVImgToVideo":
            inputs["width"] = width
            inputs["height"] = height
            if video_length is not None:
                inputs["length"] = video_length

    # Dynamically inject background image into TextEncodeQwenImageEditPlus
    if ref_image2 is not None:
        for node_id, node in workflow.items():
            if node.get("class_type") == "TextEncodeQwenImageEditPlus":
                # Add a LoadImage node for the background if not already present
                bg_node_id = None
                for nid, n in workflow.items():
                    if n.get("class_type") == "LoadImage" and n.get("_meta", {}).get("title", "") == "Background Image":
                        bg_node_id = nid
                        break
                if not bg_node_id:
                    # Create a new LoadImage node for the background
                    bg_node_id = "99"
                    workflow[bg_node_id] = {
                        "class_type": "LoadImage",
                        "_meta": {"title": "Background Image"},
                        "inputs": {"image": ref_image2}
                    }
                node["inputs"]["image2"] = [bg_node_id, 0]
                break

    return workflow


async def free_vram() -> None:
    """Ask ComfyUI to unload models and free VRAM before a new generation."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{COMFYUI_API_URL}/free",
                              json={"unload_models": True, "free_memory": True})
            logger.debug("Freed ComfyUI VRAM")
    except Exception:
        pass  # Non-critical — generation may still succeed


async def queue_prompt(workflow: Dict[str, Any]) -> str:
    """Queue a workflow on ComfyUI and return the prompt_id."""
    # Free VRAM from previous runs so models can load cleanly
    await free_vram()

    client_id = str(uuid.uuid4())
    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{COMFYUI_API_URL}/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {data}")
        logger.info(f"Queued ComfyUI prompt {prompt_id}")
        return prompt_id


async def poll_completion(prompt_id: str, timeout: int = 300, interval: float = 2.0) -> Dict[str, Any]:
    """Poll ComfyUI /history/{prompt_id} until the job completes."""
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=15) as client:
        while time.time() < deadline:
            resp = await client.get(f"{COMFYUI_API_URL}/history/{prompt_id}")
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    return data[prompt_id]
            await asyncio.sleep(interval)
    raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within {timeout}s")


async def download_output(prompt_result: Dict[str, Any], output_dir: str, prefix: str = "visual") -> Optional[str]:
    """Download the first output image/video from a completed ComfyUI job."""
    outputs = prompt_result.get("outputs", {})
    for node_id, node_output in outputs.items():
        # Check for images
        for img in node_output.get("images", []):
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            url = f"{COMFYUI_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
            return await _download_file(url, output_dir, prefix, filename)
        # Check for videos/gifs
        for vid in node_output.get("gifs", []):
            filename = vid.get("filename", "")
            subfolder = vid.get("subfolder", "")
            url = f"{COMFYUI_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
            return await _download_file(url, output_dir, prefix, filename)
    return None


async def _download_file(url: str, output_dir: str, prefix: str, original_filename: str) -> str:
    """Download a file from ComfyUI and save it locally."""
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(original_filename)[1] or ".png"
    local_path = os.path.join(output_dir, f"{prefix}{ext}")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
    logger.info(f"Downloaded ComfyUI output to {local_path}")
    return local_path


async def generate_image(prompt: str, output_dir: str, prefix: str = "visual",
                         width: int = None, height: int = None) -> str:
    """Generate an image using ComfyUI Qwen-Image + Lightning hires-fix workflow.
    Generates at 512x512, latent upscales 2x, refines, then scales to output resolution.
    """
    # Base generation size (kept small for VRAM)
    base_w = 512
    base_h = 512
    # Final output size
    out_w = width or COMFYUI_IMAGE_WIDTH
    out_h = height or COMFYUI_IMAGE_HEIGHT
    seed = random.randint(0, 2**32 - 1)

    workflow = _load_workflow("text_to_image.json")
    workflow = _inject_params(workflow, prompt, base_w, base_h, seed,
                              output_width=out_w, output_height=out_h)

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id)
    path = await download_output(result, output_dir, prefix)
    if not path:
        raise RuntimeError(f"ComfyUI image generation produced no output for prompt_id={prompt_id}")
    return path


async def generate_video(prompt: str, output_dir: str, prefix: str = "visual",
                         width: int = None, height: int = None,
                         frames: int = None, fps: int = None,
                         duration: float = None) -> str:
    """Generate a video using ComfyUI LTX-2 19B distilled workflow.
    If duration (seconds) is provided, frames are auto-calculated from duration * fps.
    Frames are capped at MAX_VIDEO_FRAMES to prevent OOM.
    """
    w = width or min(COMFYUI_IMAGE_WIDTH, 768)
    h = height or min(COMFYUI_IMAGE_HEIGHT, 512)
    video_fps = fps or COMFYUI_VIDEO_FPS
    seed = random.randint(0, 2**32 - 1)

    # Calculate frames: from explicit count, or duration * fps, or default
    if frames:
        num_frames = frames
    elif duration:
        # Round up to nearest 8 (LTX-2 latent requirement) + 1
        raw = int(duration * video_fps)
        num_frames = ((raw + 7) // 8) * 8 + 1
    else:
        num_frames = COMFYUI_VIDEO_FRAMES

    # Cap to prevent OOM
    num_frames = min(num_frames, MAX_VIDEO_FRAMES)
    logger.info(f"Video gen: {num_frames} frames @ {video_fps}fps = {num_frames/video_fps:.1f}s")

    workflow = _load_workflow("text_to_video.json")
    workflow = _inject_params(workflow, prompt, w, h, seed,
                              video_length=num_frames, fps=video_fps)

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id, timeout=600)
    path = await download_output(result, output_dir, prefix)
    if not path:
        raise RuntimeError(f"ComfyUI video generation produced no output for prompt_id={prompt_id}")
    return path


async def generate_visual(prompt: str, output_dir: str, prefix: str = "visual",
                          mode: str = None, duration: float = None,
                          ref_image: str = None,
                          width: int = None, height: int = None,
                          frames: int = None, fps: int = None) -> Tuple[str, str]:
    """
    Generate a visual (image or video) based on mode.
    If ref_image is provided (ComfyUI filename), uses character-consistent
    scene generation via TextEncodeQwenImageEditPlus.
    Returns (path, type) where type is 'image' or 'video'.
    """
    visual_mode = mode or COMFYUI_VISUAL_MODE
    if visual_mode == "video":
        if ref_image:
            # Character-consistent video: scene image with portrait → animate
            path = await generate_scene_video(
                prompt, ref_image, output_dir, prefix,
                duration=duration, width=width, height=height,
                frames=frames, fps=fps,
            )
        else:
            path = await generate_video(
                prompt, output_dir, prefix,
                duration=duration, width=width, height=height,
                frames=frames, fps=fps,
            )
        return path, "video"
    else:
        if ref_image:
            # Character-consistent image: scene with portrait reference
            path = await generate_scene_image(
                prompt, ref_image, output_dir, prefix,
                width=width, height=height,
            )
        else:
            path = await generate_image(prompt, output_dir, prefix, width=width, height=height)
        return path, "image"


async def upload_image(local_path: str) -> str:
    """
    Upload an image to ComfyUI's input directory via /upload/image.
    Returns the filename on the ComfyUI server.
    """
    filename = os.path.basename(local_path)
    async with httpx.AsyncClient(timeout=30) as client:
        with open(local_path, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            resp = await client.post(
                f"{COMFYUI_API_URL}/upload/image",
                files=files,
            )
            resp.raise_for_status()
            data = resp.json()
            uploaded_name = data.get("name", filename)
            logger.info(f"Uploaded image to ComfyUI: {uploaded_name}")
            return uploaded_name


async def generate_video_with_ref(
    prompt: str,
    ref_image_comfyui: str,
    output_dir: str,
    prefix: str = "visual",
    width: int = None,
    height: int = None,
    frames: int = None,
    fps: int = None,
    duration: float = None,
) -> str:
    """
    Generate a video guided by a reference image using LTXVImgToVideo.
    ref_image_comfyui is the filename already uploaded to ComfyUI.
    """
    w = width or min(COMFYUI_IMAGE_WIDTH, 768)
    h = height or min(COMFYUI_IMAGE_HEIGHT, 512)
    video_fps = fps or COMFYUI_VIDEO_FPS
    seed = random.randint(0, 2**32 - 1)

    # Calculate frames
    if frames:
        num_frames = frames
    elif duration:
        raw = int(duration * video_fps)
        num_frames = ((raw + 7) // 8) * 8 + 1
    else:
        num_frames = COMFYUI_VIDEO_FRAMES

    num_frames = min(num_frames, MAX_VIDEO_FRAMES)
    logger.info(f"Video gen (ref): {num_frames} frames @ {video_fps}fps, ref={ref_image_comfyui}")

    workflow = _load_workflow("text_to_video_ref.json")
    workflow = _inject_params(
        workflow, prompt, w, h, seed,
        video_length=num_frames, fps=video_fps,
        ref_image=ref_image_comfyui,
    )

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id, timeout=600)
    path = await download_output(result, output_dir, prefix)
    if not path:
        raise RuntimeError(f"ComfyUI ref video generation produced no output for prompt_id={prompt_id}")
    return path


async def generate_scene_image(
    prompt: str,
    ref_image_comfyui: str,
    output_dir: str,
    prefix: str = "visual",
    width: int = None,
    height: int = None,
    bg_image_comfyui: str = None,
) -> str:
    """
    Generate a scene image with a character portrait as face reference.
    Uses IP-Adapter FaceID Plus V2 with SDXL Lightning for character-consistent
    generation. The portrait's facial identity is injected into the diffusion
    process so the generated scene follows the prompt while preserving the
    character's appearance.
    """
    # SDXL widescreen resolution (closest native to 16:9)
    base_w = 1344
    base_h = 768
    out_w = width or COMFYUI_IMAGE_WIDTH
    out_h = height or COMFYUI_IMAGE_HEIGHT
    seed = random.randint(0, 2**32 - 1)

    logger.info(f"Scene image gen (IP-Adapter FaceID): ref={ref_image_comfyui}")

    workflow = _load_workflow("ref_scene_image.json")
    workflow = _inject_params(
        workflow, prompt, base_w, base_h, seed,
        output_width=out_w, output_height=out_h,
        ref_image=ref_image_comfyui,
    )

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id, timeout=300)
    path = await download_output(result, output_dir, prefix)
    if not path:
        raise RuntimeError(f"ComfyUI scene image generation produced no output for prompt_id={prompt_id}")
    return path


async def generate_scene_video(
    prompt: str,
    ref_image_comfyui: str,
    output_dir: str,
    prefix: str = "visual",
    width: int = None,
    height: int = None,
    frames: int = None,
    fps: int = None,
    duration: float = None,
    bg_image_comfyui: str = None,
) -> str:
    """
    Two-call approach for character-consistent video:
    1. Generate a scene image with IP-Adapter FaceID (SDXL Lightning)
    2. Animate it with LTX-2 Image-to-Video

    This produces much better results than a single combined workflow because
    each model runs at its optimal resolution and settings.
    """
    import tempfile

    logger.info(f"Scene video gen: generating scene image first with IP-Adapter FaceID")

    # Step 1: Generate scene image with IP-Adapter FaceID
    with tempfile.TemporaryDirectory() as tmp_dir:
        scene_image_path = await generate_scene_image(
            prompt, ref_image_comfyui, tmp_dir, prefix="scene_frame",
            width=768, height=512,  # Match LTX-2 input resolution
        )

        # Step 2: Upload the scene image to ComfyUI for I2V
        scene_comfyui_name = await upload_image(scene_image_path)
        logger.info(f"Scene image uploaded as: {scene_comfyui_name}, animating with LTX-2")

        # Step 3: Animate with LTX-2 I2V
        path = await generate_video_with_ref(
            prompt, scene_comfyui_name, output_dir, prefix,
            duration=duration, width=width, height=height,
            frames=frames, fps=fps,
        )

    return path


async def health_check() -> bool:
    """Check if ComfyUI is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{COMFYUI_API_URL}/system_stats")
            return resp.status_code == 200
    except Exception:
        return False
