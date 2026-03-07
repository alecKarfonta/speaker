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
COMFYUI_VIDEO_FRAMES = int(os.environ.get("COMFYUI_VIDEO_FRAMES", "97"))
COMFYUI_VIDEO_FPS = int(os.environ.get("COMFYUI_VIDEO_FPS", "24"))
MAX_VIDEO_FRAMES = 449  # Safe VRAM limit at 768x512 on 32GB GPU

# LTX-2.3 model files
LTX23_CHECKPOINT = "ltx-2.3-22b-distilled.safetensors"
LTX23_TEXT_ENCODER = "gemma_3_12B_it.safetensors"
LTX23_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
LTX23_SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

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

        # Set FPS on LTXVConditioning, SaveAnimatedWEBP, and VHS_VideoCombine
        if fps is not None:
            if ct == "LTXVConditioning" and "frame_rate" in inputs:
                inputs["frame_rate"] = float(fps)
            if ct == "SaveAnimatedWEBP" and "fps" in inputs:
                inputs["fps"] = float(fps)
            if ct == "VHS_VideoCombine" and "frame_rate" in inputs:
                inputs["frame_rate"] = float(fps)

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

        # Update resize edge for LTX-2.3 I2V preprocessing
        if ct == "ResizeImagesByLongerEdge":
            inputs["longer_edge"] = max(width, height)

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
        # Check for videos/gifs (SaveAnimatedWEBP uses 'gifs', VHS_VideoCombine uses 'gifs' too)
        for vid in node_output.get("gifs", node_output.get("videos", [])):
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


def _build_ltx23_workflow(
    text: str, width: int, height: int, frames: int, fps: int, seed: int,
    image_path: str = None,
    enable_audio: bool = False,
    two_stage: bool = False,
    negative: str = "blurry, low quality, watermark, overlay, titles, has blurbox",
    cfg: float = 4.0,
    steps: int = 20,
    lora_strength: float = 0.6,
    upscale_cfg: float = 3.0,
) -> Dict[str, Any]:
    """Build a ComfyUI workflow dict for LTX-2.3 (text-to-video or image-to-video).
    Supports optional audio generation and two-stage spatial upscale.
    Based on the LTX-2.3 reference script.
    """
    prompt = {
        # Load checkpoint
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": LTX23_CHECKPOINT}},
        # Text encoder
        "2": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": LTX23_TEXT_ENCODER,
                         "ckpt_name": LTX23_CHECKPOINT,
                         "device": "default"}},
        # Positive prompt
        "3": {"class_type": "CLIPTextEncode",
              "_meta": {"title": "Positive Prompt"},
              "inputs": {"text": text, "clip": ["2", 0]}},
        # Negative prompt
        "4": {"class_type": "CLIPTextEncode",
              "_meta": {"title": "Negative Prompt"},
              "inputs": {"text": negative, "clip": ["2", 0]}},
        # Conditioning
        "5": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                         "frame_rate": float(fps)}},
        # Empty latent video
        "13": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height,
                           "length": frames, "batch_size": 1}},
    }

    # I2V: load, resize, preprocess, inplace
    video_latent_ref = ["13", 0]
    if image_path:
        prompt.update({
            "6": {"class_type": "LoadImage",
                  "_meta": {"title": "Reference Image"},
                  "inputs": {"image": image_path}},
            "7": {"class_type": "ResizeImagesByLongerEdge",
                  "inputs": {"images": ["6", 0],
                             "longer_edge": max(width, height)}},
            "8": {"class_type": "LTXVPreprocess",
                  "inputs": {"image": ["7", 0], "img_compression": 33}},
            "14": {"class_type": "LTXVImgToVideoInplace",
                   "inputs": {"vae": ["1", 2], "image": ["8", 0],
                              "latent": ["13", 0],
                              "strength": 1.0, "bypass": False}},
        })
        video_latent_ref = ["14", 0]

    # Audio latent (optional, off by default)
    if enable_audio:
        prompt.update({
            "15": {"class_type": "LTXVAudioVAELoader",
                   "inputs": {"ckpt_name": LTX23_CHECKPOINT}},
            "16": {"class_type": "LTXVEmptyLatentAudio",
                   "inputs": {"audio_vae": ["15", 0],
                              "frames_number": frames,
                              "frame_rate": fps, "batch_size": 1}},
            "17": {"class_type": "LTXVConcatAVLatent",
                   "inputs": {"video_latent": video_latent_ref,
                              "audio_latent": ["16", 0]}},
        })
        sample_latent_ref = ["17", 0]
    else:
        sample_latent_ref = video_latent_ref

    # Distilled LoRA
    model_ref = ["1", 0]
    if lora_strength > 0:
        prompt["18"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0],
                       "lora_name": LTX23_DISTILLED_LORA,
                       "strength_model": lora_strength}
        }
        model_ref = ["18", 0]

    # Sampling
    prompt.update({
        "19": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": seed}},
        "20": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"}},
        "21": {"class_type": "LTXVScheduler",
               "inputs": {"latent": sample_latent_ref, "steps": steps,
                           "max_shift": 2.05, "base_shift": 0.95,
                           "stretch": True, "terminal": 0.1}},
        "22": {"class_type": "CFGGuider",
               "inputs": {"model": model_ref,
                           "positive": ["5", 0],
                           "negative": ["5", 1],
                           "cfg": cfg}},
        "23": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["19", 0], "guider": ["22", 0],
                           "sampler": ["20", 0], "sigmas": ["21", 0],
                           "latent_image": sample_latent_ref}},
    })

    # Separate AV if audio
    if enable_audio:
        prompt["24"] = {"class_type": "LTXVSeparateAVLatent",
                        "inputs": {"av_latent": ["23", 0]}}
        decode_source = "24"
    else:
        decode_source = "23"

    # Two-stage spatial upscale (optional)
    if two_stage:
        prompt.update({
            "25": {"class_type": "LTXVCropGuides",
                   "inputs": {"positive": ["5", 0], "negative": ["5", 1],
                              "latent": [decode_source, 0]}},
            "26": {"class_type": "LatentUpscaleModelLoader",
                   "inputs": {"model_name": LTX23_SPATIAL_UPSCALER}},
            "27": {"class_type": "LTXVLatentUpsampler",
                   "inputs": {"samples": ["25", 2],
                              "upscale_model": ["26", 0],
                              "vae": ["1", 2]}},
        })
        upscaled_video_ref = ["27", 0]
        if image_path:
            prompt["28"] = {
                "class_type": "LTXVImgToVideoInplace",
                "inputs": {"vae": ["1", 2], "image": ["8", 0],
                           "latent": ["27", 0],
                           "strength": 1.0, "bypass": False}
            }
            upscaled_video_ref = ["28", 0]
        if enable_audio:
            prompt["29"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": upscaled_video_ref,
                           "audio_latent": [decode_source, 1]}
            }
            upscale_sample_ref = ["29", 0]
        else:
            upscale_sample_ref = upscaled_video_ref
        prompt.update({
            "30": {"class_type": "RandomNoise",
                   "inputs": {"noise_seed": 0}},
            "31": {"class_type": "KSamplerSelect",
                   "inputs": {"sampler_name": "gradient_estimation"}},
            "32": {"class_type": "ManualSigmas",
                   "inputs": {"sigmas": "0.909375, 0.725, 0.421875, 0.0"}},
            "33": {"class_type": "CFGGuider",
                   "inputs": {"model": model_ref,
                              "positive": ["25", 0],
                              "negative": ["25", 1],
                              "cfg": upscale_cfg}},
            "34": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["30", 0], "guider": ["33", 0],
                              "sampler": ["31", 0], "sigmas": ["32", 0],
                              "latent_image": upscale_sample_ref}},
        })
        if enable_audio:
            prompt["35"] = {"class_type": "LTXVSeparateAVLatent",
                            "inputs": {"av_latent": ["34", 1]}}
            decode_source = "35"
        else:
            decode_source = "34"

    # Decode video
    prompt["36"] = {"class_type": "VAEDecode",
                    "inputs": {"samples": [decode_source, 0],
                               "vae": ["1", 2]}}

    # Output: audio mux or plain video
    if enable_audio:
        prompt.update({
            "37": {"class_type": "LTXVAudioVAEDecode",
                   "inputs": {"samples": [decode_source, 1],
                              "audio_vae": ["15", 0]}},
            "38": {"class_type": "CreateVideo",
                   "inputs": {"images": ["36", 0], "audio": ["37", 0],
                              "fps": fps}},
            "39": {"class_type": "SaveVideo",
                   "inputs": {"video": ["38", 0],
                              "filename_prefix": "audiobook_video",
                              "format": "mp4", "codec": "h264"}},
        })
    else:
        prompt["39"] = {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Save Video"},
            "inputs": {"images": ["36", 0],
                       "frame_rate": float(fps), "loop_count": 0,
                       "filename_prefix": "audiobook_video",
                       "format": "video/h264-mp4",
                       "pingpong": False, "save_output": True}
        }

    return prompt


async def generate_video(prompt: str, output_dir: str, prefix: str = "visual",
                         width: int = None, height: int = None,
                         frames: int = None, fps: int = None,
                         duration: float = None,
                         enable_audio: bool = False,
                         two_stage: bool = False) -> str:
    """Generate a video using ComfyUI LTX-2.3 22B workflow.
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
    stage_str = "two-stage" if two_stage else "single-stage"
    audio_str = "+audio" if enable_audio else ""
    logger.info(f"Video gen (LTX-2.3 {stage_str}{audio_str}): {num_frames} frames @ {video_fps}fps = {num_frames/video_fps:.1f}s")

    workflow = _build_ltx23_workflow(
        text=prompt, width=w, height=h, frames=num_frames,
        fps=video_fps, seed=seed,
        enable_audio=enable_audio, two_stage=two_stage,
    )

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id, timeout=900)
    path = await download_output(result, output_dir, prefix)
    if not path:
        raise RuntimeError(f"ComfyUI video generation produced no output for prompt_id={prompt_id}")
    return path


async def generate_visual(prompt: str, output_dir: str, prefix: str = "visual",
                          mode: str = None, duration: float = None,
                          ref_image: str = None,
                          width: int = None, height: int = None,
                          frames: int = None, fps: int = None,
                          enable_audio: bool = False,
                          two_stage: bool = False) -> Tuple[str, str]:
    """
    Generate a visual based on one of 5 explicit modes:
      - image:       Text → image (Qwen-Image + Lightning hires-fix)
      - video:       Text → video (LTX-2.3 22B)
      - scene_image: Portrait ref → FaceID scene image (IP-Adapter + SDXL Lightning)
      - ref_video:   Image ref → guided video (LTX-2.3 I2V)
      - scene_video: Two-stage: FaceID scene image → animate with LTX-2.3

    Returns (path, visual_type) where visual_type is 'image' or 'video'.
    """
    visual_mode = mode or COMFYUI_VISUAL_MODE

    if visual_mode == "scene_video":
        # Two-stage: FaceID scene image → animate with LTX-2.3
        if not ref_image:
            raise ValueError("scene_video mode requires a character portrait (ref_image)")
        path = await generate_scene_video(
            prompt, ref_image, output_dir, prefix,
            duration=duration, width=width, height=height,
            frames=frames, fps=fps,
            enable_audio=enable_audio, two_stage=two_stage,
        )
        return path, "video"

    elif visual_mode == "scene_image":
        # FaceID scene image via IP-Adapter + SDXL Lightning
        if not ref_image:
            raise ValueError("scene_image mode requires a character portrait (ref_image)")
        path = await generate_scene_image(
            prompt, ref_image, output_dir, prefix,
            width=width, height=height,
        )
        return path, "image"

    elif visual_mode == "ref_video":
        # Image-guided video via LTX-2.3 I2V
        if not ref_image:
            raise ValueError("ref_video mode requires a reference image (ref_image)")
        path = await generate_video_with_ref(
            prompt, ref_image, output_dir, prefix,
            duration=duration, width=width, height=height,
            frames=frames, fps=fps,
            enable_audio=enable_audio, two_stage=two_stage,
        )
        return path, "video"

    elif visual_mode == "video":
        # Basic text-to-video
        path = await generate_video(
            prompt, output_dir, prefix,
            duration=duration, width=width, height=height,
            frames=frames, fps=fps,
            enable_audio=enable_audio, two_stage=two_stage,
        )
        return path, "video"

    else:
        # Default: basic text-to-image
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
    enable_audio: bool = False,
    two_stage: bool = False,
) -> str:
    """
    Generate a video guided by a reference image using LTX-2.3 LTXVImgToVideoInplace.
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
    stage_str = "two-stage" if two_stage else "single-stage"
    audio_str = "+audio" if enable_audio else ""
    logger.info(f"Video gen (LTX-2.3 I2V {stage_str}{audio_str}): {num_frames} frames @ {video_fps}fps, ref={ref_image_comfyui}")

    workflow = _build_ltx23_workflow(
        text=prompt, width=w, height=h, frames=num_frames,
        fps=video_fps, seed=seed, image_path=ref_image_comfyui,
        enable_audio=enable_audio, two_stage=two_stage,
    )

    prompt_id = await queue_prompt(workflow)
    result = await poll_completion(prompt_id, timeout=900)
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
    enable_audio: bool = False,
    two_stage: bool = False,
) -> str:
    """
    Two-call approach for character-consistent video:
    1. Generate a scene image with IP-Adapter FaceID (SDXL Lightning)
    2. Animate it with LTX-2.3 Image-to-Video

    This produces much better results than a single combined workflow because
    each model runs at its optimal resolution and settings.
    """
    import tempfile

    logger.info(f"Scene video gen: generating scene image first with IP-Adapter FaceID")

    # Step 1: Generate scene image with IP-Adapter FaceID
    with tempfile.TemporaryDirectory() as tmp_dir:
        scene_image_path = await generate_scene_image(
            prompt, ref_image_comfyui, tmp_dir, prefix="scene_frame",
            width=768, height=512,  # Match LTX-2.3 input resolution
        )

        # Step 2: Upload the scene image to ComfyUI for I2V
        scene_comfyui_name = await upload_image(scene_image_path)
        logger.info(f"Scene image uploaded as: {scene_comfyui_name}, animating with LTX-2.3")

        # Step 3: Animate with LTX-2.3 I2V
        path = await generate_video_with_ref(
            prompt, scene_comfyui_name, output_dir, prefix,
            duration=duration, width=width, height=height,
            frames=frames, fps=fps,
            enable_audio=enable_audio, two_stage=two_stage,
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
