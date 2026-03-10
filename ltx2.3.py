#!/usr/bin/env python3
"""
Generate LTX-2.3 videos from the command line (text-to-video & image-to-video).

Usage:
  # Text-to-video (single stage, fast)
  python3 generate_ltx23_video.py --prompt "A golden retriever running through a meadow at sunset"

  # Image-to-video
  python3 generate_ltx23_video.py --image input/my_photo.jpg --prompt "A person smiling and waving"

  # Two-stage with upscale (higher quality, slower)
  python3 generate_ltx23_video.py --prompt "Ocean waves crashing on rocks" --two-stage

  # Quick test (fewer frames, no upscale)
  python3 generate_ltx23_video.py --prompt "A cat walking" --frames 33 --no-audio
"""
import argparse
import json
import os
import requests
import sys
import time
import uuid

COMFY_URL = "http://localhost:8188"

# LTX-2.3 model files
CHECKPOINT = "ltx-2.3-22b-distilled.safetensors"
TEXT_ENCODER = "gemma_3_12B_it.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
TEMPORAL_UPSCALER = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def build_prompt(image_path: str | None, text: str, negative: str,
                 frames: int, width: int, height: int, seed: int, fps: int,
                 cfg: float, steps: int, lora_strength: float,
                 two_stage: bool, upscale_cfg: float,
                 enable_audio: bool, img_strength: float) -> dict:
    """Build a ComfyUI API prompt for LTX-2.3."""

    prompt = {
        # Load checkpoint
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CHECKPOINT}},
        # Load text encoder
        "2": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": TEXT_ENCODER,
                         "ckpt_name": CHECKPOINT,
                         "device": "default"}},
        # Positive prompt
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": text, "clip": ["2", 0]}},
        # Negative prompt
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 0]}},
        # Conditioning
        "5": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                         "frame_rate": fps}},
        # Empty latent video
        "13": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height,
                           "length": frames, "batch_size": 1}},
    }

    # I2V: load, preprocess, and condition (NOT inplace — use ConditionOnly)
    latent_ref = ["13", 0]
    is_i2v = image_path is not None
    if is_i2v:
        image_name = os.path.basename(image_path)
        prompt.update({
            "6": {"class_type": "LoadImage",
                  "inputs": {"image": image_name}},
            "7": {"class_type": "ResizeImagesByLongerEdge",
                  "inputs": {"images": ["6", 0],
                             "longer_edge": max(width, height)}},
            "8": {"class_type": "LTXVPreprocess",
                  "inputs": {"image": ["7", 0], "img_compression": 18}},
            # ConditionOnly: conditions the ENTIRE generation to follow the
            # source image, not just inpaint frame 1
            "14": {"class_type": "LTXVImgToVideoConditionOnly",
                   "inputs": {"vae": ["1", 2], "image": ["8", 0],
                              "latent": ["13", 0],
                              "strength": img_strength, "bypass": False}},
        })
        latent_ref = ["14", 0]

    # Audio latent
    if enable_audio:
        prompt.update({
            "15": {"class_type": "LTXVAudioVAELoader",
                   "inputs": {"ckpt_name": CHECKPOINT}},
            "16": {"class_type": "LTXVEmptyLatentAudio",
                   "inputs": {"audio_vae": ["15", 0],
                              "frames_number": frames,
                              "frame_rate": fps, "batch_size": 1}},
            "17": {"class_type": "LTXVConcatAVLatent",
                   "inputs": {"video_latent": latent_ref,
                              "audio_latent": ["16", 0]}},
        })
        sample_latent_ref = ["17", 0]
    else:
        sample_latent_ref = latent_ref

    # LoRA (optional)
    model_ref = ["1", 0]
    if lora_strength > 0:
        prompt["18"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0],
                       "lora_name": DISTILLED_LORA,
                       "strength_model": lora_strength}
        }
        model_ref = ["18", 0]

    # First pass sampling
    # MultimodalGuider is only used when BOTH I2V AND audio are enabled
    # (it requires VIDEO + AUDIO modality parameters to work)
    use_multimodal = is_i2v and enable_audio

    prompt.update({
        "19": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": seed}},
        "20": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler_ancestral_cfg_pp" if use_multimodal else "euler"}},
        "21": {"class_type": "LTXVScheduler",
               "inputs": {"latent": sample_latent_ref, "steps": steps,
                           "max_shift": 2.05, "base_shift": 0.95,
                           "stretch": True, "terminal": 0.1}},
    })

    if use_multimodal:
        # MultimodalGuider with separate VIDEO and AUDIO parameters
        prompt.update({
            "40": {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "VIDEO",
                    "cfg": cfg,
                    "stg": 1,
                    "perturb_attn": True,
                    "rescale": 0.9,
                    "modality_scale": 3,
                    "skip_step": 0,
                    "cross_attn": True,
                }
            },
            "41": {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO",
                    "cfg": 7,
                    "stg": 1,
                    "perturb_attn": True,
                    "rescale": 0.7,
                    "modality_scale": 3,
                    "skip_step": 0,
                    "cross_attn": True,
                    "parameters": ["40", 0],
                }
            },
            "22": {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": model_ref,
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "parameters": ["41", 0],
                    "skip_blocks": "",
                }
            },
        })
    else:
        # CFGGuider for T2V or no-audio I2V
        prompt["22"] = {
            "class_type": "CFGGuider",
            "inputs": {"model": model_ref,
                        "positive": ["5", 0],
                        "negative": ["5", 1],
                        "cfg": cfg}
        }

    prompt["23"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["19", 0], "guider": ["22", 0],
                   "sampler": ["20", 0], "sigmas": ["21", 0],
                   "latent_image": sample_latent_ref}
    }

    # Separate AV latent if audio was used
    if enable_audio:
        prompt["24"] = {"class_type": "LTXVSeparateAVLatent",
                        "inputs": {"av_latent": ["23", 0]}}
        decode_source = "24"
    else:
        decode_source = "23"

    # Second stage: spatial upscale pass
    if two_stage:
        prompt.update({
            "25": {"class_type": "LTXVCropGuides",
                   "inputs": {"positive": ["5", 0], "negative": ["5", 1],
                              "latent": [decode_source, 0]}},
            "26": {"class_type": "LatentUpscaleModelLoader",
                   "inputs": {"model_name": SPATIAL_UPSCALER}},
            "27": {"class_type": "LTXVLatentUpsampler",
                   "inputs": {"samples": ["25", 2],
                              "upscale_model": ["26", 0],
                              "vae": ["1", 2]}},
        })

        # Re-apply I2V conditioning on upscaled latent if image was provided
        upscaled_video_ref = ["27", 0]
        if is_i2v:
            prompt["28"] = {
                "class_type": "LTXVImgToVideoConditionOnly",
                "inputs": {"vae": ["1", 2], "image": ["8", 0],
                           "latent": ["27", 0],
                           "strength": img_strength, "bypass": False}
            }
            upscaled_video_ref = ["28", 0]

        # Re-concat audio for second pass if audio enabled
        if enable_audio:
            prompt["29"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": upscaled_video_ref,
                           "audio_latent": [decode_source, 1]}
            }
            upscale_sample_ref = ["29", 0]
        else:
            upscale_sample_ref = upscaled_video_ref

        # Second pass sampling
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

    # Decode audio and create video with audio
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
                              "filename_prefix": "LTX23_gen",
                              "format": "mp4", "codec": "h264"}},
        })
    else:
        # Save as video without audio
        prompt["39"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {"images": ["36", 0],
                       "frame_rate": fps, "loop_count": 0,
                       "filename_prefix": "LTX23_gen",
                       "format": "video/h264-mp4",
                       "pingpong": False, "save_output": True}
        }

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Generate LTX-2.3 videos (text-to-video & image-to-video)")
    parser.add_argument("--image", default=None,
                        help="Input image path for I2V (relative to comfyui/input/)")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt describing the video")
    parser.add_argument("--negative",
                        default="blurry, low quality, watermark, overlay, titles, has blurbox",
                        help="Negative prompt")
    parser.add_argument("--frames", type=int, default=97,
                        help="Number of frames (must be 8n+1, e.g. 33, 65, 97, 121)")
    parser.add_argument("--width", type=int, default=768,
                        help="Width (divisible by 32)")
    parser.add_argument("--height", type=int, default=512,
                        help="Height (divisible by 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frame rate")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="CFG scale for first pass (3.0 recommended for I2V)")
    parser.add_argument("--steps", type=int, default=15,
                        help="Sampling steps")
    parser.add_argument("--lora", type=float, default=0.5,
                        help="Distilled LoRA strength (0 to disable)")
    parser.add_argument("--img-strength", type=float, default=0.7,
                        help="Image conditioning strength for I2V (0.0-1.0)")
    parser.add_argument("--two-stage", action="store_true",
                        help="Enable 2x spatial upscale (higher quality, slower)")
    parser.add_argument("--upscale-cfg", type=float, default=3.0,
                        help="CFG for upscale pass")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio generation")
    parser.add_argument("--url", default=COMFY_URL,
                        help="ComfyUI API URL")
    args = parser.parse_args()

    # Verify ComfyUI is reachable
    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"❌ Cannot reach ComfyUI at {args.url}")
        sys.exit(1)

    mode_str = "I2V" if args.image else "T2V"
    stage_str = "two-stage + 2x upscale" if args.two_stage else "single-stage"
    lora_str = f"LoRA {args.lora}" if args.lora > 0 else "no LoRA"
    audio_str = "with audio" if not args.no_audio else "no audio"

    prompt = build_prompt(
        image_path=args.image, text=args.prompt, negative=args.negative,
        frames=args.frames, width=args.width, height=args.height,
        seed=args.seed, fps=args.fps, cfg=args.cfg, steps=args.steps,
        lora_strength=args.lora, two_stage=args.two_stage,
        upscale_cfg=args.upscale_cfg, enable_audio=not args.no_audio,
        img_strength=args.img_strength
    )

    client_id = str(uuid.uuid4())
    print(f"🎬 LTX-2.3 | {mode_str} | {args.width}x{args.height}, {args.frames} frames")
    print(f"   {stage_str}, {lora_str}, {audio_str}")
    if args.image:
        print(f"   Image: {args.image} (strength={args.img_strength})")
    print(f"   Prompt: \"{args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}\"")

    resp = requests.post(f"{args.url}/api/prompt",
                         json={"prompt": prompt, "client_id": client_id})
    result = resp.json()

    if "error" in result:
        print(f"❌ {json.dumps(result, indent=2)}")
        sys.exit(1)
    if "node_errors" in result and result["node_errors"]:
        print(f"⚠️  {json.dumps(result['node_errors'], indent=2)}")
        sys.exit(1)

    prompt_id = result["prompt_id"]
    print(f"✅ Queued (ID: {prompt_id})")

    # Poll for completion
    start = time.time()
    while time.time() - start < 900:
        time.sleep(10)
        elapsed = time.time() - start
        try:
            h = requests.get(f"{args.url}/api/history/{prompt_id}",
                             timeout=5).json()
            if prompt_id in h:
                s = h[prompt_id]["status"]["status_str"]
                if s == "success":
                    print(f"\n✅ Done in {elapsed:.0f}s!")
                    for nid, no in h[prompt_id].get("outputs", {}).items():
                        for v in no.get("videos", no.get("gifs", [])):
                            fname = v.get("filename", "")
                            sub = v.get("subfolder", "")
                            path = f"output/{sub}/{fname}" if sub else f"output/{fname}"
                            full = os.path.join(os.path.dirname(__file__), path)
                            sz = os.path.getsize(full) / 1024 if os.path.exists(full) else 0
                            print(f"   📹 {full} ({sz:.0f}KB)")
                    return
                elif s == "error":
                    print(f"\n❌ Failed: {h[prompt_id]['status'].get('messages')}")
                    sys.exit(1)
            q = requests.get(f"{args.url}/api/queue", timeout=5).json()
            r = len(q.get("queue_running", []))
            p = len(q.get("queue_pending", []))
            print(f"  ⏳ [{elapsed:.0f}s] Running: {r}, Pending: {p}")
        except Exception as e:
            print(f"  [{elapsed:.0f}s] {e}")

    print("⚠️  Timed out after 15 minutes")
    sys.exit(1)


if __name__ == "__main__":
    main()
