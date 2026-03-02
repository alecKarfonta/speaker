#!/usr/bin/env python3
"""
Standalone ComfyUI test — no dependencies beyond Python stdlib.
Generates an image and/or video from a text prompt and saves locally.

Usage:
    python3 scripts/test_comfyui.py --mode image --prompt "A stormy ocean at night"
    python3 scripts/test_comfyui.py --mode video --prompt "Camera pushes through a misty forest"
    python3 scripts/test_comfyui.py --mode both
    python3 scripts/test_comfyui.py --mode video --duration 3.5
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "comfyui_workflows")

DEFAULT_PROMPT = (
    "A dimly lit ancient library with towering shelves of rotting leather-bound books, "
    "candlelight flickering against mildewed stone walls, dust motes floating in amber light, "
    "gothic horror atmosphere, cinematic composition, oil painting style"
)


def api(url, method="GET", data=None, timeout=30):
    """Simple HTTP helper using only stdlib."""
    req = urllib.request.Request(url)
    if data is not None:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def download(url, path, timeout=60):
    """Download a binary file."""
    resp = urllib.request.urlopen(url, timeout=timeout)
    with open(path, "wb") as f:
        f.write(resp.read())


def load_workflow(name):
    path = os.path.join(WORKFLOWS_DIR, name)
    with open(path) as f:
        return json.load(f)


def inject(workflow, prompt, width, height, seed,
           output_width=None, output_height=None,
           video_length=None, fps=None):
    """Inject prompt/dims/seed into workflow nodes."""
    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})
        meta = node.get("_meta", {}).get("title", "").lower()

        if ct == "CLIPTextEncode":
            if "positive" in meta or inp.get("text") == "PLACEHOLDER_PROMPT":
                inp["text"] = prompt

        if ct == "EmptyLatentImage":
            inp["width"] = width
            inp["height"] = height

        if ct == "EmptyLTXVLatentVideo":
            inp["width"] = width
            inp["height"] = height
            if video_length:
                inp["length"] = video_length

        if "KSampler" in ct and "seed" in inp:
            inp["seed"] = seed

        if ct == "RandomNoise" and "noise_seed" in inp:
            inp["noise_seed"] = seed

        if ct == "ImageScale" and (output_width or output_height):
            if output_width:  inp["width"] = output_width
            if output_height: inp["height"] = output_height

        if fps is not None:
            if ct == "LTXVConditioning" and "frame_rate" in inp:
                inp["frame_rate"] = float(fps)
            if ct == "SaveAnimatedWEBP" and "fps" in inp:
                inp["fps"] = float(fps)

    return workflow


def main():
    p = argparse.ArgumentParser(description="Test ComfyUI generation (standalone, no deps)")
    p.add_argument("--url", default="http://192.168.1.83:8188", help="ComfyUI API URL")
    p.add_argument("--mode", choices=["image", "video", "both"], default="both")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default="/tmp/comfyui_test")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--frames", type=int, default=65)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--timeout", type=int, default=600, help="Max seconds to wait for generation")
    args = p.parse_args()

    base = args.url.rstrip("/")
    os.makedirs(args.output, exist_ok=True)

    print(f"ComfyUI:  {base}")
    print(f"Output:   {args.output}")
    print(f"Mode:     {args.mode}")
    print(f"Prompt:   {args.prompt[:80]}...")
    print()

    # Health check
    print("Health check... ", end="", flush=True)
    try:
        api(f"{base}/system_stats", timeout=5)
        print("OK ✅")
    except Exception as e:
        print(f"FAILED ❌  ({e})")
        sys.exit(1)

    # Free VRAM
    try:
        api(f"{base}/free", data={"unload_models": True, "free_memory": True}, timeout=10)
        print("Freed VRAM ✅")
    except Exception:
        pass

    def queue_and_wait(workflow, label):
        """Queue a workflow, poll until done, download output."""
        import uuid
        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        print(f"\n--- {label} ---")
        print("Queuing... ", end="", flush=True)
        resp = api(f"{base}/prompt", data=payload, timeout=30)
        pid = resp.get("prompt_id")
        if not pid:
            print(f"FAILED: {resp}")
            return
        print(f"prompt_id={pid}")

        # Poll
        deadline = time.time() + args.timeout
        print("Waiting for completion", end="", flush=True)
        while time.time() < deadline:
            time.sleep(3)
            print(".", end="", flush=True)
            try:
                hist = api(f"{base}/history/{pid}", timeout=10)
                if pid in hist:
                    print(" done!")
                    # Download
                    outputs = hist[pid].get("outputs", {})
                    for nid, nout in outputs.items():
                        for item in nout.get("images", []) + nout.get("gifs", []):
                            fname = item["filename"]
                            sub = item.get("subfolder", "")
                            dl_url = f"{base}/view?filename={fname}&subfolder={sub}&type=output"
                            ext = os.path.splitext(fname)[1] or ".png"
                            local = os.path.join(args.output, f"{label.lower().replace(' ', '_')}{ext}")
                            download(dl_url, local)
                            sz = os.path.getsize(local) / 1024
                            print(f"✅ Saved: {local} ({sz:.0f} KB)")
                            return local
                    print("⚠️  Job completed but no output files found")
                    return None
            except Exception:
                pass

        print(f"\n❌ Timed out after {args.timeout}s")
        return None

    seed = random.randint(0, 2**32 - 1)
    t_total = time.time()

    # Image
    if args.mode in ("image", "both"):
        w = args.width or 512
        h = args.height or 512
        out_w = args.width or 1024
        out_h = args.height or 1024
        print(f"Image: {w}x{h} base → {out_w}x{out_h} output")
        wf = load_workflow("text_to_image.json")
        wf = inject(wf, args.prompt, w, h, seed, output_width=out_w, output_height=out_h)
        t0 = time.time()
        result = queue_and_wait(wf, "Image")
        if result:
            print(f"   Time: {time.time() - t0:.1f}s")

    # Video
    if args.mode in ("video", "both"):
        w = args.width or 768
        h = args.height or 512
        frames = args.frames
        if args.duration:
            raw = int(args.duration * args.fps)
            frames = ((raw + 7) // 8) * 8 + 1
        frames = min(frames, 449)
        print(f"Video: {w}x{h}, {frames} frames @ {args.fps}fps = {frames/args.fps:.1f}s")
        wf = load_workflow("text_to_video.json")
        wf = inject(wf, args.prompt, w, h, seed, video_length=frames, fps=args.fps)
        t0 = time.time()
        result = queue_and_wait(wf, "Video")
        if result:
            print(f"   Time: {time.time() - t0:.1f}s")

    print(f"\nAll done! Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
