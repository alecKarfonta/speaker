#!/usr/bin/env python3
"""
ComfyUI-only test: portrait image → upload → scene video with ref.
Bypasses LLM (uses hardcoded prompts) to verify ComfyUI pipeline works.
"""
import asyncio
import os
import sys
import logging

os.environ["COMFYUI_API_URL"] = "http://192.168.1.83:8188"
sys.path.insert(0, "/home/alec/git/speaker")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("comfyui_test")

OUTPUT_DIR = "/tmp/portrait_e2e_output"

# Hardcoded prompts (what the LLM would produce)
PORTRAIT_PROMPT = (
    "Portrait, upper body, facing camera. A young woman of nineteen with long flowing auburn hair, "
    "sharp vivid green eyes, pale fair skin. She wears a dark brown leather coat over a grey tunic. "
    "Determined expression, windswept hair. Fantasy character portrait, detailed, high quality, "
    "dramatic lighting, dark moody background"
)

SCENE_PROMPT = (
    "A young auburn-haired woman in a dark leather coat pushes open a towering rusted iron gate. "
    "Beyond stretches an abandoned garden — dead rose bushes, crumbling stone paths, a broken fountain "
    "with a black raven perched on top. Overcast twilight, muted earth tones, dramatic chiaroscuro "
    "lighting, dark fantasy oil painting aesthetic, cinematic wide shot"
)


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output dir: {OUTPUT_DIR}")

    from app.comfyui_service import generate_image, upload_image, generate_video_with_ref

    # ---- Step 1: Generate portrait image ----
    logger.info("=" * 60)
    logger.info("STEP 1: Generating portrait IMAGE via ComfyUI...")
    logger.info(f"  Prompt: {PORTRAIT_PROMPT[:80]}...")
    portrait_path = await generate_image(
        prompt=PORTRAIT_PROMPT,
        output_dir=OUTPUT_DIR,
        prefix="portrait_elena",
    )
    size_kb = os.path.getsize(portrait_path) / 1024
    logger.info(f"  ✓ Portrait saved: {portrait_path} ({size_kb:.0f} KB)")
    print()

    # ---- Step 2: Upload portrait to ComfyUI ----
    logger.info("=" * 60)
    logger.info("STEP 2: Uploading portrait to ComfyUI as reference image...")
    comfyui_name = await upload_image(portrait_path)
    logger.info(f"  ✓ Uploaded as: {comfyui_name}")
    print()

    # ---- Step 3: Generate scene video with portrait reference ----
    logger.info("=" * 60)
    logger.info("STEP 3: Generating scene VIDEO with portrait reference...")
    logger.info(f"  Scene: {SCENE_PROMPT[:80]}...")
    logger.info(f"  Ref image: {comfyui_name}")
    video_path = await generate_video_with_ref(
        prompt=SCENE_PROMPT,
        ref_image_comfyui=comfyui_name,
        output_dir=OUTPUT_DIR,
        prefix="scene_elena",
        duration=3.0,
    )
    size_kb = os.path.getsize(video_path) / 1024
    logger.info(f"  ✓ Video saved: {video_path} ({size_kb:.0f} KB)")
    print()

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info(f"  📸 Portrait:  {portrait_path}")
    logger.info(f"  🎬 Video:     {video_path}")
    logger.info(f"  📂 Output:    {OUTPUT_DIR}")
    logger.info("  The video used the portrait as a reference image in LTXVImgToVideo")


if __name__ == "__main__":
    asyncio.run(main())
