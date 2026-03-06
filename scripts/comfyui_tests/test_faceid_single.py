#!/usr/bin/env python3
"""Quick test: Single FaceID generation to verify the pipeline works."""
import asyncio, os, sys, time
os.environ["COMFYUI_API_URL"] = "http://192.168.1.83:8188"
sys.path.insert(0, "/home/alec/git/speaker")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test")

PORTRAIT = "/home/alec/git/speaker/portrait_batch_images/portrait_Elena_Blackwood.png"
OUTPUT = "/tmp/faceid_test"
SCENE = "A young woman with auburn hair in a dark leather coat stands in a mystical dead rose garden at twilight, crumbling stone paths, a raven on a broken fountain, dramatic chiaroscuro lighting, dark fantasy oil painting, cinematic"

async def main():
    os.makedirs(OUTPUT, exist_ok=True)
    from app.comfyui_service import upload_image, generate_image_with_faceid

    logger.info("Uploading portrait...")
    ref = await upload_image(PORTRAIT)
    logger.info(f"Uploaded: {ref}")

    logger.info("Generating scene with FaceID...")
    start = time.time()
    path = await generate_image_with_faceid(
        prompt=SCENE, ref_image_comfyui=ref,
        output_dir=OUTPUT, prefix="faceid_elena",
        face_weight=0.85,
    )
    elapsed = time.time() - start
    kb = os.path.getsize(path) / 1024
    logger.info(f"✓ Done in {elapsed:.1f}s: {path} ({kb:.0f} KB)")

if __name__ == "__main__":
    asyncio.run(main())
