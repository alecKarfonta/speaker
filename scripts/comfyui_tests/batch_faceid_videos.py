#!/usr/bin/env python3
"""
Chain: FaceID scene images → LTXVImgToVideo.
Takes the 10 FaceID scene images (character already in scene) and animates them
using LTXVImgToVideo, so the character stays throughout the video.
"""
import asyncio, os, sys, time, logging
os.environ["COMFYUI_API_URL"] = "http://192.168.1.83:8188"
sys.path.insert(0, "/home/alec/git/speaker")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("faceid_video")

FACEID_DIR = "/home/alec/git/speaker/portrait_faceid_output"
VIDEO_DIR = "/home/alec/git/speaker/portrait_faceid_videos"

CHARACTERS = [
    {"name": "Elena_Blackwood",
     "scene": "A young woman in a dark leather coat stands in a dead rose garden at twilight, wind gently moves her auburn hair, a raven takes flight from a fountain, leaves drift slowly, dark fantasy, cinematic, slow atmospheric movement"},
    {"name": "Marcus_Thorne",
     "scene": "An elderly scholar in navy robes walks slowly through a grand library, his hand brushing along book spines, dust motes drift in golden lamplight, warm ambient atmosphere, gentle scholarly movement"},
    {"name": "Kira_Vasquez",
     "scene": "A woman in tactical black crouches on a rain-slicked rooftop, neon lights flicker and reflect in puddles, she looks at a glowing holographic tablet, rain falling gently, cyberpunk atmosphere, subtle movement"},
    {"name": "Aldric_Fenworth",
     "scene": "A red-bearded blacksmith raises a hammer and strikes a glowing sword on an anvil, sparks fly upward, forge fires flicker behind him, magma flows in channels, epic fantasy, dynamic forging motion"},
    {"name": "Yuki_Tanaka",
     "scene": "A woman in a white kimono walks gracefully on a wooden bridge, cherry blossom petals drift through morning mist, koi ripple the pond below, serene Japanese garden, gentle flowing movement"},
    {"name": "Darius_Stone",
     "scene": "A soldier in tactical gear moves cautiously through urban ruins, dust and smoke drift through beams of dawn light, rubble shifts underfoot, tense military atmosphere, deliberate careful movement"},
    {"name": "Seraphina_Glass",
     "scene": "An ethereal woman with silver hair raises her hand, arcane blue symbols spiral upward from her palm, bioluminescent mushrooms pulse around her, magical particles swirl, fantasy forest, enchanting magic casting"},
    {"name": "Rex_Callahan",
     "scene": "A cowboy in a leather duster walks down a dusty frontier street, tumbleweeds roll past, his coat sways in the wind, sunset light casts long shadows, Western atmosphere, steady walking gait"},
    {"name": "Nadia_Okafor",
     "scene": "A regal woman in vibrant garments surveys a futuristic city from a floating platform, wind catches her golden-threaded braids, solar ships glide past, warm sunset glow, afrofuturist, gentle observational moment"},
    {"name": "Viktor_Kozlov",
     "scene": "A stern man in a dark overcoat walks across a frozen lake, breath visible in cold air, ice cracks spread from his footsteps, birch forest in misty distance, bleak Russian winter, slow contemplative walking"},
]

async def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    from app.comfyui_service import upload_image, generate_video_with_ref

    results = []
    total = len(CHARACTERS)
    start_all = time.time()

    for i, char in enumerate(CHARACTERS, 1):
        name = char["name"]
        faceid_path = os.path.join(FACEID_DIR, f"faceid_{name}.png")

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{total}] {name}")
        logger.info(f"{'='*70}")

        if not os.path.exists(faceid_path):
            logger.error(f"  ✗ FaceID image not found: {faceid_path}")
            results.append({"name": name, "status": "MISSING"})
            continue

        start = time.time()

        # Upload the FaceID scene image (character already in scene)
        try:
            ref = await upload_image(faceid_path)
            logger.info(f"  ✓ Uploaded scene image: {ref}")
        except Exception as e:
            logger.error(f"  ✗ Upload FAILED: {e}")
            results.append({"name": name, "status": "UPLOAD_FAILED", "error": str(e)})
            continue

        # Generate video from FaceID scene image
        try:
            video_path = await generate_video_with_ref(
                prompt=char["scene"],
                ref_image_comfyui=ref,
                output_dir=VIDEO_DIR,
                prefix=f"video_{name}",
                duration=3.0,
            )
            kb = os.path.getsize(video_path) / 1024
            elapsed = time.time() - start
            logger.info(f"  ✓ Video: {os.path.basename(video_path)} ({kb:.0f} KB, {elapsed:.1f}s)")
            results.append({"name": name, "status": "OK", "path": video_path, "kb": kb, "time": elapsed})
        except Exception as e:
            logger.error(f"  ✗ Video FAILED: {e}")
            results.append({"name": name, "status": "FAILED", "error": str(e)})
            continue

    elapsed_all = time.time() - start_all
    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] != "OK"]
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH COMPLETE — {elapsed_all:.0f}s total")
    logger.info(f"  ✓ Success: {len(ok)}/{total}  |  ✗ Failed: {len(failed)}/{total}")
    if failed:
        for f in failed:
            logger.info(f"    - {f['name']}: {f['status']} — {f.get('error','')[:80]}")
    logger.info(f"\n  Output: {VIDEO_DIR}")
    for f in sorted(os.listdir(VIDEO_DIR)):
        sz = os.path.getsize(os.path.join(VIDEO_DIR, f)) / 1024
        logger.info(f"    {f:40s} {sz:6.0f} KB")

if __name__ == "__main__":
    asyncio.run(main())
