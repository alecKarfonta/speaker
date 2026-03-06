#!/usr/bin/env python3
"""
Batch test: 10 character portraits → 10 scene images using IPAdapter FaceID.
Character's face identity is preserved in each scene via SDXL + InsightFace.
"""
import asyncio, os, sys, time, logging
os.environ["COMFYUI_API_URL"] = "http://192.168.1.83:8188"
sys.path.insert(0, "/home/alec/git/speaker")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("batch_faceid")

PORTRAIT_DIR = "/home/alec/git/speaker/portrait_batch_images"
OUTPUT_DIR = "/home/alec/git/speaker/portrait_faceid_output"

CHARACTERS = [
    {"name": "Elena_Blackwood",
     "scene": "A young woman with auburn hair wearing a dark leather coat stands in a mystical dead rose garden at twilight, crumbling stone paths around her, a raven perched on a broken fountain, dramatic chiaroscuro lighting, dark fantasy oil painting, cinematic, full body shot"},
    {"name": "Marcus_Thorne",
     "scene": "An elderly bearded scholar in navy blue robes walks through a vast ancient library, running his fingers along leather-bound spines, golden lamplight casting warm amber glow, dust motes in the air, Renaissance oil painting aesthetic, full body shot"},
    {"name": "Kira_Vasquez",
     "scene": "A fierce woman in a tactical black jacket crouches on a rain-slicked cyberpunk rooftop at night, neon holographic billboards behind her, she holds a glowing data tablet, electric blue and magenta reflections in puddles, Blade Runner aesthetic, full body shot"},
    {"name": "Aldric_Fenworth",
     "scene": "A stocky red-bearded blacksmith in a leather apron hammers a glowing molten sword at an anvil inside a roaring underground forge, sparks flying, rivers of magma in stone channels, deep orange crimson tones, epic fantasy art, full body shot"},
    {"name": "Yuki_Tanaka",
     "scene": "A woman in a white silk kimono with cherry blossom embroidery walks across a curved wooden bridge in a Japanese garden, pink petals drifting through misty morning air, koi pond below, stone lanterns, soft pastel watercolor aesthetic, full body shot"},
    {"name": "Darius_Stone",
     "scene": "A muscular soldier in dark tactical gear moves cautiously through a bombed-out urban street at dawn, destroyed buildings and smoking rubble around him, golden morning light breaking through dust clouds, gritty photorealistic war photography, full body shot"},
    {"name": "Seraphina_Glass",
     "scene": "An ethereal woman with long silver-white hair and a flowing iridescent gown raises her hand in an ancient bioluminescent forest at midnight, arcane blue symbols spiral upward from her palm, glowing mushrooms and phosphorescent vines around her, fantasy digital art, full body shot"},
    {"name": "Rex_Callahan",
     "scene": "A rugged cowboy in a brown leather duster and red bandana walks down a dusty frontier main street, tumbleweeds rolling past abandoned saloons, his long shadow stretching across sun-baked earth, blazing orange sunset behind mountains, Spaghetti Western aesthetic, full body shot"},
    {"name": "Nadia_Okafor",
     "scene": "A regal woman in vibrant orange and teal garments with golden-threaded braids surveys a sprawling solarpunk city from a floating platform, solar sail ships gliding between glass towers wreathed in vertical gardens, warm sunset golden light, afrofuturist aesthetic, full body shot"},
    {"name": "Viktor_Kozlov",
     "scene": "A stern man in a dark charcoal overcoat walks alone across a vast frozen lake under heavy grey skies, distant birch forest on the horizon, cracked ice patterns underfoot, breath visible in bitter cold, bleak Russian winter, muted desaturated palette, cinematic, full body shot"},
]

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from app.comfyui_service import upload_image, generate_image_with_faceid

    results = []
    total = len(CHARACTERS)
    start_all = time.time()

    for i, char in enumerate(CHARACTERS, 1):
        name = char["name"]
        portrait_path = os.path.join(PORTRAIT_DIR, f"portrait_{name}.png")

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{total}] {name}")
        logger.info(f"{'='*70}")

        if not os.path.exists(portrait_path):
            logger.error(f"  ✗ Portrait not found: {portrait_path}")
            results.append({"name": name, "status": "NO_PORTRAIT"})
            continue

        start = time.time()

        # Upload portrait
        try:
            ref = await upload_image(portrait_path)
            logger.info(f"  ✓ Uploaded: {ref}")
        except Exception as e:
            logger.error(f"  ✗ Upload FAILED: {e}")
            results.append({"name": name, "status": "UPLOAD_FAILED", "error": str(e)})
            continue

        # Generate FaceID scene
        try:
            path = await generate_image_with_faceid(
                prompt=char["scene"],
                ref_image_comfyui=ref,
                output_dir=OUTPUT_DIR,
                prefix=f"faceid_{name}",
                face_weight=0.85,
            )
            kb = os.path.getsize(path) / 1024
            elapsed = time.time() - start
            logger.info(f"  ✓ Scene: {os.path.basename(path)} ({kb:.0f} KB, {elapsed:.1f}s)")
            results.append({"name": name, "status": "OK", "path": path, "kb": kb, "time": elapsed})
        except Exception as e:
            logger.error(f"  ✗ Scene FAILED: {e}")
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
    logger.info(f"\n  Output: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        logger.info(f"    {f:40s} {sz:6.0f} KB")

if __name__ == "__main__":
    asyncio.run(main())
