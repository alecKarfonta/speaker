#!/usr/bin/env python3
"""
Batch test: 10 unique character portraits → 10 scene STILL IMAGES with portrait reference.
Uses img2img (VAE encode ref → partial denoise) to place each character in a unique scene.
"""
import asyncio
import os
import sys
import time
import logging

os.environ["COMFYUI_API_URL"] = "http://192.168.1.83:8188"
sys.path.insert(0, "/home/alec/git/speaker")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("batch")

OUTPUT_DIR = "/home/alec/git/speaker/portrait_batch_images"

CHARACTERS = [
    {
        "name": "Elena_Blackwood",
        "portrait": "Portrait, upper body, facing camera. Young woman, 19, long flowing auburn hair, sharp vivid green eyes, pale fair skin. Dark brown leather coat over grey tunic. Determined expression, windswept hair. Fantasy character, detailed, dramatic lighting, dark moody background, high quality",
        "scene": "A young auburn-haired woman in a dark leather coat stands before a towering rusted iron gate at twilight. Dead rose garden and crumbling stone paths beyond. A raven perches on a broken fountain. Dark fantasy oil painting, muted earth tones, dramatic chiaroscuro lighting, cinematic wide shot",
    },
    {
        "name": "Marcus_Thorne",
        "portrait": "Portrait, upper body, facing camera. Elderly male scholar, 65, silver beard neatly trimmed, warm brown eyes behind round wire spectacles. Weathered face with deep laugh lines. Navy blue professor robes with ink-stained cuffs. Wise gentle expression. Academic setting, warm candlelight, oil painting style, high quality",
        "scene": "An elderly bearded scholar in navy robes walks through a vast ancient library, tall bookshelves stretching to vaulted ceilings. Dust motes dance in shafts of golden lamplight. Leather-bound tomes and scattered scrolls. Warm amber tones, Renaissance oil painting aesthetic, atmospheric perspective",
    },
    {
        "name": "Kira_Vasquez",
        "portrait": "Portrait, upper body, facing camera. Hispanic woman, late 20s, short jet-black pixie cut, intense dark brown eyes, olive skin with a thin scar across left cheekbone. Tactical black jacket with raised collar. Fierce confident expression. Cyberpunk neon-lit background, cinematic, photorealistic, high quality",
        "scene": "A fierce woman with a pixie cut crouches on a rain-slicked rooftop overlooking a neon-drenched cyberpunk cityscape at night. Holographic billboards reflect in puddles. She holds a glowing data tablet. Electric blue and magenta color palette, Blade Runner aesthetic, volumetric fog, cinematic",
    },
    {
        "name": "Aldric_Fenworth",
        "portrait": "Portrait, upper body, facing camera. Male dwarf blacksmith, 40s, broad muscular build, thick copper-red braided beard, deep blue eyes, ruddy complexion with soot marks. Leather smith's apron over bare muscular arms. Proud grin. Forge lighting, fantasy art, detailed, high quality",
        "scene": "A stocky red-bearded blacksmith hammers a glowing molten sword on an anvil inside a roaring underground forge. Rivers of magma flow in channels along stone walls. Sparks fly in dramatic arcs. Deep orange and crimson tones, epic fantasy, dramatic underlighting, painterly style",
    },
    {
        "name": "Yuki_Tanaka",
        "portrait": "Portrait, upper body, facing camera. Japanese woman, early 30s, long straight black hair, serene dark eyes, porcelain skin. White and pale blue silk kimono with subtle cherry blossom embroidery. Calm contemplative expression. Soft diffused light, Japanese watercolor aesthetic, ethereal, high quality",
        "scene": "A woman in a flowing white kimono stands on a curved wooden bridge over a koi pond in a Japanese garden during cherry blossom season. Pink petals drift through misty morning air. Stone lanterns, maple trees, tranquil water. Soft pastel palette, ukiyo-e meets watercolor, serene",
    },
    {
        "name": "Darius_Stone",
        "portrait": "Portrait, upper body, facing camera. Black man, mid 35, shaved head, strong jaw, intense amber eyes, deep brown skin, short trimmed goatee. Military tactical vest over black shirt. Battle-hardened expression with subtle compassion. Studio lighting, photorealistic, cinematic portrait, high quality",
        "scene": "A muscular soldier in tactical gear leads a squad through a bombed-out urban street at dawn. Destroyed buildings, smoking rubble, overturned vehicles. Golden morning light breaks through dust clouds. Gritty photorealistic war photography, desaturated with warm highlights, intense and somber",
    },
    {
        "name": "Seraphina_Glass",
        "portrait": "Portrait, upper body, facing camera. Ethereal elven woman, timeless appearance, floor-length silver-white hair, luminous pale violet eyes, alabaster skin with faint bioluminescent freckles. Flowing iridescent gossamer gown. Enigmatic otherworldly smile. Moonlit forest background, fantasy digital art, glowing particles, high quality",
        "scene": "An ethereal silver-haired elf stands in a clearing of an ancient bioluminescent forest at midnight. Glowing mushrooms, phosphorescent vines, floating light motes surround her. She raises her hand and arcane symbols spiral upward in blue light. Fantasy digital art, deep blues and silvers, magical atmosphere",
    },
    {
        "name": "Rex_Callahan",
        "portrait": "Portrait, upper body, facing camera. Rugged male, late 40s, sun-weathered tan skin, shaggy salt-and-pepper hair, steel-grey eyes, strong nose. Brown leather duster coat, faded red bandana around neck. Squinting with wry half-smile. Western sunset backdrop, cinematic, high quality",
        "scene": "A rugged cowboy in a leather duster stands alone on a dusty main street of a frontier ghost town at golden hour. Tumbleweeds roll past abandoned wooden buildings. Mountains silhouetted against a blazing orange sunset. Spaghetti Western aesthetic, cinematic composition, warm golden tones",
    },
    {
        "name": "Nadia_Okafor",
        "portrait": "Portrait, upper body, facing camera. Nigerian woman, mid 25, rich dark skin, intricate braided updo with gold thread, warm deep brown eyes, full lips. Vibrant ankara-print garment in orange and teal. Radiant confident smile. Warm studio lighting, afrofuturist aesthetic, regal, high quality",
        "scene": "A regal woman in vibrant draped garments stands atop a futuristic floating platform above a sprawling African-inspired solarpunk city. Curved glass towers, lush vertical gardens, solar sail ships gliding between buildings. Sunset golden light. Afrofuturist aesthetic, warm vibrant palette, utopian, cinematic",
    },
    {
        "name": "Viktor_Kozlov",
        "portrait": "Portrait, upper body, facing camera. Russian man, early 50s, sharp angular features, cropped iron-grey hair, cold pale blue eyes, thin lips, clean-shaven, prominent cheekbones. Black turtleneck under charcoal wool overcoat. Calculating intense stare. Cold blue-grey lighting, noir aesthetic, film grain, high quality",
        "scene": "A stern man in a dark overcoat walks alone across a vast frozen lake under heavy grey skies. Distant birch forest on the horizon. Cracked ice patterns underfoot, breath visible in bitter cold. Bleak Russian winter, Tarkovsky aesthetic, muted desaturated palette, contemplative, cinematic",
    },
]


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from app.comfyui_service import generate_image, upload_image, generate_image_with_ref

    results = []
    total = len(CHARACTERS)
    start_all = time.time()

    for i, char in enumerate(CHARACTERS, 1):
        name = char["name"]
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{total}] {name}")
        logger.info(f"{'='*70}")

        start = time.time()

        # --- Portrait image ---
        logger.info(f"  Generating portrait image...")
        try:
            portrait_path = await generate_image(
                prompt=char["portrait"],
                output_dir=OUTPUT_DIR,
                prefix=f"portrait_{name}",
            )
            portrait_kb = os.path.getsize(portrait_path) / 1024
            logger.info(f"  ✓ Portrait: {os.path.basename(portrait_path)} ({portrait_kb:.0f} KB)")
        except Exception as e:
            logger.error(f"  ✗ Portrait FAILED: {e}")
            results.append({"name": name, "status": "PORTRAIT_FAILED", "error": str(e)})
            continue

        # --- Upload portrait to ComfyUI ---
        logger.info(f"  Uploading to ComfyUI...")
        try:
            comfyui_name = await upload_image(portrait_path)
            logger.info(f"  ✓ Uploaded: {comfyui_name}")
        except Exception as e:
            logger.error(f"  ✗ Upload FAILED: {e}")
            results.append({"name": name, "status": "UPLOAD_FAILED", "portrait": portrait_path, "error": str(e)})
            continue

        # --- Scene image with portrait reference ---
        logger.info(f"  Generating scene IMAGE with portrait reference...")
        try:
            scene_path = await generate_image_with_ref(
                prompt=char["scene"],
                ref_image_comfyui=comfyui_name,
                output_dir=OUTPUT_DIR,
                prefix=f"scene_{name}",
                denoise=0.65,
            )
            scene_kb = os.path.getsize(scene_path) / 1024
            elapsed = time.time() - start
            logger.info(f"  ✓ Scene: {os.path.basename(scene_path)} ({scene_kb:.0f} KB)")
            logger.info(f"  ⏱ {elapsed:.1f}s")
            results.append({
                "name": name, "status": "OK",
                "portrait": portrait_path, "scene": scene_path,
                "portrait_kb": portrait_kb, "scene_kb": scene_kb,
                "time_s": elapsed,
            })
        except Exception as e:
            logger.error(f"  ✗ Scene FAILED: {e}")
            results.append({"name": name, "status": "SCENE_FAILED", "portrait": portrait_path, "error": str(e)})
            continue

    # --- Summary ---
    elapsed_all = time.time() - start_all
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH COMPLETE — {elapsed_all:.0f}s total")
    logger.info(f"{'='*70}")

    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] != "OK"]
    logger.info(f"  ✓ Success: {len(ok)}/{total}")
    if failed:
        logger.info(f"  ✗ Failed:  {len(failed)}/{total}")
        for f in failed:
            logger.info(f"    - {f['name']}: {f['status']} — {f.get('error','')[:80]}")

    logger.info(f"\n  Output: {OUTPUT_DIR}")
    logger.info(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        logger.info(f"    {f:40s} {sz:6.0f} KB")


if __name__ == "__main__":
    asyncio.run(main())
