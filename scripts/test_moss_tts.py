#!/usr/bin/env python3
"""
MOSS-TTS Smoke Test
Loads the MossTTSDelay-8B model and generates a short English utterance.
Saves output WAV to /app/output/ (or ./output/ locally).
"""

import importlib.util
import os
import time
from pathlib import Path

import torch
import torchaudio

from transformers import AutoModel, AutoProcessor


def resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    """Pick the best attention backend available."""
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def main():
    model_id = os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/app/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_impl = resolve_attn_implementation(device, dtype)

    print("=" * 60)
    print("🌿 MOSS-TTS Smoke Test")
    print("=" * 60)
    print(f"  Model:     {model_id}")
    print(f"  Device:    {device}")
    print(f"  Dtype:     {dtype}")
    print(f"  Attention: {attn_impl}")
    print(f"  Output:    {output_dir}")
    print()

    # ── Disable broken cuDNN SDPA backend ──
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # ── Load processor ──
    print("⏳ Loading processor …")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    print(f"   Processor ready  ({time.perf_counter() - t0:.1f}s)")

    # ── Load model ──
    print("⏳ Loading model …")
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"   Model loaded     ({time.perf_counter() - t0:.1f}s)")

    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU memory used: {mem_gb:.2f} GB")

    # ── Generate test utterances ──
    test_texts = [
        "Hello, this is a test of the MOSS text to speech model. The quick brown fox jumps over the lazy dog.",
        "We stand on the threshold of the AI era. Artificial intelligence is no longer just a concept in laboratories.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\n🔊 Generating sample {i} …")
        print(f"   Text: {text[:80]}{'…' if len(text) > 80 else ''}")

        conversation = [processor.build_user_message(text=text)]
        batch = processor([conversation], mode="generation")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )
        gen_time = time.perf_counter() - t0

        for msg in processor.decode(outputs):
            audio = msg.audio_codes_list[0]
            sr = processor.model_config.sampling_rate
            out_path = output_dir / f"moss_sample_{i}.wav"
            torchaudio.save(str(out_path), audio.unsqueeze(0), sr)

            duration = audio.shape[-1] / sr
            rtf = gen_time / duration if duration > 0 else float("inf")
            print(f"   ✅ Saved: {out_path}")
            print(f"   Duration:  {duration:.2f}s")
            print(f"   Gen time:  {gen_time:.2f}s")
            print(f"   RTF:       {rtf:.2f}x")
            print(f"   Sample rate: {sr} Hz")

    print("\n" + "=" * 60)
    print("🌿 MOSS-TTS smoke test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
