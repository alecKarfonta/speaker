#!/usr/bin/env python3
"""One-off MOSS-SFX inference sanity check (run inside moss-sfx container)."""

import os
import sys

import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from moss_soundeffect_v2 import MossSoundEffectPipeline


def main() -> int:
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile missing", file=sys.stderr)
        return 1

    print(
        "cuda:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch:",
        torch.__version__,
        flush=True,
    )

    pipe = MossSoundEffectPipeline.from_pretrained(
        "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    eng = pipe.engine
    print(
        "BEFORE cast: dit",
        next(eng.dit.parameters()).dtype,
        "te",
        next(eng.text_encoder.parameters()).dtype,
        flush=True,
    )

    prompt = (
        "The crisp rhythmic click-clack of fast typing on a mechanical keyboard."
    )

    with torch.no_grad():
        audio = pipe(
            prompt=prompt,
            seconds=8,
            num_inference_steps=100,
            cfg_scale=4.0,
            seed=42,
        )
    print(
        "no cast: shape",
        tuple(audio.shape),
        "min",
        float(audio.min()),
        "max",
        float(audio.max()),
        "std",
        float(audio.std()),
        flush=True,
    )
    sf.write(
        "/tmp/sfx_no_cast.wav",
        audio[0, 0].detach().float().cpu().numpy(),
        pipe.sample_rate,
    )

    pipe.to(device="cuda", dtype=torch.bfloat16)
    print(
        "AFTER cast: dit",
        next(eng.dit.parameters()).dtype,
        flush=True,
    )
    with torch.no_grad():
        audio2 = pipe(
            prompt=prompt,
            seconds=8,
            num_inference_steps=100,
            cfg_scale=4.0,
            seed=42,
        )
    print(
        "after cast: std",
        float(audio2.std()),
        "max",
        float(audio2.max()),
        flush=True,
    )
    sf.write(
        "/tmp/sfx_cast.wav",
        audio2[0, 0].detach().float().cpu().numpy(),
        pipe.sample_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
