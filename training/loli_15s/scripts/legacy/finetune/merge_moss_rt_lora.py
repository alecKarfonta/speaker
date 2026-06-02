#!/usr/bin/env python3
"""Merge LoRA adapter into MOSS-TTS-Realtime base weights for faster inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import os

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[5]))
MOSS_REPO = ROOT / "third_party" / "MOSS-TTS"
DEFAULT_TRAIN_DIR = Path(
    os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training" / "loli_15s")
)
DEFAULT_CKPT = DEFAULT_TRAIN_DIR / "checkpoints/latest"
DEFAULT_OUT = DEFAULT_TRAIN_DIR / "exports/merged"
DEFAULT_BASE = "OpenMOSS-Team/MOSS-TTS-Realtime"


def _setup_imports() -> None:
    rt_dir = MOSS_REPO / "moss_tts_realtime"
    for entry in (str(MOSS_REPO), str(rt_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def main() -> int:
    p = argparse.ArgumentParser(description="Merge LoRA into MOSS-Realtime base")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = args.checkpoint.resolve()
    if not (ckpt / "adapter_model.safetensors").is_file():
        print(f"LoRA checkpoint not found: {ckpt}")
        return 1

    _setup_imports()
    from moss_tts_realtime.finetuning.lora_patch import apply_lora
    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime

    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {args.base} ...")
    model = MossTTSRealtime.from_pretrained(
        args.base,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    cfg = json.loads((ckpt / "adapter_config.json").read_text(encoding="utf-8"))
    model = apply_lora(
        model,
        r=int(cfg.get("r", 16)),
        alpha=int(cfg.get("lora_alpha", 32)),
        dropout=float(cfg.get("lora_dropout", 0.05)),
    )
    weights = load_file(str(ckpt / "adapter_model.safetensors"))
    remapped = {
        k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        ): v
        for k, v in weights.items()
    }
    model.load_state_dict(remapped, strict=False)

    print("Merging LoRA into base ...")
    merged = model.merge_and_unload()
    merged.eval()

    print(f"Saving merged model to {out} ...")
    merged.save_pretrained(str(out), safe_serialization=True)

    tok_src = ckpt if (ckpt / "tokenizer_config.json").is_file() else args.base
    print(f"Saving tokenizer from {tok_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_src))
    tokenizer.save_pretrained(str(out))

    meta = {
        "source_checkpoint": str(ckpt),
        "base_model": args.base,
        "merged_for": "MOSS_RT_NATIVE_VOICE inference without PeftModel",
    }
    (out / "merge_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.")
    print(f"\nServe with:\n  MOSS_RT_MODEL_ID={out} MOSS_RT_NATIVE_VOICE=true ./scripts/start-moss-realtime.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
