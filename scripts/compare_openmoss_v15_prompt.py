#!/usr/bin/env python3
"""Compare patched openmoss C++ prompt vs HuggingFace MOSS-TTS-v1.5 chat template."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoTokenizer

MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
REF_WAV = Path(__file__).resolve().parents[1] / "data/voices/loli/loli_15s.wav"
TEXT = "Hi everyone! Thanks for listening — hope you enjoy the story!"


def openmoss_user_inst(text: str, language: str = "English", reference_block: str = "") -> str:
    """Mirror openmoss build_user_inst() in pipeline.cpp."""
    ref_line = "None\n" if not reference_block else f"[S1]:\n{reference_block}\n"
    return (
        "<user_inst>\n"
        f"- Reference(s):\n{ref_line}"
        "- Instruction:\nNone\n"
        "- Tokens:\nNone\n"
        "- Quality:\nNone\n"
        "- Sound Event:\nNone\n"
        "- Ambient Sound:\nNone\n"
        f"- Language:\n{language}\n"
        f"- Text:\n{text}\n"
        "</user_inst>"
    )


def openmoss_reference_block(tok, d, ref_codes_n_frames: int) -> str:
    """Literal token-string reference block (codec frames not encoded here)."""
    audio_start = tok.decode([d["audio_start_token_id"]])
    audio_end = tok.decode([d["audio_end_token_id"]])
    user_slot = tok.decode([d["audio_user_slot_token_id"]])
    n_vq = d["n_vq"]
    s = audio_start
    s += user_slot * ref_codes_n_frames
    s += user_slot * (n_vq - 1)
    s += audio_end
    return s


def openmoss_prompt_v15_patched(tok, body: str) -> str:
    im_start = tok.decode([151644])
    im_end = tok.decode([151645])
    return im_start + "user\n" + body + im_end + "\n" + im_start + "assistant\n"


def openmoss_prompt_v10_stock(tok, body: str, d) -> str:
    im_start = tok.decode([151644])
    im_end = tok.decode([151645])
    audio_start = tok.decode([d["audio_start_token_id"]])
    return (
        im_start + "user\n" + body + im_end + "\n" + im_start + "assistant\n" + audio_start
    )


def diff_tokens(tok, a: str, b: str, label_a: str, label_b: str) -> None:
    ids_a = tok.encode(a, add_special_tokens=False)
    ids_b = tok.encode(b, add_special_tokens=False)
    print(f"\n=== {label_a} vs {label_b} ===")
    print(f"chars: {len(a)} vs {len(b)}")
    print(f"tokens: {len(ids_a)} vs {len(ids_b)}")
    print(f"equal: {ids_a == ids_b}")
    if ids_a != ids_b:
        n = min(len(ids_a), len(ids_b))
        for i in range(n):
            if ids_a[i] != ids_b[i]:
                print(f"first diff @ {i}: {ids_a[i]!r} vs {ids_b[i]!r}")
                print(f"  {label_a}: {tok.decode([ids_a[i]])!r}")
                print(f"  {label_b}: {tok.decode([ids_b[i]])!r}")
                lo, hi = max(0, i - 3), min(n, i + 4)
                print(f"  context {label_a}: {ids_a[lo:hi]} -> {[tok.decode([x]) for x in ids_a[lo:hi]]}")
                print(f"  context {label_b}: {ids_b[lo:hi]} -> {[tok.decode([x]) for x in ids_b[lo:hi]]}")
                break
        print(f"tail {label_a}: {ids_a[-6:]} -> {[tok.decode([x]) for x in ids_a[-6:]]}")
        print(f"tail {label_b}: {ids_b[-6:]} -> {[tok.decode([x]) for x in ids_b[-6:]]}")


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    d = proc.model_config.__dict__

    # Plain
    body_plain = openmoss_user_inst(TEXT)
    cpp_plain = openmoss_prompt_v15_patched(tok, body_plain)
    py_plain = tok.apply_chat_template(
        [{"role": "user", "content": body_plain}],
        add_generation_prompt=True,
        tokenize=False,
    )
    diff_tokens(tok, cpp_plain, py_plain, "openmoss_v15_patched", "python_v15")

    # Clone via processor (ground truth)
    ref_audio, sr = sf.read(str(REF_WAV), dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0)
    codes_list = proc.encode_audios_from_wav([ref_tensor], sr)
    t_ref = int(codes_list[0].shape[-1])
    n_vq = int(codes_list[0].shape[0])

    conv = [proc.build_user_message(text=TEXT, reference=codes_list, language="English")]
    py_clone = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    ref_block = openmoss_reference_block(tok, {"audio_start_token_id": d["audio_start_token_id"],
                                               "audio_end_token_id": d["audio_end_token_id"],
                                               "audio_user_slot_token_id": d["audio_user_slot_token_id"],
                                               "n_vq": n_vq}, t_ref)
    body_clone = openmoss_user_inst(TEXT, reference_block=ref_block)
    cpp_clone = openmoss_prompt_v15_patched(tok, body_clone)
    diff_tokens(tok, cpp_clone, py_clone, "openmoss_clone", "python_clone")

    # Stock v1.0-style prompt (prefilled audio_start) vs python — should differ from v1.5
    cpp_v10 = openmoss_prompt_v10_stock(tok, body_plain, d)
    diff_tokens(tok, cpp_v10, py_plain, "openmoss_v10_stock", "python_v15")

    print(f"\nreference frames encoded: {t_ref}, n_vq={n_vq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
