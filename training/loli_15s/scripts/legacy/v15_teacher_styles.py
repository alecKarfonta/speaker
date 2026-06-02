"""MOSS v1.5 style profiles for varied teacher synthesis (instruction + sampling)."""

from __future__ import annotations

import copy
import random
import re

# Base tuned for loli clone; profiles nudge prosody per utterance type.
OPENMOSS_TEACHER_BASE = {
    "text_temperature": 0.75,
    "text_top_p": 0.6,
    "text_top_k": 30,
    "audio_temperature": 0.6,
    "audio_top_p": 0.6,
    "audio_top_k": 30,
    "audio_repetition_penalty": 1.08,
}

STYLE_PROFILES: dict[str, dict] = {
    "cheerful": {
        "instruction": (
            "Bright, upbeat young female voice — cheerful, lively, and friendly."
        ),
        "sampling": {"audio_temperature": 0.62, "text_temperature": 0.78},
    },
    "excited": {
        "instruction": (
            "Excited, energetic young girl — fast-paced, amazed, with bright emphasis."
        ),
        "sampling": {"audio_temperature": 0.68, "text_temperature": 0.82, "audio_top_p": 0.65},
    },
    "cozy": {
        "instruction": (
            "Warm, cozy young female voice — soft, comforting, like a bedtime story."
        ),
        "sampling": {"audio_temperature": 0.52, "text_temperature": 0.68, "audio_top_p": 0.55},
    },
    "storytelling": {
        "instruction": (
            "Expressive storyteller voice — clear pacing, gentle drama, engaging narration."
        ),
        "sampling": {"audio_temperature": 0.58, "text_temperature": 0.72, "audio_top_p": 0.58},
    },
    "curious": {
        "instruction": (
            "Curious, wondering young voice — light questioning tone, playful intrigue."
        ),
        "sampling": {"audio_temperature": 0.60, "text_temperature": 0.76},
    },
    "gentle": {
        "instruction": (
            "Gentle, reassuring young female voice — calm, supportive, and kind."
        ),
        "sampling": {"audio_temperature": 0.54, "text_temperature": 0.70},
    },
    "playful": {
        "instruction": (
            "Playful, teasing young girl voice — bouncy rhythm, fun and mischievous."
        ),
        "sampling": {"audio_temperature": 0.64, "text_temperature": 0.80},
    },
    "whisper_soft": {
        "instruction": (
            "Soft, intimate young voice — quiet and tender, as if sharing a secret."
        ),
        "sampling": {"audio_temperature": 0.50, "text_temperature": 0.65, "audio_top_p": 0.52},
    },
}


def sampling_for_style(style: str) -> dict:
    out = copy.deepcopy(OPENMOSS_TEACHER_BASE)
    profile = STYLE_PROFILES.get(style, STYLE_PROFILES["cheerful"])
    out.update(profile.get("sampling", {}))
    return out


def instruction_for_style(style: str) -> str:
    return STYLE_PROFILES.get(style, STYLE_PROFILES["cheerful"])["instruction"]


def infer_style(text: str, length: str | None = None, rng: random.Random | None = None) -> str:
    t = text.lower()
    if any(w in t for w in ("once upon", "story", "bunny", "festival", "adventure", "chapter")):
        return "storytelling"
    if any(w in t for w in ("!", "wow", "amazing", "yay", "we did it")):
        return "excited"
    if any(w in t for w in ("?", "what do you", "guess what", "can you", "tell me")):
        return "curious"
    if any(w in t for w in ("cozy", "bed", "sleep", "dream", "tea", "peaceful", "blanket", "lantern")):
        return "cozy"
    if any(w in t for w in ("nervous", "okay", "breath", "stronger", "it's okay")):
        return "gentle"
    if any(w in t for w in ("surprise", "fun", "party", "cookies", "map", "secret")):
        return "playful"
    if any(w in t for w in ("whisper", "quiet", "softly", "shh")):
        return "whisper_soft"
    if length == "long":
        return "storytelling"
    if length == "short" and rng and rng.random() < 0.35:
        return rng.choice(["excited", "playful", "curious"])
    return "cheerful"


def assign_row_styles(row: dict, rng: random.Random) -> dict:
    """Attach style + instruction fields in-place; return row."""
    if row["type"] == "single":
        style = infer_style(row["text"], row.get("length"), rng)
        row["style"] = style
        row["instruction"] = instruction_for_style(style)
    else:
        for turn in row["turns"]:
            if turn["role"] != "assistant":
                continue
            style = infer_style(turn["text"], None, rng)
            turn["style"] = style
            turn["instruction"] = instruction_for_style(style)
    return row
