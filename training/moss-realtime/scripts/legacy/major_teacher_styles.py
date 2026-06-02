"""MOSS v1.5 style profiles for Major (mature, direct) teacher synthesis."""

from __future__ import annotations

import copy
import random

OPENMOSS_TEACHER_BASE = {
    "text_temperature": 0.72,
    "text_top_p": 0.58,
    "text_top_k": 28,
    "audio_temperature": 0.58,
    "audio_top_p": 0.58,
    "audio_top_k": 28,
    "audio_repetition_penalty": 1.06,
}

STYLE_PROFILES: dict[str, dict] = {
    "calm": {
        "instruction": (
            "Calm, controlled adult female voice — steady pacing, clear diction, "
            "confident and professional."
        ),
        "sampling": {"audio_temperature": 0.55, "text_temperature": 0.70},
    },
    "direct": {
        "instruction": (
            "Direct, no-nonsense adult voice — concise delivery, firm tone, "
            "like a field briefing."
        ),
        "sampling": {"audio_temperature": 0.56, "text_temperature": 0.68},
    },
    "analytical": {
        "instruction": (
            "Analytical narrator voice — measured rhythm, thoughtful pauses, "
            "explaining complex ideas clearly."
        ),
        "sampling": {"audio_temperature": 0.54, "text_temperature": 0.66, "audio_top_p": 0.55},
    },
    "dry": {
        "instruction": (
            "Dry, understated adult voice — minimal drama, subtle irony, "
            "composed and precise."
        ),
        "sampling": {"audio_temperature": 0.52, "text_temperature": 0.65},
    },
    "urgent": {
        "instruction": (
            "Urgent but controlled voice — slightly faster pace, focused intensity, "
            "without shouting."
        ),
        "sampling": {"audio_temperature": 0.62, "text_temperature": 0.74, "audio_top_p": 0.60},
    },
    "reflective": {
        "instruction": (
            "Reflective, intimate adult voice — softer volume, contemplative mood, "
            "late-night conversation."
        ),
        "sampling": {"audio_temperature": 0.50, "text_temperature": 0.64},
    },
    "storytelling": {
        "instruction": (
            "Cinematic storyteller voice — vivid pacing, restrained emotion, "
            "engaging long-form narration."
        ),
        "sampling": {"audio_temperature": 0.57, "text_temperature": 0.71},
    },
}


def sampling_for_style(style: str) -> dict:
    out = copy.deepcopy(OPENMOSS_TEACHER_BASE)
    profile = STYLE_PROFILES.get(style, STYLE_PROFILES["calm"])
    out.update(profile.get("sampling", {}))
    return out


def instruction_for_style(style: str) -> str:
    return STYLE_PROFILES.get(style, STYLE_PROFILES["calm"])["instruction"]


def infer_style(text: str, length: str | None = None, rng: random.Random | None = None) -> str:
    t = text.lower()
    if any(w in t for w in ("report", "analysis", "data", "probability", "system", "protocol")):
        return "analytical"
    if any(w in t for w in ("now", "immediately", "move", "alert", "danger", "hurry")):
        return "urgent"
    if any(w in t for w in ("remember", "years ago", "used to", "wonder", "quiet")):
        return "reflective"
    if any(w in t for w in ("ironically", "obviously", "frankly", "anyway")):
        return "dry"
    if any(w in t for w in ("listen", "understand", "mission", "orders", "objective")):
        return "direct"
    if length in ("long", "medium_long"):
        return "storytelling"
    if length == "medium" and rng and rng.random() < 0.3:
        return rng.choice(["analytical", "direct", "calm"])
    return "calm"


def assign_row_styles(row: dict, rng: random.Random) -> dict:
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
