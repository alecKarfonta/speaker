"""
AI-powered character detection and voice matching for the audiobook generator.
Uses an OpenAI-compatible LLM endpoint to analyze book text.
"""

import json
import logging
import re
import requests
from typing import Dict, List, Optional, Any

from .config import settings

logger = logging.getLogger("speaker.audiobook.llm")

# ============================================================
# Prompt templates
# ============================================================

CHARACTER_ANALYSIS_PROMPT = """You are a book analysis assistant. Analyze the following book text and identify all speaking characters.

For each character, provide:
1. Their name (exactly as it appears in the text)
2. A short description (1 sentence)
3. Voice traits: estimated age range, gender, and vocal quality (e.g., "deep", "soft", "raspy", "cheerful")

Also identify the narrator's voice style.

Available TTS voices to assign from:
{voices}

Respond ONLY with valid JSON in this exact format, no other text:
```json
{{
  "characters": [
    {{
      "name": "Character Name",
      "description": "Brief description of the character",
      "voice_traits": {{
        "age": "young/middle-aged/elderly",
        "gender": "male/female",
        "quality": "deep/soft/raspy/cheerful/authoritative/gentle"
      }},
      "suggested_voice": "best_matching_voice_name_from_available_list"
    }}
  ],
  "narrator": {{
    "description": "Narrator style description",
    "suggested_voice": "best_matching_voice_name_from_available_list"
  }}
}}
```

Book text (first 8000 characters):
---
{text}
---"""

EMOTION_ANALYSIS_PROMPT = """Analyze the emotional tone of this text passage and respond with ONLY a single word from this list: neutral, happy, sad, angry, fearful, surprised, disgusted, contemptuous.

Text: "{text}"

Emotion:"""

SCENE_PROMPT_TEMPLATE = """You are a visual scene describer for an audiobook illustration system. Given a passage from a book, describe a cinematic visual scene that could illustrate this passage.

Rules:
- Describe the setting, lighting, colors, mood, and camera angle
- Be specific and vivid — this will be used as an image generation prompt
- Do NOT include any text, words, letters, or UI elements in the scene
- Do NOT describe dialogue or speech — only the visual environment and action
- Keep it to 1-2 sentences, under 100 words
- Focus on atmosphere and composition
- IMPORTANT: Maintain visual consistency with the established style

Chapter: {chapter_title}
Character: {character}
Emotion: {emotion}
{style_section}{context_section}
Passage: "{text}"

Scene description:"""

VISUAL_STYLE_TEMPLATE = """Analyze this book text and generate a consistent visual art style description that should be used for ALL illustrations in this audiobook.

Consider:
- What artistic medium fits the genre? (oil painting, watercolor, digital art, anime, photorealistic, etc.)
- What color palette suits the mood? (warm earth tones, cool blues, vibrant, muted, etc.)
- What lighting style? (cinematic, soft diffused, dramatic chiaroscuro, golden hour, etc.)
- What overall aesthetic? (fantasy, gothic, minimalist, baroque, noir, etc.)

Respond with ONLY the style description — a single line of comma-separated attributes, under 50 words. Example: "oil painting, warm earth tones, golden hour lighting, fantasy aesthetic, soft brush strokes, cinematic composition"

Book text:
---
{text}
---

Visual style:"""


# ============================================================
# LLM API Client
# ============================================================

def _call_llm(
    prompt: str,
    system_message: str = "You are a helpful literary analysis assistant.",
    max_tokens: Optional[int] = None,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call the OpenAI-compatible LLM endpoint.
    Returns the response text, or None if the call fails.
    """
    url = settings.instruct_llm_api_url
    api_key = settings.instruct_llm_api_key
    model = settings.instruct_llm_model
    max_tok = max_tokens or settings.instruct_llm_max_tokens

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tok,
        "temperature": temperature,
    }

    try:
        logger.info(f"Calling LLM at {url} (model={model})")
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Strip thinking blocks if present (Qwen3 thinking model)
        content = _strip_thinking_blocks(content)

        logger.info(f"LLM response received ({len(content)} chars)")
        return content

    except requests.exceptions.ConnectionError:
        logger.warning(f"LLM endpoint unreachable at {url}")
        return None
    except requests.exceptions.Timeout:
        logger.warning("LLM request timed out")
        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 thinking model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not extract JSON from LLM response: {text[:200]}...")
    return None


# ============================================================
# Public API
# ============================================================

def check_llm_available() -> bool:
    """Check if the LLM endpoint is reachable."""
    try:
        url = settings.instruct_llm_api_url.replace("/chat/completions", "/models")
        resp = requests.get(url, timeout=5, headers={
            "Authorization": f"Bearer {settings.instruct_llm_api_key}"
        } if settings.instruct_llm_api_key else {})
        return resp.status_code == 200
    except Exception:
        return False


def analyze_characters(
    text: str,
    available_voices: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to analyze book text and identify characters with voice suggestions.

    Returns dict with:
    - characters: list of {name, description, voice_traits, suggested_voice}
    - narrator: {description, suggested_voice}
    - character_voice_map: dict of character_name -> suggested_voice

    Returns None if LLM is unavailable or analysis fails.
    """
    # Truncate text to ~8000 chars for the prompt (enough for character detection)
    truncated = text[:8000]

    voices_str = ", ".join(available_voices) if available_voices else "No voices available"

    prompt = CHARACTER_ANALYSIS_PROMPT.format(
        voices=voices_str,
        text=truncated,
    )

    response = _call_llm(prompt, max_tokens=4000, temperature=0.3)
    if not response:
        return None

    parsed = _extract_json(response)
    if not parsed:
        return None

    # Validate and sanitize the response
    try:
        characters = parsed.get("characters", [])
        narrator = parsed.get("narrator", {})

        # Build character_voice_map from suggestions
        character_voice_map = {}
        character_descriptions = {}
        detected_characters = []

        for char in characters:
            name = char.get("name", "").strip()
            if not name:
                continue

            detected_characters.append(name)
            description = char.get("description", "")
            suggested_voice = char.get("suggested_voice", "")

            # Validate suggested voice exists
            if suggested_voice and suggested_voice in available_voices:
                character_voice_map[name] = suggested_voice
            elif available_voices:
                # Assign a voice round-robin if suggestion is invalid
                idx = len(character_voice_map) % len(available_voices)
                character_voice_map[name] = available_voices[idx]

            if description:
                # Include voice traits in description
                traits = char.get("voice_traits", {})
                trait_str = ""
                if traits:
                    parts = []
                    if traits.get("age"):
                        parts.append(traits["age"])
                    if traits.get("gender"):
                        parts.append(traits["gender"])
                    if traits.get("quality"):
                        parts.append(traits["quality"])
                    trait_str = f" ({', '.join(parts)})" if parts else ""
                character_descriptions[name] = f"{description}{trait_str}"

        narrator_voice = narrator.get("suggested_voice", "")
        if narrator_voice and narrator_voice not in available_voices:
            narrator_voice = available_voices[0] if available_voices else ""

        return {
            "characters": characters,
            "detected_characters": detected_characters,
            "character_voice_map": character_voice_map,
            "character_descriptions": character_descriptions,
            "narrator_voice": narrator_voice,
            "narrator_description": narrator.get("description", ""),
        }

    except Exception as e:
        logger.error(f"Error processing LLM character analysis: {e}")
        return None


def detect_segment_emotion(text: str) -> Optional[str]:
    """
    Detect the emotional tone of a text segment.
    Returns an emotion string or None.
    """
    if len(text) < 20:
        return None

    prompt = EMOTION_ANALYSIS_PROMPT.format(text=text[:500])
    response = _call_llm(prompt, max_tokens=20, temperature=0.1)

    if not response:
        return None

    # Clean and validate
    emotion = response.strip().lower().rstrip(".")
    valid_emotions = {"neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "contemptuous"}
    if emotion in valid_emotions:
        return emotion

    # Try to find a valid emotion in the response
    for e in valid_emotions:
        if e in emotion:
            return e

    return None


def generate_scene_prompt(
    text: str,
    chapter_title: str = "",
    character: str = "Narrator",
    emotion: str = "neutral",
    visual_style: str = "",
    prev_scene: str = "",
    next_text: str = "",
) -> Optional[str]:
    """
    Generate a cinematic scene description from segment text.
    Returns a concise prompt suitable for image/video generation.

    Args:
        visual_style: Project-level style string for consistency
        prev_scene: The scene prompt from the previous segment (for continuity)
        next_text: Text of the next segment (for anticipating scene transitions)
    """
    # Build optional sections
    style_section = ""
    if visual_style:
        style_section = f"\nVisual Style (maintain this): {visual_style}\n"

    context_section = ""
    context_parts = []
    if prev_scene:
        context_parts.append(f"Previous scene: {prev_scene}")
    if next_text:
        context_parts.append(f"Next passage preview: {next_text[:200]}")
    if context_parts:
        context_section = "\nContext for continuity:\n" + "\n".join(context_parts) + "\n"

    prompt = SCENE_PROMPT_TEMPLATE.format(
        text=text[:800],
        chapter_title=chapter_title or "Unknown",
        character=character or "Narrator",
        emotion=emotion or "neutral",
        style_section=style_section,
        context_section=context_section,
    )
    response = _call_llm(
        prompt,
        system_message="You are a visual scene describer. Respond with only the scene description, nothing else.",
        max_tokens=150,
        temperature=0.5,
    )
    if response:
        scene = response.strip().strip('"').strip()
        # Append style to the scene prompt so it's included in the ComfyUI prompt
        if visual_style and visual_style.lower() not in scene.lower():
            scene = f"{scene}, {visual_style}"
        return scene
    return None


def generate_visual_style(text: str) -> Optional[str]:
    """
    Analyze book text and generate a consistent visual style description.
    This style is stored at the project level and appended to all scene prompts.
    """
    # Use first ~2000 chars to capture the genre/tone
    sample = text[:2000]
    prompt = VISUAL_STYLE_TEMPLATE.format(text=sample)
    response = _call_llm(
        prompt,
        system_message="You are a visual art director. Respond with only the style description.",
        max_tokens=80,
        temperature=0.4,
    )
    if response:
        style = response.strip().strip('"').strip()
        return style
    return None
