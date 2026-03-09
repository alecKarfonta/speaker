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
      "visual_profile": {{
        "face": "eyes, hair, facial hair, structure",
        "skin": "complexion, texture, scars",
        "build": "body type, height, posture",
        "clothing": "typical attire, accessories"
      }},
      "voice_description": "Detailed description of the character's voice (e.g. 'A deep, raspy voice of an old man' or 'A bright, cheerful young girl')",
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
    "voice_description": "Detailed description of the narrator's voice",
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

# variant for video/I2V — focuses on action/dynamics, no art style tags
SCENE_PROMPT_VIDEO_TEMPLATE = """You are a visual scene describer for an audiobook video system. Given a passage from a book, describe a short cinematic video scene.

Rules:
- Focus on CHARACTER ACTIONS and CAMERA MOVEMENT (e.g. "slowly turns", "camera pulls back")
- Describe what is HAPPENING, not art style or aesthetics
- Be specific about motion: walking, gesturing, wind blowing, light flickering
- Do NOT include art style keywords (no "oil painting", "watercolor", etc.)
- Do NOT include quality tags (no "masterpiece", "high quality", etc.)
- Keep it to 1-2 sentences, under 80 words

Chapter: {chapter_title}
Character: {character}
Emotion: {emotion}
{context_section}
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
    """Remove <think>...</think> blocks from Qwen3 thinking model output.
    If stripping leaves nothing, try to extract useful content from inside the blocks."""
    # Try stripping closed think blocks
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped:
        return stripped

    # If stripping left nothing, the actual content might be inside the think block
    # or after an unclosed <think> tag
    think_match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.DOTALL)
    if think_match:
        inner = think_match.group(1).strip()
        # Look for JSON inside the think block
        json_match = re.search(r'[\[{].*[}\]]', inner, flags=re.DOTALL)
        if json_match:
            logger.info("Extracted JSON from inside <think> block")
            return json_match.group()

    # Also handle unclosed <think> — content after it
    if "<think>" in text and "</think>" not in text:
        after = text.split("<think>", 1)[0].strip()
        if after:
            return after

    # Last resort: return original text with tags stripped
    return re.sub(r"</?think>", "", text).strip()


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
        url = settings.instruct_llm_api_url
        headers = {"Content-Type": "application/json"}
        if settings.instruct_llm_api_key:
            headers["Authorization"] = f"Bearer {settings.instruct_llm_api_key}"
        # Send a minimal chat request — cheaper than /v1/models which 404s on some servers
        resp = requests.post(url, json={
            "model": settings.instruct_llm_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }, headers=headers, timeout=10)
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
        character_visual_profiles = {}
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
                # Use explicit voice description if provided, otherwise build from traits
                voice_desc = char.get("voice_description", "")
                if not voice_desc:
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
                    voice_desc = f"{description}{trait_str}"
                
                character_descriptions[name] = voice_desc

            if "visual_profile" in char:
                character_visual_profiles[name] = char["visual_profile"]


        narrator_voice = narrator.get("suggested_voice", "")
        if narrator_voice and narrator_voice not in available_voices:
            narrator_voice = available_voices[0] if available_voices else ""
            
        narrator_desc = narrator.get("voice_description", "") or narrator.get("description", "")

        return {
            "characters": characters,
            "detected_characters": detected_characters,
            "character_voice_map": character_voice_map,
            "character_descriptions": character_descriptions,
            "character_visual_profiles": character_visual_profiles,
            "narrator_voice": narrator_voice,
            "narrator_description": narrator_desc,
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
    response = _call_llm(prompt, max_tokens=500, temperature=0.1)

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
    for_video: bool = False,
) -> Optional[str]:
    """
    Generate a cinematic scene description from segment text.
    Returns a concise prompt suitable for image/video generation.

    Args:
        visual_style: Project-level style string for consistency
        prev_scene: The scene prompt from the previous segment (for continuity)
        next_text: Text of the next segment (for anticipating scene transitions)
        for_video: If True, optimize prompt for I2V (action-focused, no style tags)
    """
    # Build optional sections
    context_section = ""
    context_parts = []
    if prev_scene:
        context_parts.append(f"Previous scene: {prev_scene}")
    if next_text:
        context_parts.append(f"Next passage preview: {next_text[:200]}")
    if context_parts:
        context_section = "\nContext for continuity:\n" + "\n".join(context_parts) + "\n"

    if for_video:
        # Video mode: action-focused prompt, no style tags
        prompt = SCENE_PROMPT_VIDEO_TEMPLATE.format(
            text=text[:800],
            chapter_title=chapter_title or "Unknown",
            character=character or "Narrator",
            emotion=emotion or "neutral",
            context_section=context_section,
        )
    else:
        # Image mode: include visual style
        style_section = ""
        if visual_style:
            style_section = f"\nVisual Style (maintain this): {visual_style}\n"
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
        max_tokens=1000,
        temperature=0.5,
    )
    if response:
        scene = response.strip().strip('"').strip()
        # Only append style for image mode (style tags hurt I2V quality)
        if not for_video and visual_style and visual_style.lower() not in scene.lower():
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
        max_tokens=1000,
        temperature=0.4,
    )
    if response:
        style = response.strip().strip('"').strip()
        return style
    return None


# ============================================================
# Character appearance extraction
# ============================================================

CHARACTER_EXTRACTION_TEMPLATE = """Analyze the following book text and identify all important named characters.
For each character, provide:
1. A detailed physical appearance description suitable for generating portrait images.
   Include: approximate age, gender, hair color/style, eye color, skin tone, build/height, distinguishing features, and typical clothing/attire.
2. A portrait generation prompt optimized for AI image generation (Stable Diffusion style).
   Format: single paragraph, include "portrait, upper body, facing camera" and quality keywords.
3. A voice description for a text-to-speech voice generator.
   Describe: gender, approximate age, vocal quality (deep/soft/raspy/breathy/warm/cold), speaking pace, accent if applicable, emotional tone.

Be specific and visual — these descriptions will be used to generate consistent character portraits and voices.

Book text (excerpt):
{text}

Respond ONLY with valid JSON, no other text:
```json
[
  {{
    "name": "Character Name",
    "description": "Detailed physical appearance: age, gender, hair, eyes, build, clothing, distinguishing features",
    "portrait_prompt": "portrait, upper body, facing camera, [age] [gender] with [hair] and [eyes], [clothing], [style keywords], masterpiece, high quality, detailed",
    "voice_prompt": "A [quality] [gender] voice, approximately [age] years old, [pace] speaking pace, [tone] tone, [accent if any]"
  }}
]
```"""


def extract_characters(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Analyze book text and extract named characters with detailed visual descriptions.
    Returns a list of dicts with 'name' and 'description' keys.
    """
    prompt = CHARACTER_EXTRACTION_TEMPLATE.format(text=text)
    response = _call_llm(
        prompt,
        system_message="You are a literary analyst specializing in visual character descriptions. Respond with only valid JSON.",
        max_tokens=4000,
        temperature=0.3,
    )
    if not response:
        return None

    # Parse JSON from response
    def _try_parse_characters(text: str):
        """Try to parse character list from text, with multiple fallback strategies."""
        # Strip markdown code fences if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to extract JSON array
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return None
        
        raw = json_match.group()
        
        # Try direct parsing first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Fix trailing commas before ] or }
        fixed = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix unescaped newlines in strings
        fixed2 = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group().replace('\n', ' '), fixed, flags=re.DOTALL)
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass
        
        # Last resort: extract individual objects with regex
        objects = re.findall(r'\{[^{}]*\}', raw, re.DOTALL)
        if objects:
            results = []
            for obj_str in objects:
                try:
                    obj_str = re.sub(r',\s*}', '}', obj_str)
                    results.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    continue
            if results:
                return results
        
        return None

    try:
        characters = _try_parse_characters(response)
        if characters:
            result = []
            for c in characters:
                if isinstance(c, dict) and "name" in c:
                    entry = {
                        "name": c["name"],
                        "description": c.get("description", ""),
                    }
                    if c.get("portrait_prompt"):
                        entry["portrait_prompt"] = c["portrait_prompt"]
                    if c.get("voice_prompt"):
                        entry["voice_prompt"] = c["voice_prompt"]
                    result.append(entry)
            if result:
                logger.info(f"Extracted {len(result)} characters from book text")
                return result
            else:
                logger.warning(f"No valid characters in parsed JSON: {characters[:200]}")
        else:
            logger.warning(f"Could not parse JSON from LLM response: {response[:300]}")
    except Exception as e:
        logger.warning(f"Failed to parse character extraction: {e}")

    return None


PORTRAIT_PROMPT_TEMPLATE = """Convert this character description into a concise image generation prompt for a character portrait.
The prompt should be optimized for AI image generation (Stable Diffusion style).
Focus on the most visually distinctive features.
Format: a single paragraph, no line breaks, include "portrait, upper body, facing camera" and a quality suffix.

Character: {name}
Description: {description}

Respond with ONLY the image prompt, nothing else."""


def generate_portrait_prompt(name: str, description: str) -> Optional[str]:
    """
    Convert a character's appearance description into an optimized image generation prompt.
    """
    prompt = PORTRAIT_PROMPT_TEMPLATE.format(name=name, description=description)
    response = _call_llm(
        prompt,
        system_message="You are an expert at writing image generation prompts. Respond with only the prompt.",
        max_tokens=1000,
        temperature=0.4,
    )
    if response:
        return response.strip().strip('"').strip()
    return None


NARRATOR_VOICE_PROMPT_TEMPLATE = """Analyze this book text and describe the ideal narrator voice for reading this book aloud as an audiobook.

Consider the book's genre, tone, setting, and target audience.

Describe the narrator voice with these attributes:
- Gender (male/female/androgynous)
- Approximate age (young adult, middle-aged, elderly)
- Vocal quality (warm, authoritative, gentle, dramatic, gravelly, silky, etc.)
- Speaking pace (slow and measured, moderate, brisk and energetic)
- Emotional tone (calm, passionate, mysterious, cheerful, somber)
- Accent or dialect if the setting suggests one (British, Southern American, neutral, etc.)

Respond with ONLY the voice description — a single paragraph under 80 words. Do NOT include the book title or author.

Book text (excerpt):
---
{text}
---

Narrator voice description:"""


def generate_narrator_voice_prompt(text: str) -> Optional[str]:
    """
    Analyze book text and generate a voice description for the narrator.
    Returns a concise voice description suitable for MOSS VoiceGenerator.
    """
    # Use first ~2000 chars to capture genre/tone
    sample = text[:2000]
    prompt = NARRATOR_VOICE_PROMPT_TEMPLATE.format(text=sample)
    response = _call_llm(
        prompt,
        system_message="You are an audiobook casting director. Respond with only the voice description.",
        max_tokens=500,
        temperature=0.4,
    )
    if response:
        desc = response.strip().strip('"').strip()
        logger.info(f"Generated narrator voice prompt: {desc[:100]}...")
        return desc
    return None


