"""
Text parsing utilities for the audiobook generator.
Handles chapter splitting, segment chunking, and character detection.
"""

import re
from typing import List, Tuple, Optional
from .audiobook_models import Chapter, Segment


# Common chapter heading patterns (no inline flags — use re.MULTILINE in code)
CHAPTER_PATTERNS = {
    "auto": None,  # Try all patterns
    "chapter_number": r"^(?:Chapter|CHAPTER)\s+\d+[^\n]*",
    "chapter_word": r"^(?:Chapter|CHAPTER)\s+[A-Z][^\n]*",
    "part": r"^(?:Part|PART)\s+\d+[^\n]*",
    "separator": r"^(?:---+|\*\*\*+|===+)\s*$",
    "roman": r"^(?:Chapter|CHAPTER)\s+[IVXLCDM]+[^\n]*",
    "numbered_dot": r"^\d+\.\s+[A-Z][^\n]*",
}

# Full set of speech/dialogue attribution verbs (shared between detection and assignment)
SPEECH_VERBS = (
    "said|asked|replied|exclaimed|whispered|shouted|muttered|cried|called|yelled|"
    "answered|remarked|continued|added|began|explained|insisted|suggested|demanded|"
    "declared|announced|agreed|admitted|argued|begged|complained|confessed|confirmed|"
    "denied|gasped|groaned|grumbled|hissed|howled|laughed|moaned|mumbled|murmured|"
    "observed|pleaded|promised|protested|questioned|recalled|repeated|responded|"
    "roared|sang|screamed|sighed|snapped|sobbed|stammered|stated|urged|warned|wondered"
)

# Patterns for detecting dialogue and character attribution
DIALOGUE_PATTERNS = [
    # "dialogue," said Character  /  "dialogue," Character said
    rf'"[^"]+"\s*,?\s*(?:{SPEECH_VERBS})\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
    # Character said, "dialogue"
    rf'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:{SPEECH_VERBS})\s*,?\s*"[^"]+"',
]

# Words that look like character names but aren't
FALSE_POSITIVE_NAMES = {
    "The", "Then", "There", "They", "This", "That", "Those", "These",
    "But", "And", "She", "He", "His", "Her", "Its", "Our", "Your",
    "What", "When", "Where", "Why", "How", "Who", "Which",
    "Just", "Still", "Even", "Now", "Yet", "Also", "Only",  
    "Chapter", "Part", "Book", "Section", "Page",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
}


def detect_chapter_pattern(text: str) -> Optional[str]:
    """Auto-detect the most likely chapter pattern in the text."""
    best_pattern = None
    best_count = 0
    
    for name, pattern in CHAPTER_PATTERNS.items():
        if name == "auto" or pattern is None:
            continue
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) > best_count and len(matches) >= 2:
            best_count = len(matches)
            best_pattern = name
    
    return best_pattern


def parse_book_text(text: str, chapter_pattern: str = "auto") -> List[Chapter]:
    """
    Split raw text into chapters.
    
    Args:
        text: Full book text
        chapter_pattern: Pattern name from CHAPTER_PATTERNS, a custom regex, or "auto"
    
    Returns:
        List of Chapter objects with segments
    """
    text = text.strip()
    if not text:
        return []
    
    # Resolve pattern
    if chapter_pattern == "auto":
        detected = detect_chapter_pattern(text)
        if detected:
            pattern = CHAPTER_PATTERNS[detected]
        else:
            # No chapters detected — treat entire text as one chapter
            chapter = Chapter(index=0, title="Full Text")
            chapter.segments = split_chapter_into_segments(text)
            return [chapter]
    elif chapter_pattern in CHAPTER_PATTERNS:
        pattern = CHAPTER_PATTERNS[chapter_pattern]
    else:
        # Custom regex
        pattern = chapter_pattern
    
    if not pattern:
        chapter = Chapter(index=0, title="Full Text")
        chapter.segments = split_chapter_into_segments(text)
        return [chapter]
    
    # Use re.split with captured group to keep the headings
    # Pass re.MULTILINE flag to make ^ and $ match line boundaries
    splits = re.split(f"({pattern})", text, flags=re.MULTILINE)
    
    chapters = []
    
    # Handle text before first chapter heading
    if splits[0].strip():
        ch = Chapter(index=0, title="Prologue")
        ch.segments = split_chapter_into_segments(splits[0].strip())
        if ch.segments:
            chapters.append(ch)
    
    # Process heading + content pairs
    i = 1
    while i < len(splits):
        heading = splits[i].strip() if i < len(splits) else ""
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        
        if content or heading:
            ch = Chapter(
                index=len(chapters),
                title=heading,
            )
            ch.segments = split_chapter_into_segments(content) if content else []
            chapters.append(ch)
        
        i += 2
    
    # If no chapters were created, treat as single chapter
    if not chapters:
        ch = Chapter(index=0, title="Full Text")
        ch.segments = split_chapter_into_segments(text)
        chapters.append(ch)
    
    return chapters


def split_chapter_into_segments(
    text: str,
    max_chars: int = 800,
    min_chars: int = 100,
) -> List[Segment]:
    """
    Break chapter text into TTS-sized segments at natural boundaries.
    
    Strategy:
    1. Normalize text — rejoin PDF line breaks, keep true paragraph breaks
    2. Split into paragraphs (double newline separated)
    3. For each paragraph:
       - If it fits in max_chars, make it a segment
       - If too long, split at sentence boundaries into sub-segments
    4. Never split mid-sentence
    
    Target: ~100-200 words per segment (~15-30s of TTS audio)
    """
    text = _normalize_text_for_segmentation(text)
    if not text:
        return []
    
    # Split into paragraphs on double newlines
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If still only one big block (common in PDFs), try to recover paragraphs
    # by splitting at sentence boundaries  
    if len(paragraphs) <= 1 and len(text) > max_chars:
        paragraphs = _recover_paragraphs(text, target_size=max_chars // 2)
    
    segments = []
    
    for para in paragraphs:
        if len(para) <= max_chars:
            # Paragraph fits, add as segment
            segments.append(Segment(text=para))
        else:
            # Split the oversized paragraph at sentence boundaries
            sentences = _split_into_sentences(para)
            current_text = ""
            for sent in sentences:
                if len(current_text) + len(sent) + 1 > max_chars and current_text.strip():
                    segments.append(Segment(text=current_text.strip()))
                    current_text = ""
                current_text += sent + " "
            
            if current_text.strip():
                segments.append(Segment(text=current_text.strip()))

    # Optional: merge very short segments if they are consecutive and from the same paragraph split
    # But to respect paragraphs, avoid merging across paragraph boundaries
    # For now, keep as is to preserve pacing

    return segments


def _normalize_text_for_segmentation(text: str) -> str:
    """
    Normalize text for segmentation:
    - Rejoin lines broken by PDF column layout (single newlines within paragraphs)
    - Preserve true paragraph breaks (double newlines)
    """
    text = text.strip()
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ---------------------------------------------------------------
    # Heal word-wrap artifacts BEFORE marking paragraph boundaries.
    # This prevents a mid-word break like "p\naragraphs" from being
    # locked into a paragraph split when surrounded by double newlines.
    # ---------------------------------------------------------------

    # 1. Re-join hyphenated word breaks  (conti-\ntinued → continued)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. Re-join non-hyphenated line breaks with a space.
    #    A line ending with a lowercase letter/digit immediately followed
    #    by a line starting with a lowercase letter is a wrapped line.
    #    Insert a space to prevent word mashing.
    #    e.g. "the\nbottom" → "the bottom"  (NOT "thebottom")
    #    NOTE: Use \n (NOT \n\n?) so we only rejoin single-newline word-wrap,
    #    never double-newline paragraph breaks.
    text = re.sub(r"([a-z0-9])\n([a-z])", r"\1 \2", text)

    # ---------------------------------------------------------------
    # Now it is safe to mark true paragraph breaks and join the rest.
    # ---------------------------------------------------------------

    # Preserve true paragraph breaks (double+ newlines) with a placeholder
    text = re.sub(r"\n\s*\n", "\n<PARA>\n", text)

    # Replace ALL remaining single newlines with spaces
    text = re.sub(r"\n", " ", text)

    # Restore paragraph breaks
    text = text.replace("<PARA>", "\n\n")

    # Clean up multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Re-normalize multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _recover_paragraphs(text: str, target_size: int = 1500) -> List[str]:
    """
    For text with no paragraph breaks, split into paragraph-like chunks
    at sentence boundaries near the target_size.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text]
    
    paragraphs = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) + 1 > target_size and current:
            paragraphs.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip() if current else sent
    
    if current.strip():
        paragraphs.append(current.strip())
    
    return paragraphs if paragraphs else [text]


# Abbreviations and patterns that should NOT trigger a sentence split
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col", "maj",
    "rev", "hon", "pres", "rep", "sen",
    "vs", "etc", "approx", "dept", "est", "fig", "inc", "ltd", "co",
    "vol", "ed", "trans", "illus", "feat",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "no", "nos", "op", "cit", "seq", "al", "viz",
    "a.m", "p.m", "e.g", "i.e", "cf", "ibid",
    "u.s", "u.k", "u.n",
}


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences with proper handling of:
    - Abbreviations (Mr., Dr., etc.)
    - Initials (J. K. Rowling)
    - Decimal numbers (3.14)
    - Ellipses (...)
    - Quoted speech ending a sentence
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    sentences = []
    current = ""
    i = 0
    
    while i < len(text):
        char = text[i]
        current += char
        
        # Check for sentence-ending punctuation
        if char in ".!?":
            # Look ahead to decide if this is a real sentence break
            
            # Handle ellipsis (... or …)
            if char == "." and i + 1 < len(text) and text[i + 1] == ".":
                i += 1
                continue
            
            # Check for closing quotes/parens after the punctuation
            j = i + 1
            while j < len(text) and text[j] in '"\'")\u201d\u2019':
                current += text[j]
                j += 1
            
            # Check if what follows looks like a new sentence
            # (whitespace then capital letter or end of text)
            rest = text[j:]
            if not rest.strip():
                # End of text
                sentences.append(current.strip())
                current = ""
                i = j
                continue
            
            # Check for whitespace followed by uppercase = new sentence
            ws_match = re.match(r"\s+", rest)
            if ws_match:
                after_ws = rest[ws_match.end():]
                if after_ws and after_ws[0].isupper():
                    # Check if the word before the period is an abbreviation
                    word_before = re.search(r"(\w+)\.$", current)
                    if word_before and word_before.group(1).lower() in _ABBREVIATIONS:
                        # This is an abbreviation, not a sentence end
                        i = j
                        continue
                    
                    # Check for single-letter initial (A. B. Smith)
                    initial_match = re.search(r"\b[A-Z]\.$", current)
                    if initial_match:
                        # Single letter initial — not a sentence break
                        i = j
                        continue
                    
                    # This is a real sentence break
                    sentences.append(current.strip())
                    current = ""
                    i = j
                    continue
            
            # No whitespace after punctuation or followed by lowercase — not a sentence end
            i = j
            continue
        
        i += 1
    
    # Flush remaining text
    if current.strip():
        sentences.append(current.strip())
    
    return sentences


def detect_characters(text: str) -> List[str]:
    """
    Detect character names from dialogue patterns in the text.
    
    Returns a deduplicated, sorted list of character names.
    """
    characters = set()
    
    for pattern in DIALOGUE_PATTERNS:
        matches = re.findall(pattern, text)
        for name in matches:
            name = name.strip()
            if name and name not in FALSE_POSITIVE_NAMES and len(name) > 1:
                characters.add(name)
    
    return sorted(characters)


def assign_segment_voices(
    segments: List[Segment],
    character_map: dict,
    narrator_voice: str,
) -> List[Segment]:
    """
    Assign voices to segments based on character detection.
    
    For each segment:
    - If it contains dialogue from a mapped character, assign that character's voice
    - Otherwise assign the narrator voice
    """
    for seg in segments:
        # Check if segment is primarily dialogue from a known character
        assigned = False
        for char_name, voice_name in character_map.items():
            # Check if this character speaks in this segment (uses full speech verb set)
            char_pattern = rf'(?:{SPEECH_VERBS})\s+{re.escape(char_name)}|{re.escape(char_name)}\s+(?:{SPEECH_VERBS})'
            if re.search(char_pattern, seg.text, re.IGNORECASE):
                seg.voice_name = voice_name
                seg.character = char_name
                assigned = True
                break
        
        if not assigned:
            seg.voice_name = narrator_voice
            seg.character = None
    
    return segments
