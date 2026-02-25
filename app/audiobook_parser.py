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

# Patterns for detecting dialogue and character attribution
DIALOGUE_PATTERNS = [
    # "dialogue," said Character  /  "dialogue," Character said
    r'"[^"]+"\s*,?\s*(?:said|asked|replied|exclaimed|whispered|shouted|muttered|cried|called|yelled|answered|remarked|continued|added|began|explained|insisted|suggested|demanded|declared|announced|agreed|admitted|argued|begged|complained|confessed|confirmed|denied|gasped|groaned|grumbled|hissed|howled|laughed|moaned|mumbled|murmured|observed|pleaded|promised|protested|questioned|recalled|repeated|responded|roared|sang|screamed|sighed|snapped|sobbed|stammered|stated|urged|warned|wondered)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
    # Character said, "dialogue"
    r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:said|asked|replied|exclaimed|whispered|shouted|muttered|cried|called|yelled|answered|remarked|continued|added|began|explained|insisted|suggested|demanded|declared|announced|agreed|admitted|argued|begged|complained|confessed|confirmed|denied|gasped|groaned|grumbled|hissed|howled|laughed|moaned|mumbled|murmured|observed|pleaded|promised|protested|questioned|recalled|repeated|responded|roared|sang|screamed|sighed|snapped|sobbed|stammered|stated|urged|warned|wondered)\s*,?\s*"[^"]+"',
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
    max_chars: int = 3000,
    min_chars: int = 100,
) -> List[Segment]:
    """
    Break chapter text into TTS-sized segments at natural boundaries.
    
    Strategy:
    1. Split on paragraph breaks first
    2. If a paragraph is too long, split on sentence boundaries
    3. Merge short paragraphs together up to max_chars
    """
    text = text.strip()
    if not text:
        return []
    
    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If only one paragraph (or no paragraph breaks), split on sentences
    if len(paragraphs) <= 1:
        paragraphs = _split_into_sentences(text)
    
    # Merge/split paragraphs into segments of appropriate size
    segments = []
    current_text = ""
    
    for para in paragraphs:
        # If this single paragraph exceeds max, split it by sentences
        if len(para) > max_chars:
            # Flush current
            if current_text.strip():
                segments.append(Segment(text=current_text.strip()))
                current_text = ""
            
            # Split long paragraph into sentence groups
            sentences = _split_into_sentences(para)
            for sent in sentences:
                if len(current_text) + len(sent) + 1 > max_chars and current_text.strip():
                    segments.append(Segment(text=current_text.strip()))
                    current_text = ""
                current_text += sent + " "
            
            if current_text.strip():
                segments.append(Segment(text=current_text.strip()))
                current_text = ""
        
        elif len(current_text) + len(para) + 2 > max_chars:
            # Would exceed limit — flush current, start new
            if current_text.strip():
                segments.append(Segment(text=current_text.strip()))
            current_text = para + "\n\n"
        
        else:
            current_text += para + "\n\n"
    
    # Flush remaining
    if current_text.strip():
        segments.append(Segment(text=current_text.strip()))
    
    # Merge segments that are too short with neighbors
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if len(merged[-1].text) < min_chars:
                merged[-1].text += "\n\n" + seg.text
            else:
                merged.append(seg)
        segments = merged
    
    return segments


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences at . ! ? boundaries."""
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


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
            # Check if this character speaks in this segment
            char_pattern = rf'(?:said|asked|replied|exclaimed|whispered|shouted)\s+{re.escape(char_name)}|{re.escape(char_name)}\s+(?:said|asked|replied|exclaimed|whispered|shouted)'
            if re.search(char_pattern, seg.text, re.IGNORECASE):
                seg.voice_name = voice_name
                seg.character = char_name
                assigned = True
                break
        
        if not assigned:
            seg.voice_name = narrator_voice
            seg.character = None
    
    return segments
