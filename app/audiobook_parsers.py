"""
Book file parsers for EPUB and PDF import.
Extracts text and chapter structure from uploaded book files.
"""

import io
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger("speaker.audiobook.parsers")


@dataclass
class ParsedChapter:
    """A chapter extracted from a book file."""
    title: str
    text: str


@dataclass
class ParsedBook:
    """Result of parsing a book file."""
    title: str
    chapters: List[ParsedChapter]
    raw_text: str  # Full text for character detection


# ============================================================
# EPUB Parser
# ============================================================

def parse_epub(file_bytes: bytes) -> ParsedBook:
    """
    Parse an EPUB file into chapters with preserved structure.
    Uses ebooklib for EPUB reading and BeautifulSoup for HTML extraction.
    """
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(io.BytesIO(file_bytes))

    # Get book title
    title = "Untitled"
    dc_title = book.get_metadata("DC", "title")
    if dc_title:
        title = dc_title[0][0]

    chapters: List[ParsedChapter] = []
    full_text_parts: List[str] = []

    # Read spine items (ordered content)
    spine_items = []
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item and item.get_type() == ITEM_DOCUMENT:
            spine_items.append(item)

    # If spine is empty, fall back to all document items
    if not spine_items:
        spine_items = list(book.get_items_of_type(ITEM_DOCUMENT))

    for item in spine_items:
        html_content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "lxml")

        # Remove script, style, and nav elements
        for tag in soup.find_all(["script", "style", "nav"]):
            tag.decompose()

        # Try to find chapter title from headings
        chapter_title = None
        for heading_tag in ["h1", "h2", "h3"]:
            heading = soup.find(heading_tag)
            if heading:
                heading_text = heading.get_text(strip=True)
                if heading_text and len(heading_text) < 200:
                    chapter_title = heading_text
                    break

        # Extract text
        text = soup.get_text(separator="\n", strip=True)
        text = _clean_text(text)

        # Skip near-empty items (covers, copyright pages, etc.)
        if len(text.strip()) < 50:
            continue

        if not chapter_title:
            chapter_title = f"Chapter {len(chapters) + 1}"

        chapters.append(ParsedChapter(title=chapter_title, text=text))
        full_text_parts.append(text)

    # If we got no chapters, treat the whole thing as one
    if not chapters:
        all_text = "\n\n".join(full_text_parts) if full_text_parts else ""
        if all_text.strip():
            chapters = [ParsedChapter(title="Chapter 1", text=all_text)]

    raw_text = "\n\n".join(full_text_parts)
    logger.info(f"Parsed EPUB '{title}': {len(chapters)} chapters, {len(raw_text)} chars")
    return ParsedBook(title=title, chapters=chapters, raw_text=raw_text)


# ============================================================
# PDF Parser
# ============================================================

def parse_pdf(file_bytes: bytes) -> ParsedBook:
    """
    Parse a PDF file into chapters with heuristic chapter detection.
    Uses PyMuPDF (fitz) for text extraction.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")

    title = doc.metadata.get("title", "") or "Untitled"
    if title == "Untitled" or not title.strip():
        # Try to get title from first page
        if doc.page_count > 0:
            first_text = doc[0].get_text("text").strip()
            first_line = first_text.split("\n")[0].strip() if first_text else ""
            if first_line and len(first_line) < 100:
                title = first_line

    # Extract all text page by page
    pages_text: List[str] = []
    for page in doc:
        text = page.get_text("text")
        text = _clean_pdf_text(text)
        if text.strip():
            pages_text.append(text)

    doc.close()

    if not pages_text:
        return ParsedBook(title=title, chapters=[], raw_text="")

    full_text = "\n\n".join(pages_text)

    # Try to split into chapters using heading patterns
    chapters = _split_pdf_into_chapters(full_text)

    if not chapters:
        # Fall back: treat entire text as one chapter
        chapters = [ParsedChapter(title="Chapter 1", text=full_text)]

    logger.info(f"Parsed PDF '{title}': {len(chapters)} chapters, {len(full_text)} chars")
    return ParsedBook(title=title, chapters=chapters, raw_text=full_text)


def _split_pdf_into_chapters(text: str) -> List[ParsedChapter]:
    """
    Split PDF text into chapters using common heading patterns.
    Looks for lines like "Chapter 1", "CHAPTER ONE", "Part I", etc.
    """
    # Common chapter heading patterns
    chapter_patterns = [
        r"^(?:CHAPTER|Chapter)\s+\w+.*$",           # Chapter 1, Chapter One
        r"^(?:PART|Part)\s+\w+.*$",                  # Part I, Part One
        r"^\d+\.\s+\w+.*$",                          # 1. Title
        r"^[IVXLC]+\.\s+\w+.*$",                     # IV. Title
    ]

    combined = "|".join(f"({p})" for p in chapter_patterns)
    pattern = re.compile(combined, re.MULTILINE)

    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return []

    chapters: List[ParsedChapter] = []

    # Text before first chapter heading
    preamble = text[:matches[0].start()].strip()
    if preamble and len(preamble) > 100:
        chapters.append(ParsedChapter(title="Preamble", text=preamble))

    for i, match in enumerate(matches):
        chapter_title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()

        if chapter_text:
            chapters.append(ParsedChapter(title=chapter_title, text=chapter_text))

    return chapters


# ============================================================
# Text cleaning utilities
# ============================================================

def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove junk."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive blank lines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove common junk patterns
    text = re.sub(r"^\s*Page \d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


def _clean_pdf_text(text: str) -> str:
    """Additional cleaning for PDF-extracted text."""
    text = _clean_text(text)

    # Remove page numbers (standalone numbers on a line)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove common header/footer artifacts
    text = re.sub(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", text, flags=re.MULTILINE)

    # Re-join hyphenated words at line breaks (common in PDFs)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


# ============================================================
# Main dispatcher
# ============================================================

SUPPORTED_EXTENSIONS = {".epub", ".pdf", ".txt", ".text", ".md"}


def parse_uploaded_file(filename: str, file_bytes: bytes) -> ParsedBook:
    """
    Parse an uploaded book file. Dispatches to the appropriate parser
    based on file extension.
    
    Returns a ParsedBook with title, chapters, and raw text.
    
    Raises ValueError for unsupported formats.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == ".epub":
        return parse_epub(file_bytes)
    elif ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext in (".txt", ".text", ".md"):
        # Plain text: return as single chapter, let the existing parser handle splitting
        text = file_bytes.decode("utf-8", errors="replace")
        text = _clean_text(text)
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        return ParsedBook(
            title=title,
            chapters=[ParsedChapter(title="Full Text", text=text)],
            raw_text=text,
        )
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
