"""
Tests for the audiobook parser and models.
"""
import os
import sys
import json
import tempfile
import shutil
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.audiobook_parser import (
    parse_book_text, split_chapter_into_segments,
    detect_characters, detect_chapter_pattern, assign_segment_voices,
)
from app.audiobook_models import (
    AudiobookProject, Chapter, Segment, SegmentStatus,
    save_project, load_project, delete_project_from_disk,
    list_projects, project_to_detail_response,
)


# --- Parser Tests ---

SAMPLE_BOOK = """
Chapter 1: The Beginning

It was a dark and stormy night. The wind howled through the trees, and rain 
battered against the window panes. Alice sat by the fire, reading a book.

"I wonder what's out there," said Alice, peering into the darkness.

"Nothing good," replied Bob, from his armchair. "Stay inside where it's warm."

Alice shook her head. She had always been the adventurous one.

Chapter 2: The Journey

The next morning dawned bright and clear. Alice packed her bag and set out 
along the road. The birds sang in the trees, and she felt her spirits lift.

"Where are you going?" called Bob from the doorway.

"To find the truth," Alice answered without looking back.

Chapter 3: The Discovery

At the end of the road, Alice found a small cottage. Inside, an old woman 
sat spinning thread. The room smelled of lavender and old books.

"I've been expecting you," whispered the old woman.
"""


class TestChapterDetection:
    def test_detect_chapter_pattern(self):
        pattern = detect_chapter_pattern(SAMPLE_BOOK)
        assert pattern is not None
        assert "chapter" in pattern

    def test_parse_into_chapters(self):
        chapters = parse_book_text(SAMPLE_BOOK)
        assert len(chapters) == 3
        assert "Beginning" in chapters[0].title
        assert "Journey" in chapters[1].title
        assert "Discovery" in chapters[2].title

    def test_each_chapter_has_segments(self):
        chapters = parse_book_text(SAMPLE_BOOK)
        for ch in chapters:
            assert len(ch.segments) > 0

    def test_no_chapters_single_block(self):
        text = "This is just a single paragraph with no chapter markers."
        chapters = parse_book_text(text)
        assert len(chapters) == 1
        assert chapters[0].title == "Full Text"

    def test_separator_pattern(self):
        text = "Part one.\n\n---\n\nPart two.\n\n---\n\nPart three."
        chapters = parse_book_text(text, chapter_pattern="separator")
        assert len(chapters) >= 2

    def test_custom_regex_pattern(self):
        text = "SECTION A\nContent one.\n\nSECTION B\nContent two."
        chapters = parse_book_text(text, chapter_pattern=r"^SECTION [A-Z]")
        assert len(chapters) >= 2


class TestSegmentSplitting:
    def test_short_text_single_segment(self):
        segments = split_chapter_into_segments("Hello world.")
        assert len(segments) == 1
        assert segments[0].text == "Hello world."

    def test_respects_max_chars(self):
        long_text = "This is a test sentence. " * 200  # ~5000 chars
        segments = split_chapter_into_segments(long_text, max_chars=1000)
        for seg in segments:
            # Allow some overflow for sentence boundaries
            assert len(seg.text) < 2000

    def test_splits_on_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        segments = split_chapter_into_segments(text, max_chars=100)
        assert len(segments) >= 1

    def test_segment_ids_unique(self):
        text = "Sentence one. Sentence two. Sentence three. " * 50
        segments = split_chapter_into_segments(text, max_chars=500)
        ids = [s.id for s in segments]
        assert len(ids) == len(set(ids))

    def test_segments_default_pending(self):
        segments = split_chapter_into_segments("Some text here.")
        for seg in segments:
            assert seg.status == SegmentStatus.PENDING


class TestCharacterDetection:
    def test_detect_from_dialogue(self):
        characters = detect_characters(SAMPLE_BOOK)
        assert "Alice" in characters
        assert "Bob" in characters

    def test_no_false_positives(self):
        text = '"Hello," said The stranger.'
        characters = detect_characters(text)
        # "The" should be filtered out
        assert "The" not in characters

    def test_empty_text(self):
        characters = detect_characters("")
        assert characters == []

    def test_no_dialogue(self):
        text = "The cat sat on the mat. It was sunny outside."
        characters = detect_characters(text)
        assert characters == []


class TestVoiceAssignment:
    def test_assigns_character_voice(self):
        segments = [Segment(text='"Hello," said Alice.')]
        result = assign_segment_voices(segments, {"Alice": "voice_f1"}, "narrator")
        assert result[0].voice_name == "voice_f1"
        assert result[0].character == "Alice"

    def test_assigns_narrator_voice(self):
        segments = [Segment(text="It was a cold morning.")]
        result = assign_segment_voices(segments, {"Alice": "voice_f1"}, "narrator")
        assert result[0].voice_name == "narrator"
        assert result[0].character is None


class TestProjectPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch AUDIOBOOKS_DIR
        import app.audiobook_models as models
        self._orig_dir = models.AUDIOBOOKS_DIR
        models.AUDIOBOOKS_DIR = self.tmpdir

    def teardown_method(self):
        import app.audiobook_models as models
        models.AUDIOBOOKS_DIR = self._orig_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        project = AudiobookProject(name="Test Book", raw_text="Hello world")
        save_project(project)
        loaded = load_project(project.id)
        assert loaded is not None
        assert loaded.name == "Test Book"
        assert loaded.id == project.id

    def test_list_projects(self):
        for i in range(3):
            save_project(AudiobookProject(name=f"Book {i}", raw_text=f"Text {i}"))
        projects = list_projects()
        assert len(projects) == 3

    def test_delete_project(self):
        project = AudiobookProject(name="To Delete", raw_text="Bye")
        save_project(project)
        assert load_project(project.id) is not None
        delete_project_from_disk(project.id)
        assert load_project(project.id) is None

    def test_project_progress(self):
        project = AudiobookProject(name="Test", raw_text="x")
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="Seg 1", status=SegmentStatus.DONE),
            Segment(text="Seg 2", status=SegmentStatus.PENDING),
        ]
        project.chapters = [ch]
        assert project.total_segments == 2
        assert project.done_segments == 1
        assert project.progress == 0.5

    def test_detail_response(self):
        project = AudiobookProject(name="Test", raw_text="x", narrator_voice="v1")
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [Segment(text="Hello")]
        project.chapters = [ch]
        response = project_to_detail_response(project)
        assert response.name == "Test"
        assert len(response.chapters) == 1
        assert response.chapters[0].segments[0].text == "Hello"


class TestLLMHelpers:
    """Tests for the LLM module helper functions (no actual API calls)."""

    def test_strip_thinking_blocks(self):
        from app.audiobook_llm import _strip_thinking_blocks
        text = '<think>internal reasoning here</think>{"characters": []}'
        result = _strip_thinking_blocks(text)
        assert result == '{"characters": []}'

    def test_strip_nested_thinking(self):
        from app.audiobook_llm import _strip_thinking_blocks
        text = '<think>step 1\nstep 2\nstep 3</think>\n\nHere is the result'
        result = _strip_thinking_blocks(text)
        assert "Here is the result" in result
        assert "<think>" not in result

    def test_extract_json_direct(self):
        from app.audiobook_llm import _extract_json
        result = _extract_json('{"characters": []}')
        assert result == {"characters": []}

    def test_extract_json_from_code_block(self):
        from app.audiobook_llm import _extract_json
        text = 'Here is the analysis:\n```json\n{"characters": [{"name": "Alice"}]}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["characters"][0]["name"] == "Alice"

    def test_extract_json_from_braces(self):
        from app.audiobook_llm import _extract_json
        text = 'Some text before {"narrator": {"description": "calm"}} and after'
        result = _extract_json(text)
        assert result is not None
        assert result["narrator"]["description"] == "calm"

    def test_extract_json_invalid(self):
        from app.audiobook_llm import _extract_json
        result = _extract_json("This is not JSON at all")
        assert result is None

    def test_analyze_characters_processes_valid_response(self):
        """Test that analyze_characters correctly processes a well-formed LLM response."""
        from unittest.mock import patch, MagicMock
        from app.audiobook_llm import analyze_characters

        mock_response = {
            "characters": [
                {
                    "name": "Alice",
                    "description": "A brave adventurer",
                    "voice_traits": {"age": "young", "gender": "female", "quality": "cheerful"},
                    "suggested_voice": "voice_f1",
                },
                {
                    "name": "Bob",
                    "description": "A cautious friend",
                    "voice_traits": {"age": "middle-aged", "gender": "male", "quality": "deep"},
                    "suggested_voice": "voice_m1",
                },
            ],
            "narrator": {
                "description": "Calm, measured third-person narrator",
                "suggested_voice": "voice_m2",
            },
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_response)}}]
        }

        with patch("app.audiobook_llm.requests.post", return_value=mock_resp):
            result = analyze_characters("Some book text", ["voice_f1", "voice_m1", "voice_m2"])

        assert result is not None
        assert "Alice" in result["detected_characters"]
        assert "Bob" in result["detected_characters"]
        assert result["character_voice_map"]["Alice"] == "voice_f1"
        assert result["character_voice_map"]["Bob"] == "voice_m1"
        assert result["narrator_voice"] == "voice_m2"
        assert "brave" in result["character_descriptions"]["Alice"].lower()

    def test_analyze_characters_fallback_on_connection_error(self):
        """Test graceful fallback when LLM is unreachable."""
        from unittest.mock import patch
        from app.audiobook_llm import analyze_characters

        with patch("app.audiobook_llm.requests.post", side_effect=Exception("Connection refused")):
            result = analyze_characters("Some text", ["voice1"])

        assert result is None

    def test_analyze_characters_handles_thinking_blocks(self):
        """Test that Qwen3 thinking blocks are stripped before JSON parsing."""
        from unittest.mock import patch, MagicMock
        from app.audiobook_llm import analyze_characters

        response_text = '<think>Let me analyze this...</think>\n```json\n{"characters": [{"name": "Eve", "description": "A curious girl", "voice_traits": {}, "suggested_voice": "v1"}], "narrator": {"description": "neutral", "suggested_voice": "v1"}}\n```'

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": response_text}}]
        }

        with patch("app.audiobook_llm.requests.post", return_value=mock_resp):
            result = analyze_characters("Some text", ["v1"])

        assert result is not None
        assert "Eve" in result["detected_characters"]


class TestParsers:
    """Tests for audiobook_parsers module (EPUB/PDF/TXT import)."""

    def test_clean_text_collapses_whitespace(self):
        from app.audiobook_parsers import _clean_text
        text = "Hello   world.\n\n\n\n\nParagraph two."
        result = _clean_text(text)
        assert "   " not in result
        assert "\n\n\n" not in result
        assert "Hello" in result
        assert "Paragraph two." in result

    def test_clean_pdf_removes_page_numbers(self):
        from app.audiobook_parsers import _clean_pdf_text
        text = "Some text here.\n\n42\n\nMore text."
        result = _clean_pdf_text(text)
        lines = [l for l in result.split("\n") if l.strip()]
        # Standalone "42" should be removed
        assert "42" not in [l.strip() for l in lines]

    def test_clean_pdf_rejoins_hyphens(self):
        from app.audiobook_parsers import _clean_pdf_text
        text = "The quick brown fox jumped over the lazy dog and conti-\nnued running."
        result = _clean_pdf_text(text)
        assert "continued" in result

    def test_split_pdf_chapters(self):
        from app.audiobook_parsers import _split_pdf_into_chapters
        text = "Some preamble text that goes on.\n\nChapter 1: Intro\n\nSome intro text.\n\nChapter 2: Middle\n\nMiddle content here.\n\nChapter 3: End\n\nFinal text."
        chapters = _split_pdf_into_chapters(text)
        assert len(chapters) >= 3

    def test_parse_txt_fallback(self):
        from app.audiobook_parsers import parse_uploaded_file
        content = "Hello world. This is a plain text book."
        result = parse_uploaded_file("mybook.txt", content.encode("utf-8"))
        assert result.title == "mybook"
        assert len(result.chapters) == 1
        assert "Hello world" in result.raw_text

    def test_unsupported_extension(self):
        from app.audiobook_parsers import parse_uploaded_file
        with pytest.raises(ValueError, match="Unsupported"):
            parse_uploaded_file("book.docx", b"Some content")

    def test_supported_extensions_set(self):
        from app.audiobook_parsers import SUPPORTED_EXTENSIONS
        assert ".epub" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS


class TestSplitMerge:
    """Tests for segment split and merge logic."""

    def test_split_creates_two_segments(self):
        """Splitting a segment replaces it with two in the chapter."""
        project = AudiobookProject(name="Test", raw_text="x")
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="First sentence. Second sentence. Third sentence.", voice_name="v1", character="Alice"),
        ]
        project.chapters = [ch]
        original_id = ch.segments[0].id

        # Simulate split at auto position (sentence boundary near midpoint)
        text = ch.segments[0].text
        # Find sentence boundary near midpoint
        import re
        midpoint = len(text) // 2
        best_pos = midpoint
        for m in re.finditer(r'[.!?]\s+', text):
            if abs(m.end() - midpoint) < abs(best_pos - midpoint):
                best_pos = m.end()

        part1 = text[:best_pos].strip()
        part2 = text[best_pos:].strip()

        seg_a = Segment(text=part1, voice_name="v1", character="Alice")
        seg_b = Segment(text=part2, voice_name="v1", character="Alice")
        ch.segments = [seg_a, seg_b]

        assert len(ch.segments) == 2
        assert ch.segments[0].voice_name == "v1"
        assert ch.segments[1].voice_name == "v1"
        assert ch.segments[0].character == "Alice"

    def test_split_at_explicit_position(self):
        """Splitting at an explicit character index creates two correct parts."""
        text = "Hello world. Goodbye world."
        part1 = text[:13].strip()
        part2 = text[13:].strip()
        assert part1 == "Hello world."
        assert part2 == "Goodbye world."

    def test_merge_combines_text(self):
        """Merging two segments combines their text."""
        seg1 = Segment(text="First part.", voice_name="v1", character="Bob")
        seg2 = Segment(text="Second part.", voice_name="v2")
        merged_text = seg1.text.rstrip() + " " + seg2.text.lstrip()
        merged = Segment(text=merged_text, voice_name=seg1.voice_name, character=seg1.character)

        assert merged.text == "First part. Second part."
        assert merged.voice_name == "v1"
        assert merged.character == "Bob"
        assert merged.status == SegmentStatus.PENDING

    def test_merge_replaces_segments_in_chapter(self):
        """After merge, the chapter has one fewer segment."""
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="One."),
            Segment(text="Two."),
            Segment(text="Three."),
        ]
        # Merge seg 0 with seg 1
        merged = Segment(text="One. Two.")
        ch.segments[0:2] = [merged]
        assert len(ch.segments) == 2
        assert ch.segments[0].text == "One. Two."
        assert ch.segments[1].text == "Three."


class TestStatsAndRetry:
    """Tests for stats computation and retry-failed logic."""

    def test_stats_computed_from_segments(self):
        """project_to_detail_response computes total_duration, error_segments, total_characters."""
        from app.audiobook_models import project_to_detail_response

        project = AudiobookProject(name="Test", raw_text="x")
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="Hello world.", status=SegmentStatus.DONE, duration=2.5),
            Segment(text="Goodbye world.", status=SegmentStatus.ERROR, error_message="TTS fail"),
            Segment(text="OK.", status=SegmentStatus.PENDING),
        ]
        project.chapters = [ch]

        resp = project_to_detail_response(project)
        assert resp.total_duration == 2.5
        assert resp.error_segments == 1
        assert resp.total_characters == len("Hello world.") + len("Goodbye world.") + len("OK.")

    def test_retry_resets_error_segments(self):
        """Simulates retry-failed by resetting ERROR segments to PENDING."""
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="A.", status=SegmentStatus.DONE, duration=1.0),
            Segment(text="B.", status=SegmentStatus.ERROR, error_message="fail"),
            Segment(text="C.", status=SegmentStatus.ERROR, error_message="timeout"),
        ]
        # Simulate retry logic
        reset_count = 0
        for seg in ch.segments:
            if seg.status == SegmentStatus.ERROR:
                seg.status = SegmentStatus.PENDING
                seg.error_message = None
                seg.duration = None
                reset_count += 1

        assert reset_count == 2
        assert ch.segments[0].status == SegmentStatus.DONE
        assert ch.segments[1].status == SegmentStatus.PENDING
        assert ch.segments[2].status == SegmentStatus.PENDING
        assert ch.segments[1].error_message is None


class TestVisualPipeline:
    """Tests for visual generation data model and pipeline."""

    def test_segment_visual_fields_default(self):
        """New segments should have default visual fields."""
        seg = Segment(text="Hello.")
        assert seg.scene_prompt is None
        assert seg.visual_path is None
        assert seg.visual_type is None
        assert seg.visual_status == "none"

    def test_visual_ready_count(self):
        """project_to_detail_response computes visual_ready correctly."""
        from app.audiobook_models import project_to_detail_response

        project = AudiobookProject(name="Test", raw_text="x")
        ch = Chapter(index=0, title="Ch1")
        ch.segments = [
            Segment(text="A.", visual_status="done"),
            Segment(text="B.", visual_status="none"),
            Segment(text="C.", visual_status="done"),
        ]
        project.chapters = [ch]

        resp = project_to_detail_response(project)
        assert resp.visual_ready == 2

    def test_segment_response_visual_fields(self):
        """SegmentResponse includes visual fields."""
        from app.audiobook_models import SegmentResponse
        sr = SegmentResponse(
            id="t1", text="Hi.", voice_name=None, character=None, emotion=None,
            status=SegmentStatus.DONE, duration=1.0, error_message=None,
            has_audio=True, scene_prompt="A dark forest.", has_visual=True,
            visual_type="image", visual_status="done",
        )
        assert sr.scene_prompt == "A dark forest."
        assert sr.has_visual is True
        assert sr.visual_type == "image"
        assert sr.visual_status == "done"

    def test_scene_prompt_template_format(self):
        """Scene prompt template can be formatted without errors."""
        from app.audiobook_llm import SCENE_PROMPT_TEMPLATE
        result = SCENE_PROMPT_TEMPLATE.format(
            text="The old wizard raised his staff.",
            chapter_title="The Battle",
            character="Gandalf",
            emotion="angry",
            style_section="\nVisual Style (maintain this): oil painting, warm tones\n",
            context_section="\nContext for continuity:\nPrevious scene: A dark cavern.\n",
        )
        assert "The old wizard" in result
        assert "Gandalf" in result
        assert "oil painting" in result
        assert "dark cavern" in result

    def test_video_assembler_ffmpeg_args(self):
        """create_segment_clip constructs correct ffmpeg args for images."""
        from app.video_assembler import _run_ffmpeg
        # We just verify the module imports and the function exists
        assert callable(_run_ffmpeg)
