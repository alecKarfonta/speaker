# Audiobook Generator — Feature Requests

## 1. 🎭 AI-Powered Character Detection & Voice Matching
**Priority:** High | **Status:** In Progress

Use an LLM to intelligently parse book text and:
- Identify all speaking characters, including indirect attribution ("The old man chuckled")
- Auto-suggest voice characteristics per character (age, gender, tone)
- Match characters to the best available voice from the library
- Detect narrator tone shifts (action vs. quiet) and apply emotion tags

**Goal:** Turn manual character→voice mapping into a one-click smart assignment.

---

## 2. ⚡ Real-Time Generation Pipeline with WebSocket Progress
**Priority:** Medium | **Status:** Planned

Replace synchronous generation with a live streaming pipeline:
- WebSocket endpoint streaming per-segment progress updates
- Live animated progress dashboard
- Parallel segment generation (2-3 concurrent, GPU permitting)
- Pause/resume/cancel controls
- Auto-play segments as they complete

**Goal:** Make long generation runs interactive instead of blocking.

---

## 3. 📖 EPUB/PDF Import with Structural Awareness
**Priority:** Medium | **Status:** Planned

Support proper book file formats beyond raw text:
- EPUB parsing with native chapter/metadata extraction
- PDF extraction with layout-aware paragraph detection
- Smart cleanup (strip page numbers, headers/footers, footnotes)
- Preserve italics/bold as SSML emphasis markup
- Auto-import table of contents as chapter structure

**Goal:** Drag-and-drop a book file and have it ready to generate immediately.


2. 👁️ Live Audio-Visual Preview in UI
Let the user preview each segment — click a segment row and see its video clip playing with the narration audio overlaid. Right now you generate everything blindly and only see results after a full export. With per-segment preview you get instant feedback: bad scene prompt? Regenerate just that one. Audio pacing off? Adjust. This makes the creative workflow interactive instead of batch-and-pray.

3. ✏️ Scene Prompt Editor
The LLM generates scene prompts automatically, but the user can't see or edit them before visual generation. A bad scene prompt = wasted 2 minutes of GPU time. Adding a prompt editor panel — show the generated prompt, let users tweak it ("make the castle darker", "add rain"), then regenerate — gives creative control. It should use the llm to create a prompt for each section based on the book text, but I should also be able to edit and rerun the gneration
