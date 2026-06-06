#!/usr/bin/env python3
"""Listen page from raw/pruned emotion WAVs (no QC manifest required)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=ROOT / "training/loli_15s_batch3/corpus/texts.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max", type=int, default=50)
    args = p.parse_args()

    texts: dict[str, str] = {}
    if args.corpus.is_file():
        for line in args.corpus.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                texts[row["id"]] = row["text"]

    args.out.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out / "clips"
    clips_dir.mkdir(exist_ok=True)

    wavs = sorted(args.wav_dir.glob("*__emotion__*.wav"))[: args.max]
    rows_html = []
    for w in wavs:
        cid = w.stem.split("__", 1)[0]
        parts = w.stem.split("__")
        gap = parts[1] if len(parts) > 1 else ""
        style = parts[2] if len(parts) > 2 else ""
        length = parts[3] if len(parts) > 3 else ""
        dest = clips_dir / w.name
        if not dest.is_file():
            shutil.copy2(w, dest)
        text = texts.get(cid, "")
        rel = f"clips/{w.name}"
        rows_html.append(
            f"<tr><td><audio controls preload='none' src='{rel}'></audio></td>"
            f"<td><code>{style}</code></td><td>{length}</td>"
            f"<td><small>{text[:140]}</small></td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Loli emotion teachers</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#0f1419;color:#e6edf3}}
h1{{color:#7ee787}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #30363d;padding:.5rem}} th{{background:#21262d}}
audio{{width:300px}}
</style></head><body>
<h1>Emotion teacher clips ({len(wavs)} shown)</h1>
<p>Source: <code>{args.wav_dir}</code></p>
<table><tr><th>Audio</th><th>Style</th><th>Length</th><th>Text</th></tr>
{''.join(rows_html)}
</table></body></html>"""
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {args.out / 'index.html'} ({len(wavs)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
