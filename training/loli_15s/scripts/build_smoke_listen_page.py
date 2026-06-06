#!/usr/bin/env python3
"""Build HTML listen page from smoke QC manifest (pass + quarantine samples)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-each", type=int, default=15, help="Max clips per bucket to copy")
    p.add_argument("--emotion-only", action="store_true", help="Only pass/trim clips with __emotion__ in name")
    args = p.parse_args()

    smoke = args.smoke_dir
    out = args.out
    manifest = smoke / "qc/prune_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}")

    by_wav: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            by_wav[json.loads(line)["wav"]] = json.loads(line)

    pruned = smoke / "wavs/v15_pruned"
    raw = smoke / "wavs/v15"
    quar = smoke / "wavs/v15_quarantine"

    out.mkdir(parents=True, exist_ok=True)
    (out / "pass").mkdir(exist_ok=True)
    (out / "quarantine").mkdir(exist_ok=True)

    rows: list[dict] = []
    for wav_name, rec in sorted(by_wav.items()):
        action = rec.get("action", "?")
        src = None
        bucket = "other"
        if action in ("pass", "trim"):
            src = pruned / wav_name if (pruned / wav_name).is_file() else raw / wav_name
            bucket = "pass"
        elif action == "quarantine":
            src = quar / wav_name if (quar / wav_name).is_file() else raw / wav_name
            bucket = "quarantine"
        if src is None or not src.is_file():
            continue
        if args.emotion_only and "__emotion__" not in wav_name:
            continue
        rows.append({**rec, "bucket": bucket, "src": src})

    pass_n = quar_n = 0
    html_rows = []
    for rec in rows:
        b = rec["bucket"]
        if b == "pass" and pass_n >= args.max_each:
            continue
        if b == "quarantine" and quar_n >= args.max_each:
            continue
        dest_name = rec["wav"]
        dest = out / b / dest_name
        shutil.copy2(rec["src"], dest)
        if b == "pass":
            pass_n += 1
        else:
            quar_n += 1
        rel = f"{b}/{dest_name}"
        cref = rec.get("cos_ref")
        ctchr = rec.get("cos_teacher")
        html_rows.append(
            f'<tr><td><audio controls preload="none" src="{rel}"></audio></td>'
            f'<td>{b}</td><td>{rec.get("action")}</td>'
            f'<td>{cref if cref is not None else "—"}</td>'
            f'<td>{ctchr if ctchr is not None else "—"}</td>'
            f'<td>{rec.get("wer", "—")}</td>'
            f'<td>{"yes" if rec.get("likely_cutoff") else "—"}</td>'
            f'<td>{rec.get("missing_tail_words", "—")}</td>'
            f'<td>{", ".join(rec.get("reasons") or [])}</td>'
            f'<td><small>{rec.get("ref_text", "")[:120]}</small></td></tr>'
        )

    summary_path = smoke / "qc/prune_report.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text()).get("summary", {})

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Loli smoke 100 — QC listen</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1419; color: #e6edf3; }}
h1,h2 {{ color: #7ee787; }}
table {{ border-collapse: collapse; width: 100%; }}
th,td {{ border: 1px solid #30363d; padding: 0.5rem; vertical-align: top; }}
th {{ background: #21262d; }}
audio {{ width: 280px; }}
.muted {{ color: #8b949e; }}
</style></head><body>
<h1>Loli smoke 100 — teacher QC</h1>
<p class="muted">Floor: cos(ref) and cos(teacher) ≥ 0.5. Pass bucket: trimmed OK. Quarantine: failed QC.</p>
<pre>{json.dumps(summary.get("by_action", summary), indent=2)}</pre>
<h2>Samples (up to {args.max_each} per bucket)</h2>
<table>
<tr><th>Audio</th><th>Bucket</th><th>Action</th><th>cos(ref)</th><th>cos(tchr)</th><th>WER</th><th>Cutoff?</th><th>Missing words</th><th>Reasons</th><th>Text</th></tr>
{"".join(html_rows)}
</table>
</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"Copied pass={pass_n} quarantine={quar_n} → {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
