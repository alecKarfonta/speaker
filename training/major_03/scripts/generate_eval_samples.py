#!/usr/bin/env python3
"""Native-voice (no ref WAV) eval samples for major_03 MOSS-RT LoRA."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

import requests

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
TRAIN_DIR = Path(os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training" / "major_03"))
DEFAULT_OUT = TRAIN_DIR / "eval" / "listen" / "epoch11_native"
DEFAULT_API = os.environ.get("MOSS_RT_API", "http://127.0.0.1:8016")

# No reference audio — text only (native-voice merged model).
SAMPLES: list[tuple[str, str, str]] = [
    ("01_greeting_short.wav", "greeting", "Good morning. Systems are online and we are ready to proceed."),
    (
        "02_question.wav",
        "question",
        "Consider this: what happens to the timeline if we delay the handoff by even one cycle?",
    ),
    (
        "03_directive.wav",
        "directive",
        "Listen. The short version is we move at dawn, before the network traffic spikes again.",
    ),
    (
        "04_calm_medium.wav",
        "calm",
        "The corridor lights steadied. She studied the readout without blinking, counting heartbeats "
        "the way others count coins. Somewhere above them, turbines sang a low hymn.",
    ),
    (
        "05_story_long.wav",
        "story",
        "Rain had scrubbed the streets clean, leaving reflections that looked like second skylines. "
        "He walked without hurry, hands in pockets, mind already three meetings ahead. A drone passed "
        "overhead, whispering regulations into the wind. At the corner cafe he ordered black coffee "
        "and wrote one sentence in a notebook: adapt, then commit.",
    ),
    (
        "06_signoff.wav",
        "sign-off",
        "That is the part people forget. We proceed from there. I will leave the conclusion to you.",
    ),
    (
        "07_quiet.wav",
        "quiet",
        "Silence can be a weapon, or a shield, depending on who breaks it first.",
    ),
    (
        "08_punctuation.wav",
        "punctuation",
        "To be clear — the data suggests failure; the team chose hope anyway. Nothing more complicated than that.",
    ),
    (
        "09_list.wav",
        "list",
        "Three priorities: verify the uplink, isolate the faulty node, and document every decision.",
    ),
    (
        "10_reflective.wav",
        "reflective",
        "Trust is a resource you spend carefully, not a currency you print on demand.",
    ),
    (
        "11_technical.wav",
        "technical",
        "The interface between hardware and instinct is thinner than most people admit. The rest is execution.",
    ),
    (
        "12_closer.wav",
        "closer",
        "You already know what comes next. Make of that what you will.",
    ),
]

# Subset for sampling sweep (same text, different decode params).
SWEEP_ANCHORS: list[tuple[str, str, str]] = [
    ("02_question.wav", "question", SAMPLES[1][2]),
    ("03_directive.wav", "directive", SAMPLES[2][2]),
    ("05_story_long.wav", "story", SAMPLES[4][2]),
    ("11_technical.wav", "technical", SAMPLES[10][2]),
]

# Locked decode for production / A/B (default = warm_092_072 sweep winner).
LOCKED_PRESETS: dict[str, tuple[float, float, int, float]] = {
    "default": (0.92, 0.72, 40, 1.05),  # warm_092_072
    "legacy_08_06": (0.80, 0.60, 30, 1.10),
    "light": (0.82, 0.62, 35, 1.08),
}

# Named presets: (label, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty)
SWEEP_PRESETS: list[tuple[str, float, float, int, float]] = [
    ("default_08_06", 0.80, 0.60, 30, 1.10),
    ("warm_092_072", 0.92, 0.72, 40, 1.05),
    ("warm_095_072", 0.95, 0.72, 40, 1.05),
    ("hot_100_075", 1.00, 0.75, 50, 1.05),
    ("hot_105_080", 1.05, 0.80, 50, 1.03),
    ("v15_teacher_base", 0.58, 0.58, 28, 1.06),
]

# Fine grid around hot_100_075 (T=1.0 p=0.75 k=50) — story-focused eval set.
HOT_ZONE_PRESETS: list[tuple[str, float, float, int, float]] = [
    ("T095_p075_k50_r105", 0.95, 0.75, 50, 1.05),
    ("T098_p075_k50_r105", 0.98, 0.75, 50, 1.05),
    ("T100_p070_k50_r105", 1.00, 0.70, 50, 1.05),
    ("T100_p072_k50_r105", 1.00, 0.72, 50, 1.05),
    ("T100_p075_k40_r105", 1.00, 0.75, 40, 1.05),
    ("T100_p075_k50_r103", 1.00, 0.75, 50, 1.03),
    ("T100_p075_k50_r105", 1.00, 0.75, 50, 1.05),  # sweep winner (hot_100_075)
    ("T100_p075_k50_r108", 1.00, 0.75, 50, 1.08),
    ("T100_p075_k60_r105", 1.00, 0.75, 60, 1.05),
    ("T100_p078_k50_r105", 1.00, 0.78, 50, 1.05),
    ("T100_p080_k50_r105", 1.00, 0.80, 50, 1.05),
    ("T102_p075_k50_r105", 1.02, 0.75, 50, 1.05),
    ("T105_p075_k50_r105", 1.05, 0.75, 50, 1.05),
]

HOT_ZONE_CLIPS: list[tuple[str, str, str]] = [
    ("05_story_long.wav", "story", SAMPLES[4][2]),
    (
        "12_story_city.wav",
        "story-city",
        "The city breathed differently after midnight, as if it were waiting for instructions. "
        "She crossed the bridge without looking down at the water, trusting the rail more than the map. "
        "Every light she passed seemed to nod in agreement.",
    ),
    (
        "13_story_archive.wav",
        "story-archive",
        "He opened the archive box and found letters no one had read in decades. "
        "The ink had faded but the intent had not. One paragraph changed how he understood "
        "the project, and himself.",
    ),
    (
        "14_story_dawn.wav",
        "story-dawn",
        "At dawn the harbor woke in layers: gulls, engines, then voices. "
        "They timed the launch to the slack tide and a window of quiet spectrum. "
        "Failure was possible; hesitation was not.",
    ),
    ("04_calm_medium.wav", "calm", SAMPLES[3][2]),
    ("02_question.wav", "question", SAMPLES[1][2]),
]

# Fine grid around T=0.92 p=0.72 k=40 (listening winner); story-only + consistency reps.
TARGET_SWEEP_PRESETS: list[tuple[str, float, float, int, float]] = [
    ("T088_p072_k40_r105", 0.88, 0.72, 40, 1.05),
    ("T090_p072_k40_r105", 0.90, 0.72, 40, 1.05),
    ("T092_p070_k40_r105", 0.92, 0.70, 40, 1.05),
    ("T092_p072_k35_r105", 0.92, 0.72, 35, 1.05),
    ("T092_p072_k40_r103", 0.92, 0.72, 40, 1.03),
    ("T092_p072_k40_r105", 0.92, 0.72, 40, 1.05),  # center (≈ warm_092 + k40)
    ("T092_p072_k40_r108", 0.92, 0.72, 40, 1.08),
    ("T092_p072_k45_r105", 0.92, 0.72, 45, 1.05),
    ("T092_p074_k40_r105", 0.92, 0.74, 40, 1.05),
    ("T094_p072_k40_r105", 0.94, 0.72, 40, 1.05),
    ("T096_p072_k40_r105", 0.96, 0.72, 40, 1.05),
]

TARGET_SWEEP_CLIPS: list[tuple[str, str, str]] = [
    c for c in HOT_ZONE_CLIPS if c[1].startswith("story") or c[0].startswith("05_story")
]

TARGET_CONSISTENCY_REPS = 2


def sweep_html_title(dir_name: str) -> str:
    if "target" in dir_name:
        return "major_03 epoch-11 — target sweep (T≈0.92 p≈0.72 k=40, consistency)"
    if "hot_zone" in dir_name:
        return "major_03 epoch-11 — hot zone sweep (T≈1.0 p≈0.75, story focus)"
    return "major_03 epoch-11 — sampling sweep (native, no ref)"


def stream_to_wav(
    api: str,
    text: str,
    *,
    audio_temperature: float | None = None,
    audio_top_p: float | None = None,
    audio_top_k: int | None = None,
    audio_repetition_penalty: float | None = None,
) -> tuple[bytes, dict]:
    payload: dict = {"text": text, "language": "en"}
    if audio_temperature is not None:
        payload["audio_temperature"] = audio_temperature
    if audio_top_p is not None:
        payload["audio_top_p"] = audio_top_p
    if audio_top_k is not None:
        payload["audio_top_k"] = audio_top_k
    if audio_repetition_penalty is not None:
        payload["audio_repetition_penalty"] = audio_repetition_penalty

    t0 = time.perf_counter()
    resp = requests.post(
        f"{api.rstrip('/')}/tts/stream",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    raw = resp.content
    wall_s = time.perf_counter() - t0

    pcm = bytearray()
    sr = 24000
    offset = 0
    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + audio_len + meta_len > len(raw):
            break
        if audio_len > 0:
            chunk = raw[offset : offset + audio_len]
            try:
                with wave.open(io.BytesIO(chunk), "rb") as wf:
                    sr = wf.getframerate()
                    pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                pcm.extend(chunk)
        offset += audio_len + meta_len

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(pcm))

    audio_s = len(pcm) / (sr * 2) if pcm else 0.0
    return buf.getvalue(), {"wall_s": round(wall_s, 2), "audio_s": round(audio_s, 2), "sample_rate": sr}


def write_index(out: Path, rows: list[dict], *, html_title: str) -> None:
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        f"<title>{html_title}</title>",
        "<style>body{font-family:system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;vertical-align:top}",
        "audio{width:100%}</style></head><body>",
        f"<h1>{html_title}</h1>",
        "<p>Native voice (no reference WAV). MOSS_RT_NATIVE_VOICE=true.</p>"
        + (
            f"<p>Sampling: T={rows[0].get('audio_temperature')} top_p={rows[0].get('audio_top_p')} "
            f"rep={rows[0].get('audio_repetition_penalty')}</p>"
            if rows and rows[0].get("audio_temperature") is not None
            else ""
        ),
        "<table><tr><th>#</th><th>Tag</th><th>Text</th><th>Audio</th><th>dur</th></tr>",
    ]
    for i, row in enumerate(rows, 1):
        html.append(
            f"<tr><td>{i}</td><td>{row['tag']}</td><td>{row['text']}</td>"
            f"<td><audio controls src='{row['file']}'></audio></td>"
            f"<td>{row.get('audio_s', '?')}s</td></tr>"
        )
    html.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _score_cell(scores: dict | None) -> str:
    if not scores:
        return ""
    parts = []
    if scores.get("cos_ref") is not None:
        parts.append(f"ref <b>{scores['cos_ref']:.3f}</b>")
    if scores.get("cos_teacher") is not None:
        parts.append(f"tchr {scores['cos_teacher']:.3f}")
    if scores.get("wer") is not None:
        parts.append(f"WER {scores['wer']:.2f}")
    if not parts:
        return ""
    return "<br><span class=\"scores\">" + " · ".join(parts) + "</span>"


def write_sweep_index(
    out: Path,
    rows: list[dict],
    *,
    html_title: str,
    scores_by_file: dict[str, dict] | None = None,
    preset_ranking: list[dict] | None = None,
) -> None:
    all_presets = sorted({r["preset"] for r in rows})
    if preset_ranking:
        presets = [r["preset"] for r in preset_ranking if r["preset"] in all_presets]
        for p in all_presets:
            if p not in presets:
                presets.append(p)
    else:
        presets = all_presets

    rank_by_preset = {r["preset"]: r for r in (preset_ranking or [])}
    clips: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        cid = r["clip_id"]
        if cid in seen:
            continue
        seen.add(cid)
        clips.append({"clip_id": cid, "tag": r["tag"], "text": r["text"]})

    ranking_block = ""
    if preset_ranking:
        rrows = []
        for r in preset_ranking:
            ct = (
                f"{r['cos_teacher_median']:.3f}"
                if r.get("cos_teacher_median") is not None
                else "—"
            )
            std = r.get("cos_ref_std")
            std_s = f"{std:.3f}" if std is not None else "—"
            rrows.append(
                f"<tr><td>{r['rank']}</td><td><b>{r['preset']}</b></td>"
                f"<td>{r['cos_ref_median']:.3f}</td><td>{std_s}</td>"
                f"<td>{r['cos_ref_mean']:.3f}</td><td>{ct}</td></tr>"
            )
        ranking_block = (
            "<h2>Preset ranking</h2>"
            "<p class=\"muted\">Sorted by median <b>cos(ref)</b>, then lowest <b>σ</b> (consistency). "
            "Listen below — ref/tchr scores match human judgment well on this model.</p>"
            "<table class=\"rank\"><tr><th>#</th><th>Preset</th><th>cos(ref) med</th>"
            "<th>cos(ref) σ</th><th>mean</th><th>cos(tchr) med</th></tr>"
            + "".join(rrows)
            + "</table>"
        )

    nav_links = "".join(
        f'<a href="#{p}">#{rank_by_preset[p]["rank"]} {p}</a> '
        for p in presets
        if p in rank_by_preset
    )

    sections: list[str] = []
    for p in presets:
        rk = rank_by_preset.get(p)
        sample = next((r for r in rows if r["preset"] == p), None)
        decode = ""
        if sample:
            decode = (
                f"T={sample.get('audio_temperature')} "
                f"p={sample.get('audio_top_p')} "
                f"k={sample.get('audio_top_k')} "
                f"rep={sample.get('audio_repetition_penalty')}"
            )
        hdr = f"<h2 id=\"{p}\">"
        if rk:
            std = rk.get("cos_ref_std")
            std_s = f" · σ={std:.3f}" if std is not None else ""
            hdr += (
                f"#{rk['rank']} <b>{p}</b> — ref med {rk['cos_ref_median']:.3f}{std_s}"
                f" · tchr med {rk.get('cos_teacher_median', 0):.3f}"
            )
        else:
            hdr += f"<b>{p}</b>"
        hdr += f"<br><small class=\"muted\">{decode}</small></h2>"
        section_cls = "preset-section best" if rk and rk.get("rank") == 1 else "preset-section"
        clip_blocks: list[str] = []
        for clip in clips:
            clip_rows = sorted(
                [r for r in rows if r["preset"] == p and r["clip_id"] == clip["clip_id"]],
                key=lambda x: x.get("rep", 1),
            )
            if not clip_rows:
                continue
            reps_html: list[str] = []
            for r in clip_rows:
                sc = (scores_by_file or {}).get(r["file"])
                rep_lbl = f"run {r.get('rep', 1)}" if len(clip_rows) > 1 else "output"
                score_html = _score_cell(sc) if sc else (
                    '<br><span class="scores muted">scores pending — run bench</span>'
                )
                reps_html.append(
                    f'<div class="rep"><div class="rep-label">{rep_lbl}</div>'
                    f'<audio controls preload="none" src="{r["file"]}"></audio>'
                    f"<small>{r.get('audio_s', '?')}s</small>{score_html}</div>"
                )
            snippet = clip["text"][:160] + ("…" if len(clip["text"]) > 160 else "")
            clip_blocks.append(
                f'<div class="clip"><h3>{clip["tag"]}</h3>'
                f'<p class="clip-text">{snippet}</p>'
                f'<div class="reps">{"".join(reps_html)}</div></div>'
            )
        sections.append(f'<section class="{section_cls}">{hdr}{"".join(clip_blocks)}</section>')

    bench_note = ""
    if scores_by_file:
        bench_note = (
            f'<p class="muted">Scores: <code>eval/bench/{out.name}/scores.json</code> '
            "(SpeechBrain ECAPA).</p>"
        )
    elif rows:
        bench_note = '<p class="muted">Run bench to attach cos(ref) / cos(tchr) scores.</p>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{html_title}</title>
<style>
body{{font-family:system-ui;max-width:52rem;margin:2rem auto;padding:0 1.5rem;line-height:1.4}}
table.rank{{border-collapse:collapse;width:100%;font-size:0.9rem;margin-bottom:1.5rem}}
table.rank td,table.rank th{{border:1px solid #ccc;padding:0.45rem 0.6rem;text-align:left}}
nav{{margin:1rem 0;font-size:0.85rem;line-height:1.8}}
nav a{{margin-right:0.5rem}}
.preset-section{{border:1px solid #ddd;border-radius:8px;padding:1rem 1.2rem;margin:2rem 0}}
.preset-section.best{{border-color:#7cb342;background:#f9fbe7}}
.clip{{margin:1.25rem 0 0;padding-top:1rem;border-top:1px solid #eee}}
.clip:first-of-type{{border-top:none;padding-top:0}}
.clip h3{{margin:0 0 0.35rem;font-size:1rem}}
.clip-text{{margin:0 0 0.75rem;color:#444;font-size:0.85rem}}
.reps{{display:flex;flex-wrap:wrap;gap:1rem}}
.rep{{flex:1;min-width:14rem;max-width:24rem}}
.rep-label{{font-size:0.75rem;font-weight:600;color:#666;margin-bottom:0.25rem}}
audio{{width:100%;display:block;margin:0.25rem 0}}
.scores{{color:#0d47a1;font-size:0.82rem;display:block;margin-top:0.2rem}}
.muted{{color:#666}}
</style></head><body>
<h1>{html_title}</h1>
<p>Native voice, no ref. Each section = one decode preset; multiple <b>runs</b> per clip when testing consistency.</p>
{ranking_block}
<nav>{nav_links}</nav>
<h2>Listen by preset</h2>
{"".join(sections)}
{bench_note}
</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    if not scores_by_file:
        (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_sweep_grid(
    api: str,
    out: Path,
    *,
    presets: list[tuple[str, float, float, int, float]],
    clips: list[tuple[str, str, str]],
    html_title: str,
    consistency_reps: int = 1,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_reps = max(1, consistency_reps)
    print(
        f"API: {api}\nOut: {out}\n"
        f"Presets: {len(presets)} × clips: {len(clips)} × reps: {n_reps}\n"
    )
    for preset_name, temp, top_p, top_k, rep in presets:
        preset_dir = out / preset_name
        preset_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- {preset_name} (T={temp} top_p={top_p} top_k={top_k} rep={rep}) ---")
        for fname, tag, text in clips:
            stem = Path(fname).stem
            for run_i in range(1, n_reps + 1):
                suffix = f"_r{run_i}" if n_reps > 1 else ""
                out_name = f"{stem}__{preset_name}{suffix}.wav"
                wav, meta = stream_to_wav(
                    api,
                    text,
                    audio_temperature=temp,
                    audio_top_p=top_p,
                    audio_top_k=top_k,
                    audio_repetition_penalty=rep,
                )
                rel = f"{preset_name}/{out_name}"
                (out / rel).write_bytes(wav)
                row = {
                    "file": rel,
                    "preset": preset_name,
                    "clip_id": fname,
                    "tag": tag,
                    "text": text,
                    "rep": run_i,
                    "audio_temperature": temp,
                    "audio_top_p": top_p,
                    "audio_top_k": top_k,
                    "audio_repetition_penalty": rep,
                    **meta,
                }
                rows.append(row)
                print(f"  {out_name}  {meta['audio_s']}s")
    write_sweep_index(out, rows, html_title=html_title)
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} WAVs → {out}")
    return 0


def run_sweep(api: str, out: Path) -> int:
    return run_sweep_grid(
        api,
        out,
        presets=SWEEP_PRESETS,
        clips=SWEEP_ANCHORS,
        html_title="major_03 epoch-11 — sampling sweep (native, no ref)",
    )


def run_hot_zone_sweep(api: str, out: Path) -> int:
    return run_sweep_grid(
        api,
        out,
        presets=HOT_ZONE_PRESETS,
        clips=HOT_ZONE_CLIPS,
        html_title="major_03 epoch-11 — hot zone sweep (T≈1.0 p≈0.75, story focus)",
    )


def run_target_sweep(api: str, out: Path) -> int:
    return run_sweep_grid(
        api,
        out,
        presets=TARGET_SWEEP_PRESETS,
        clips=TARGET_SWEEP_CLIPS,
        html_title="major_03 epoch-11 — target sweep (T≈0.92 p≈0.72 k=40, consistency)",
        consistency_reps=TARGET_CONSISTENCY_REPS,
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Grid of audio_temperature/top_p presets on anchor clips",
    )
    p.add_argument(
        "--sweep-hot",
        action="store_true",
        help="Fine grid around T=1.0 p=0.75 k=50 on story-heavy clips",
    )
    p.add_argument(
        "--sweep-target",
        action="store_true",
        help="Fine grid around T=0.92 p=0.72 k=40; story clips × 2 consistency runs",
    )
    p.add_argument(
        "--preset",
        choices=tuple(LOCKED_PRESETS.keys()),
        default=None,
        help="Locked sampling (default=warm_092 0.92/0.72/k40; legacy_08_06; light)",
    )
    p.add_argument("--wait-health", type=int, default=600)
    p.add_argument(
        "--refresh-html-only",
        action="store_true",
        help="Rebuild sweep index.html from manifest + bench scores (no TTS)",
    )
    args = p.parse_args()
    if args.sweep and args.out == DEFAULT_OUT:
        args.out = TRAIN_DIR / "eval" / "listen" / "epoch11_sampling_sweep"
    if args.sweep_hot and args.out == DEFAULT_OUT:
        args.out = TRAIN_DIR / "eval" / "listen" / "epoch11_hot_zone_sweep"
    if args.sweep_target and args.out == DEFAULT_OUT:
        args.out = TRAIN_DIR / "eval" / "listen" / "epoch11_target_sweep"
    if args.preset and args.out == DEFAULT_OUT:
        args.out = TRAIN_DIR / "eval" / "listen" / f"epoch11_{args.preset}"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.refresh_html_only:
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            print(f"Missing {manifest_path}", file=sys.stderr)
            return 1
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        scores_path = TRAIN_DIR / "eval" / "bench" / out.name / "scores.json"
        scores_by_file: dict[str, dict] = {}
        preset_ranking: list[dict] = []
        if scores_path.is_file():
            payload = json.loads(scores_path.read_text(encoding="utf-8"))
            scores_by_file = {c["file"]: c for c in payload.get("clips", [])}
            preset_ranking = payload.get("preset_ranking") or payload.get("summary", {}).get(
                "preset_ranking", []
            )
        title = sweep_html_title(out.name)
        write_sweep_index(
            out,
            rows,
            html_title=title,
            scores_by_file=scores_by_file or None,
            preset_ranking=preset_ranking or None,
        )
        print(f"Refreshed {out / 'index.html'} ({len(scores_by_file)} scored clips)")
        return 0

    api = args.api_url.rstrip("/")

    if args.wait_health > 0:
        deadline = time.time() + args.wait_health
        while time.time() < deadline:
            try:
                r = requests.get(f"{api}/health", timeout=5)
                if r.ok:
                    h = r.json()
                    if h.get("status") == "ready" and h.get("realtime_enabled"):
                        break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            print(f"Server not ready at {api}/health (native_voice) after {args.wait_health}s")
            return 1

    if args.sweep_target:
        return run_target_sweep(api, out)
    if args.sweep_hot:
        return run_hot_zone_sweep(api, out)
    if args.sweep:
        return run_sweep(api, out)

    decode: tuple[float, float, int, float] | None = None
    if args.preset:
        decode = LOCKED_PRESETS[args.preset]
        temp, top_p, top_k, rep = decode

    rows: list[dict] = []
    if decode:
        temp, top_p, top_k, rep = decode
        title = (
            f"major_03 epoch-11 — preset={args.preset} "
            f"(T={temp} top_p={top_p} rep={rep}, native no ref)"
        )
        print(f"API: {api}\nOut: {out}\nDecode: T={temp} top_p={top_p} top_k={top_k} rep={rep}\n")
    else:
        title = "major_03 epoch-11 — native voice (no ref)"
        print(f"API: {api}\nOut: {out}\n")
    for fname, tag, text in SAMPLES:
        if decode:
            temp, top_p, top_k, rep = decode
            wav, meta = stream_to_wav(
                api,
                text,
                audio_temperature=temp,
                audio_top_p=top_p,
                audio_top_k=top_k,
                audio_repetition_penalty=rep,
            )
        else:
            wav, meta = stream_to_wav(api, text)
        path = out / fname
        path.write_bytes(wav)
        row = {"file": fname, "tag": tag, "text": text, **meta}
        if decode:
            row.update(
                {
                    "preset": args.preset,
                    "audio_temperature": temp,
                    "audio_top_p": top_p,
                    "audio_top_k": top_k,
                    "audio_repetition_penalty": rep,
                }
            )
        rows.append(row)
        print(f"  {fname}  {meta['audio_s']}s  ({tag})")
    write_index(out, rows, html_title=title)
    print(f"\nWrote {len(rows)} WAVs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
