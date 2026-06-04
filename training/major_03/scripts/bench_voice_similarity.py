#!/usr/bin/env python3
"""
Speaker-embedding benchmark: clone WAVs vs reference enrollment and teacher targets.

Uses SpeechBrain ECAPA-TDNN (cosine similarity). Optional STT WER on generated clips.

Examples:
  # Score native eval folder (default paths)
  python3 scripts/bench_voice_similarity.py

  # Another listen folder + HTML report
  python3 scripts/bench_voice_similarity.py \\
    --gen-dir eval/listen/epoch11_default \\
    --out-dir eval/bench/epoch11_default

  # Skip STT if whisper API is down
  python3 scripts/bench_voice_similarity.py --no-stt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
TRAIN_DIR = Path(os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training" / "major_03"))
LEGACY = ROOT / "training" / "moss-realtime" / "scripts" / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from build_realtime_finetune_dataset import normalize, word_error_rate  # noqa: E402

DEFAULT_REF = ROOT / "data/voices/major/major_2_03_cleaned.wav"
DEFAULT_GEN = TRAIN_DIR / "eval/listen/epoch11_native"
DEFAULT_OUT = TRAIN_DIR / "eval/bench/epoch11_native"
DEFAULT_TRAIN_RAW = TRAIN_DIR / "train_raw.jsonl"
DEFAULT_TEACHER_ROOT = TRAIN_DIR / "wavs/v15_pruned"
STT_API = os.environ.get("STT_API", "http://localhost:8603/v1/audio/transcriptions")
STT_MODEL = os.environ.get("STT_MODEL", "base")
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WER_PASS = 0.35


@dataclass
class ClipScore:
    file: str
    tag: str
    text: str
    gen_wav: str
    teacher_wav: str | None
    teacher_id: str | None
    teacher_match_score: float | None
    cos_ref: float
    cos_teacher: float | None
    duration_s: float
    wer: float | None
    wer_pass: bool | None
    hyp_snippet: str = ""
    preset: str | None = None
    rep: int | None = None


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_manifest(gen_dir: Path) -> list[dict]:
    manifest_path = gen_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for wav in sorted(gen_dir.glob("*.wav")):
        rows.append({"file": wav.name, "tag": wav.stem, "text": ""})
    return rows


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration) if info.samplerate else 0.0


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def build_teacher_index(train_raw: Path, teacher_root: Path) -> list[tuple[str, Path, str, set[str]]]:
    rows: list[tuple[str, Path, str, set[str]]] = []
    for line in train_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = row.get("id", "")
        convs = row.get("conversations") or []
        if not convs:
            continue
        text = str(convs[0].get("text", ""))
        wav_rel = str(convs[0].get("wav", ""))
        wav_path = (ROOT / wav_rel) if not Path(wav_rel).is_absolute() else Path(wav_rel)
        if not wav_path.is_file():
            alt = teacher_root / f"{tid.replace('_v15', '')}.wav"
            if alt.is_file():
                wav_path = alt
            else:
                stem = Path(wav_rel).name
                alt2 = teacher_root / stem
                if alt2.is_file():
                    wav_path = alt2
                else:
                    continue
        rows.append((tid, wav_path, text, token_set(text)))
    return rows


def find_teacher(
    eval_text: str,
    index: list[tuple[str, Path, str, set[str]]],
    *,
    min_recall: float = 0.55,
) -> tuple[Path, str, float] | None:
    if not eval_text.strip():
        return None
    eval_tokens = token_set(eval_text)
    if not eval_tokens:
        return None
    best: tuple[float, str, Path, str] | None = None
    for tid, wav_path, text, teach_tokens in index:
        if not teach_tokens:
            continue
        overlap = len(eval_tokens & teach_tokens)
        recall = overlap / len(eval_tokens)
        precision = overlap / len(teach_tokens)
        # Favor clips that contain the eval phrase (high recall); tie-break shorter teacher text.
        score = recall * 0.85 + precision * 0.15
        if recall < min_recall:
            continue
        if best is None or score > best[0] or (score == best[0] and len(text) < len(best[3])):
            best = (score, tid, wav_path, text)
    if best is None:
        # Fallback: best recall even below threshold
        for tid, wav_path, text, teach_tokens in index:
            overlap = len(eval_tokens & teach_tokens)
            recall = overlap / len(eval_tokens) if eval_tokens else 0.0
            if best is None or recall > best[0]:
                best = (recall, tid, wav_path, text)
        if best is None or best[0] < 0.25:
            return None
    return best[2], best[1], float(best[0])


class EcapaEncoder:
    def __init__(self, device: str, cache_dir: Path) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            savedir=str(cache_dir),
            run_opts={"device": device},
        )

    @torch.inference_mode()
    def embed(self, wav_path: Path) -> np.ndarray:
        signal = self.model.load_audio(str(wav_path))
        emb = self.model.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy()


def transcribe_simple(wav_path: Path, api: str, model: str) -> str:
    import requests

    with wav_path.open("rb") as f:
        r = requests.post(
            api,
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"model": model, "language": "en", "response_format": "json"},
            timeout=120,
        )
    r.raise_for_status()
    return str(r.json().get("text", "")).strip()


def summarize_presets(clips: list[ClipScore]) -> list[dict]:
    """Aggregate per decode preset; rank by median cos(ref) (voice identity)."""
    by_name: dict[str, list[ClipScore]] = {}
    for c in clips:
        if not c.preset:
            continue
        by_name.setdefault(c.preset, []).append(c)
    if len(by_name) < 2:
        return []

    rows: list[dict] = []
    for name, group in by_name.items():
        refs = [c.cos_ref for c in group]
        teachers = [c.cos_teacher for c in group if c.cos_teacher is not None]
        wers = [c.wer for c in group if c.wer is not None]
        cos_std = round(statistics.pstdev(refs), 4) if len(refs) > 1 else 0.0
        rows.append(
            {
                "preset": name,
                "n": len(group),
                "cos_ref_mean": round(statistics.mean(refs), 4),
                "cos_ref_median": round(statistics.median(refs), 4),
                "cos_ref_std": cos_std,
                "cos_ref_min": round(min(refs), 4),
                "cos_teacher_median": round(statistics.median(teachers), 4) if teachers else None,
                "wer_median": round(statistics.median(wers), 4) if wers else None,
            }
        )
    # Prefer high median identity and low variance across runs/clips.
    rows.sort(
        key=lambda r: (r["cos_ref_median"], -(r.get("cos_ref_std") or 0.0)),
        reverse=True,
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _refresh_sweep_listen_index(
    gen_dir: Path,
    preset_ranking: list[dict],
    clips: list[ClipScore],
) -> None:
    import importlib.util

    ges_path = TRAIN_DIR / "scripts" / "generate_eval_samples.py"
    spec = importlib.util.spec_from_file_location("generate_eval_samples", ges_path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = json.loads((gen_dir / "manifest.json").read_text(encoding="utf-8"))
    scores_by_file = {c.file: asdict(c) for c in clips}
    title = mod.sweep_html_title(gen_dir.name)
    mod.write_sweep_index(
        gen_dir,
        rows,
        html_title=title,
        scores_by_file=scores_by_file,
        preset_ranking=preset_ranking,
    )
    print(f"Updated listen page: {gen_dir / 'index.html'}")


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def render_html(
    out_path: Path,
    summary: dict,
    clips: list[ClipScore],
    gen_dir: Path,
    ref_path: Path,
) -> None:
    rel_gen = os.path.relpath(gen_dir.resolve(), out_path.parent.resolve())
    show_preset = any(c.preset for c in clips)
    preset_hdr = "<th>Preset</th>" if show_preset else ""
    rows_html = []
    for c in sorted(clips, key=lambda x: x.cos_ref):
        gen_rel = f"{rel_gen}/{c.file}".replace("\\", "/")
        teach_cell = "—"
        if c.teacher_wav:
            teach_rel = os.path.relpath(Path(c.teacher_wav).resolve(), out_path.parent.resolve())
            teach_cell = f'<a href="{teach_rel}">{Path(c.teacher_wav).name}</a>'
        wer_cell = f"{c.wer:.2f}" if c.wer is not None else "—"
        ct = f"{c.cos_teacher:.3f}" if c.cos_teacher is not None else "—"
        preset_col = f"<td>{c.preset or '—'}</td>" if show_preset else ""
        rows_html.append(
            f"<tr><td><a href=\"{gen_rel}\">{c.file}</a></td>"
            f"{preset_col}"
            f"<td>{c.tag}</td><td>{c.cos_ref:.3f}</td><td>{ct}</td>"
            f"<td>{wer_cell}</td><td>{teach_cell}</td>"
            f"<td class=\"muted\">{c.text[:120]}{'…' if len(c.text) > 120 else ''}</td></tr>"
        )
    preset_rows = summary.get("preset_ranking") or []
    preset_table = ""
    if preset_rows:
        pr_lines = []
        for r in preset_rows:
            ct = f"{r['cos_teacher_median']:.3f}" if r.get("cos_teacher_median") is not None else "—"
            pr_lines.append(
                f"<tr><td>{r['rank']}</td><td><b>{r['preset']}</b></td><td>{r['n']}</td>"
                f"<td>{r['cos_ref_median']:.3f}</td><td>{r['cos_ref_mean']:.3f}</td>"
                f"<td>{r['cos_ref_min']:.3f}</td><td>{ct}</td></tr>"
            )
        preset_table = f"""
<h2>Preset ranking (best params)</h2>
<p class="muted">Ranked by median cos(ref) across anchors — higher = closer to Major enrollment.</p>
<table>
<tr><th>#</th><th>Preset</th><th>n</th><th>cos(ref) median</th><th>mean</th><th>min</th><th>cos(teacher) median</th></tr>
{''.join(pr_lines)}
</table>
"""
    s = summary
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Voice similarity — {gen_dir.name}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1200px; }}
.cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
.card {{ background: #f4f4f5; padding: 0.75rem 1rem; border-radius: 8px; min-width: 140px; }}
.card b {{ display: block; font-size: 1.4rem; }}
.muted {{ color: #666; font-size: 0.9rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #eee; }}
tr:hover {{ background: #fafafa; }}
</style></head><body>
<h1>Voice similarity bench</h1>
<p class="muted">Generated {s.get('generated_at', '')} · model {s.get('embedding_model', '')}<br>
Gen: <code>{gen_dir}</code> · Ref: <code>{ref_path}</code></p>
<div class="cards">
  <div class="card"><span class="muted">Clips</span><b>{s['n']}</b></div>
  <div class="card"><span class="muted">cos(ref) median</span><b>{s.get('cos_ref_median', 0):.3f}</b></div>
  <div class="card"><span class="muted">cos(ref) mean</span><b>{s.get('cos_ref_mean', 0):.3f}</b></div>
  <div class="card"><span class="muted">cos(teacher) median</span><b>{s.get('cos_teacher_median') if s.get('cos_teacher_median') is not None else '—'}</b></div>
  <div class="card"><span class="muted">WER median</span><b>{s.get('wer_median') if s.get('wer_median') is not None else '—'}</b></div>
  <div class="card"><span class="muted">Voice match score</span><b>{s.get('voice_match_score') if s.get('voice_match_score') is not None else '—'}</b></div>
</div>
{preset_table}
<p class="muted">Per-clip table sorted by cos(ref) ascending (worst first). Voice match = median cos(ref) on clips with WER &lt; {WER_PASS} (or all if STT skipped).</p>
<table>
<tr><th>Generated</th>{preset_hdr}<th>Tag</th><th>cos(ref)</th><th>cos(teacher)</th><th>WER</th><th>Teacher WAV</th><th>Text</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ECAPA voice-similarity benchmark for major_03")
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ref-wav", type=Path, default=DEFAULT_REF)
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    _default_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=os.environ.get("BENCH_DEVICE", _default_dev))
    parser.add_argument("--cache-dir", type=Path, default=TRAIN_DIR / ".cache" / "ecapa")
    parser.add_argument("--stt-api", default=STT_API)
    parser.add_argument("--stt-model", default=STT_MODEL)
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--min-teacher-recall", type=float, default=0.55)
    args = parser.parse_args()

    gen_dir = args.gen_dir if args.gen_dir.is_absolute() else TRAIN_DIR / args.gen_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else TRAIN_DIR / args.out_dir
    ref_path = args.ref_wav if args.ref_wav.is_absolute() else ROOT / args.ref_wav
    train_raw = args.train_raw if args.train_raw.is_absolute() else TRAIN_DIR / args.train_raw
    teacher_root = args.teacher_root if args.teacher_root.is_absolute() else TRAIN_DIR / args.teacher_root

    if not gen_dir.is_dir():
        print(f"Missing gen dir: {gen_dir}", file=sys.stderr)
        return 1
    if not ref_path.is_file():
        print(f"Missing ref wav: {ref_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(gen_dir)
    if not manifest:
        print(f"No clips in {gen_dir}", file=sys.stderr)
        return 1

    print(f"Loading ECAPA on {args.device}…")
    encoder = EcapaEncoder(args.device, args.cache_dir)
    ref_emb = encoder.embed(ref_path)
    print(f"Ref embedding: {ref_path}")

    teacher_index = build_teacher_index(train_raw, teacher_root) if train_raw.is_file() else []
    print(f"Teacher index: {len(teacher_index)} clips")

    clips: list[ClipScore] = []
    for row in manifest:
        rel = row["file"]
        wav_path = gen_dir / rel
        if not wav_path.is_file():
            print(f"  skip missing: {rel}")
            continue
        text = str(row.get("text", ""))
        tag = str(row.get("tag", Path(rel).stem))
        preset = row.get("preset")
        rep = row.get("rep")

        gen_emb = encoder.embed(wav_path)
        cos_r = cos_sim(gen_emb, ref_emb)

        teacher_match = find_teacher(text, teacher_index, min_recall=args.min_teacher_recall)
        cos_t: float | None = None
        teacher_wav_s: str | None = None
        teacher_id: str | None = None
        match_score: float | None = None
        if teacher_match:
            t_path, teacher_id, match_score = teacher_match
            teacher_wav_s = str(t_path)
            t_emb = encoder.embed(t_path)
            cos_t = cos_sim(gen_emb, t_emb)

        wer: float | None = None
        wer_pass: bool | None = None
        hyp_snippet = ""
        if not args.no_stt and text.strip():
            try:
                hyp = transcribe_simple(wav_path, args.stt_api, args.stt_model)
                wer = word_error_rate(text, hyp)
                wer_pass = wer <= WER_PASS
                hyp_snippet = hyp[:200]
            except Exception as exc:
                print(f"  STT failed {rel}: {exc}")

        clips.append(
            ClipScore(
                file=rel,
                tag=tag,
                text=text,
                gen_wav=str(wav_path),
                teacher_wav=teacher_wav_s,
                teacher_id=teacher_id,
                teacher_match_score=match_score,
                cos_ref=round(cos_r, 4),
                cos_teacher=round(cos_t, 4) if cos_t is not None else None,
                duration_s=round(wav_duration(wav_path), 2),
                wer=round(wer, 4) if wer is not None else None,
                wer_pass=wer_pass,
                hyp_snippet=hyp_snippet,
                preset=preset,
                rep=int(rep) if rep is not None else None,
            )
        )
        ct_s = f"{cos_t:.3f}" if cos_t is not None else "—"
        print(f"  {rel}: cos_ref={cos_r:.3f} cos_teacher={ct_s}")

    if not clips:
        print("No scored clips.", file=sys.stderr)
        return 1

    cos_refs = [c.cos_ref for c in clips]
    cos_teachers = [c.cos_teacher for c in clips if c.cos_teacher is not None]
    wers = [c.wer for c in clips if c.wer is not None]

    gated = [c.cos_ref for c in clips if c.wer_pass is True or c.wer is None]
    voice_match = statistics.median(gated) if gated else None

    preset_ranking = summarize_presets(clips)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": ECAPA_SOURCE,
        "gen_dir": str(gen_dir),
        "ref_wav": str(ref_path),
        "n": len(clips),
        "cos_ref_mean": round(statistics.mean(cos_refs), 4),
        "cos_ref_median": round(statistics.median(cos_refs), 4),
        "cos_ref_p10": round(pct(cos_refs, 10) or 0.0, 4),
        "cos_teacher_mean": round(statistics.mean(cos_teachers), 4) if cos_teachers else None,
        "cos_teacher_median": round(statistics.median(cos_teachers), 4) if cos_teachers else None,
        "wer_mean": round(statistics.mean(wers), 4) if wers else None,
        "wer_median": round(statistics.median(wers), 4) if wers else None,
        "voice_match_score": round(voice_match, 4) if voice_match is not None else None,
        "stt_enabled": not args.no_stt,
        "preset_ranking": preset_ranking,
        "best_preset": preset_ranking[0]["preset"] if preset_ranking else None,
    }

    scores_path = out_dir / "scores.json"
    payload = {"summary": summary, "preset_ranking": preset_ranking, "clips": [asdict(c) for c in clips]}
    scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_html(out_dir / "report.html", summary, clips, gen_dir, ref_path)

    if preset_ranking and (gen_dir / "manifest.json").is_file():
        _refresh_sweep_listen_index(gen_dir, preset_ranking, clips)

    print()
    print(f"Wrote {scores_path}")
    print(f"Wrote {out_dir / 'report.html'}")
    print(
        f"Summary: n={summary['n']} cos_ref median={summary['cos_ref_median']:.3f} "
        f"mean={summary['cos_ref_mean']:.3f}"
        + (
            f" | cos_teacher median={summary['cos_teacher_median']:.3f}"
            if summary.get("cos_teacher_median") is not None
            else ""
        )
        + (
            f" | voice_match={summary['voice_match_score']:.3f}"
            if summary.get("voice_match_score") is not None
            else ""
        )
    )
    if preset_ranking:
        print("\nPreset ranking (by median cos(ref), best first):")
        for r in preset_ranking:
            ct = f" teacher={r['cos_teacher_median']:.3f}" if r.get("cos_teacher_median") else ""
            print(
                f"  {r['rank']}. {r['preset']}: cos_ref median={r['cos_ref_median']:.3f}"
                f" mean={r['cos_ref_mean']:.3f} min={r['cos_ref_min']:.3f}{ct}"
            )
        print(f"\n→ Best preset by embedding: {summary['best_preset']}")
        print("  (Confirm with listening — ECAPA is a proxy, not MOS.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
