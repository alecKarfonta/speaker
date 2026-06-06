#!/usr/bin/env python3
"""Generate MOSS v1.5 teacher WAVs and train_raw.jsonl for Realtime SFT."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
import soundfile as sf

import os
import sys

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
_LEGACY = Path(__file__).resolve().parent
_MOSS_LEGACY = ROOT / "training/moss-realtime/scripts/legacy"
for _p in (_LEGACY, _MOSS_LEGACY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from corpus_wav_names import wav_filename_for_row  # noqa: E402
from generate_voice_clone_batch import (  # noqa: E402
    OPENMOSS_SAMPLING,
    ensure_openmoss_server,
    generate_openmoss,
    generate_openmoss_cli,
    health_url_from_api,
    openmoss_health_ok,
    openmoss_token_budget,
    port_from_api,
    preload_reference,
)
from v15_teacher_styles import instruction_for_style, sampling_for_style  # noqa: E402

STT_API = os.environ.get("STT_API", "http://localhost:8603/v1/audio/transcriptions")
REF_REL = "data/voices/loli/loli_15s.wav"

V15_MODEL = ROOT / "openmoss/weights/moss-tts-v15-q8_0.gguf"
V10_MODEL = ROOT / "openmoss/weights/moss-tts-v10-q8_0.gguf"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(ref: str, hyp: str) -> float:
    ref_w = normalize(ref).split()
    hyp_w = normalize(hyp).split()
    if not ref_w:
        return 1.0 if hyp_w else 0.0
    if not hyp_w:
        return 1.0
    n, m = len(ref_w), len(hyp_w)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def transcribe(wav_path: Path) -> str:
    with wav_path.open("rb") as f:
        r = requests.post(
            STT_API,
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"model": "base", "language": "en", "response_format": "json"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def audio_ok(path: Path, min_dur: float, max_dur: float) -> tuple[bool, float, float]:
    if not path.is_file() or path.stat().st_size < 1024:
        return False, 0.0, 0.0
    x, sr = sf.read(str(path))
    dur = len(x) / sr if sr else 0.0
    peak = float(abs(x).max()) if len(x) else 0.0
    if dur < min_dur or dur > max_dur or peak < 0.01:
        return False, dur, peak
    return True, dur, peak


def set_teacher_model(teacher: str) -> None:
    if teacher == "v15":
        os.environ["OPENMOSS_MODEL_VERSION"] = "v15"
        os.environ["OPENMOSS_MODEL"] = str(V15_MODEL)
    elif teacher == "v10":
        os.environ["OPENMOSS_MODEL_VERSION"] = "v10"
        os.environ["OPENMOSS_MODEL"] = str(V10_MODEL)
    else:
        raise ValueError(teacher)


def synth_opts(teacher: str, text: str, meta: dict | None) -> tuple[str | None, dict | None]:
    """v1.5: instruction + per-style sampling; v1.0: plain clone."""
    if teacher != "v15" or not meta:
        return None, None
    style = meta.get("style", "cheerful")
    instruction = meta.get("instruction") or instruction_for_style(style)
    return instruction, sampling_for_style(style)


def _transient_tts_error(detail: str) -> bool:
    s = detail.lower()
    return any(
        x in s
        for x in (
            "connection refused",
            "remote end closed",
            "connection aborted",
            "remotedisconnected",
            "timed out",
            "timeout",
        )
    )


def synthesize_teacher(
    ref: Path,
    text: str,
    out: Path,
    api: str,
    use_cli: bool,
    teacher: str = "v15",
    meta: dict | None = None,
) -> tuple[bool, str]:
    tokens, max_new = openmoss_token_budget(text, ref)
    instruction, sampling = synth_opts(teacher, text, meta)
    if use_cli:
        return generate_openmoss_cli(ref, text, out, tokens, max_new, instruction=instruction)
    return generate_openmoss(
        ref, text, out, api, tokens, max_new,
        instruction=instruction, sampling=sampling,
    )


def synthesize_teacher_retry(
    ref: Path,
    text: str,
    out: Path,
    api: str,
    use_cli: bool,
    teacher: str = "v15",
    meta: dict | None = None,
    max_retries: int | None = None,
) -> tuple[bool, str]:
    """Retry TTS on connection errors; restart openmoss if health check fails."""
    retries = max_retries if max_retries is not None else int(os.environ.get("TTS_MAX_RETRIES", "5"))
    last_detail = ""
    for attempt in range(retries):
        ok, detail = synthesize_teacher(ref, text, out, api, use_cli, teacher, meta)
        if ok:
            return ok, detail
        last_detail = detail
        if use_cli or not _transient_tts_error(detail) or attempt >= retries - 1:
            return ok, detail
        health = health_url_from_api(api)
        print(f"TTS retry {attempt + 1}/{retries} ({detail[:60]!r})", flush=True)
        recovered = False
        for _ in range(15):
            if openmoss_health_ok(health):
                recovered = True
                break
            time.sleep(2)
        if not recovered:
            gpu = os.environ.get("OPENMOSS_MAIN_GPU", str(port_from_api(api) - 8014))
            os.environ["OPENMOSS_MAIN_GPU"] = gpu
            os.environ["OPENMOSS_PORT"] = str(port_from_api(api))
            ensure_openmoss_server(api, force_restart=True)
        time.sleep(3)
    return False, last_detail


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_id_set(path: Path) -> set[str] | None:
    if not path.is_file():
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_single_row(corpus_id: str, teacher: str, text: str, wav_rel: str) -> dict:
    return {
        "id": f"{corpus_id}_{teacher}",
        "ref_wav": REF_REL,
        "teacher": teacher,
        "conversations": [{"role": "assistant", "text": text, "wav": wav_rel}],
    }


def wav_rel_path(wav_rel_root: Path, teacher: str, filename: str) -> str:
    return str((wav_rel_root / teacher / filename).as_posix())


def load_written_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file():
        return set()
    ids: set[str] = set()
    for line in jsonl_path.read_text(errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except json.JSONDecodeError:
            continue
    return ids


def build_multi_row(corpus_id: str, teacher: str, turns: list[dict], wav_rels: list[str]) -> dict:
    conv = []
    ai = 0
    for turn in turns:
        entry = {"role": turn["role"], "text": turn["text"]}
        if turn["role"] == "assistant":
            entry["wav"] = wav_rels[ai]
            ai += 1
        conv.append(entry)
    return {
        "id": f"{corpus_id}_{teacher}",
        "ref_wav": REF_REL,
        "teacher": teacher,
        "conversations": conv,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training/loli_15s/corpus/texts.jsonl")
    parser.add_argument("--ref", type=Path, default=ROOT / "data/voices/loli/loli_15s.wav")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training/loli_15s")
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="Write WAVs here (default: <out-dir>/wavs). Use tmpfs path to reduce disk IO.",
    )
    parser.add_argument(
        "--wav-rel-root",
        type=Path,
        default=Path("training/loli_15s/wavs"),
        help="Path prefix stored in train_raw.jsonl (under --root)",
    )
    parser.add_argument("--teachers", default="v15")
    parser.add_argument("--api", default="http://127.0.0.1:8014/tts")
    parser.add_argument("--train-ids", type=Path, default=ROOT / "training/loli_15s/corpus/train_ids.txt")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--use-cli", action="store_true", help="Use openmoss CLI instead of HTTP")
    parser.add_argument("--max-wer", type=float, default=0.35)
    parser.add_argument("--min-dur", type=float, default=1.0)
    parser.add_argument("--max-dur", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=0, help="Max corpus rows (0=all)")
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for parallel workers")
    parser.add_argument("--num-shards", type=int, default=1, help="Total parallel shards")
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not start openmoss on first teacher (launcher already started it)",
    )
    parser.add_argument(
        "--voice-qc",
        action="store_true",
        help="ECAPA cos(ref)/cos(teacher) floor (same thresholds as prune QC)",
    )
    parser.add_argument("--min-cos-ref", type=float, default=0.5)
    parser.add_argument("--min-cos-teacher", type=float, default=0.5)
    parser.add_argument("--teacher-train-raw", type=Path, default=None)
    parser.add_argument("--teacher-pool", type=Path, default=None)
    args = parser.parse_args()
    if os.environ.get("VOICE_QC", "").strip() in ("1", "true", "yes"):
        args.voice_qc = True

    if not args.ref.is_file():
        raise SystemExit(f"Reference missing: {args.ref}")

    preload_reference(args.ref)

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    corpus = load_corpus(args.corpus)
    train_ids = load_id_set(args.train_ids)
    if train_ids is not None:
        corpus = [r for r in corpus if r["id"] in train_ids]
    if args.limit:
        corpus = corpus[: args.limit]
    if args.num_shards > 1:
        corpus = [r for i, r in enumerate(corpus) if i % args.num_shards == args.shard_id]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_root = args.wav_dir if args.wav_dir is not None else args.out_dir / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    canonical_wav_root = args.out_dir / "wavs"
    wav_rel_root = args.wav_rel_root
    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    jsonl_path = args.out_dir / f"train_raw{suffix}.jsonl"
    tag = f"[shard {args.shard_id}/{args.num_shards}] " if args.num_shards > 1 else ""

    voice_gate = None
    if args.voice_qc:
        from voice_ecapa_qc import VoiceQcGate, build_teacher_index, default_device

        train_raw = args.teacher_train_raw or (args.out_dir / "train_raw.jsonl")
        if not train_raw.is_absolute():
            train_raw = ROOT / train_raw
        pool = args.teacher_pool or (args.out_dir / "wavs/v15_pruned")
        if not pool.is_absolute():
            pool = ROOT / pool
        index = build_teacher_index(train_raw, pool, root=ROOT)
        cache = args.out_dir / ".cache/ecapa"
        voice_gate = VoiceQcGate(
            ref_wav=args.ref,
            teacher_index=index,
            device=default_device(),
            cache_dir=cache,
            min_cos_ref=args.min_cos_ref,
            min_cos_teacher=args.min_cos_teacher,
        )
        print(
            f"Voice QC at gen: cos(ref/tchr) >= {args.min_cos_ref}, "
            f"pool={len(index)} teachers",
            flush=True,
        )

    stats = {"ok": 0, "fail": 0, "skip": 0, "by_teacher": {t: {"ok": 0, "fail": 0} for t in teachers}}
    written_ids = load_written_ids(jsonl_path) if args.skip_existing else set()
    jsonl_f = jsonl_path.open("a" if written_ids else "w")
    t0 = time.time()

    prev_teacher: str | None = None
    for teacher in teachers:
        set_teacher_model(teacher)
        teacher_dir = wav_root / teacher
        canonical_dir = canonical_wav_root / teacher
        teacher_dir.mkdir(parents=True, exist_ok=True)
        if not args.use_cli and (prev_teacher is not None or not args.no_auto_start):
            ensure_openmoss_server(
                api=args.api,
                force_restart=prev_teacher is not None,
            )
        prev_teacher = teacher

        for item in corpus:
            cid = item["id"]
            wav_paths: list[Path] = []
            texts: list[str] = []
            ok_all = True

            if item["type"] == "single":
                text = item["text"]
                wav = teacher_dir / wav_filename_for_row(item)
                texts = [text]
                meta = item if teacher == "v15" else None
                canon = canonical_dir / wav.name
                if args.skip_existing:
                    for src in (wav, canon):
                        if src.is_file() and src.stat().st_size > 1024:
                            if src != wav:
                                wav.write_bytes(src.read_bytes())
                            stats["skip"] += 1
                            break
                    else:
                        ok, detail = synthesize_teacher_retry(
                            args.ref, text, wav, args.api, args.use_cli, teacher, meta,
                        )
                        if not ok:
                            print(f"{tag}FAIL {teacher}/{cid}: {detail}", flush=True)
                            stats["fail"] += 1
                            stats["by_teacher"][teacher]["fail"] += 1
                            continue
                else:
                    ok, detail = synthesize_teacher_retry(
                        args.ref, text, wav, args.api, args.use_cli, teacher, meta,
                    )
                    if not ok:
                        print(f"{tag}FAIL {teacher}/{cid}: {detail}", flush=True)
                        stats["fail"] += 1
                        stats["by_teacher"][teacher]["fail"] += 1
                        continue
                wav_paths = [wav]
            else:
                for ti, turn in enumerate(item["turns"]):
                    if turn["role"] != "assistant":
                        continue
                    text = turn["text"]
                    turn_row = {**item, "style": turn.get("style", item.get("style")), "text": text}
                    wav = teacher_dir / wav_filename_for_row(turn_row, turn_idx=ti)
                    texts.append(text)
                    meta = turn if teacher == "v15" else None
                    canon = canonical_dir / wav.name
                    skipped = False
                    if args.skip_existing:
                        for src in (wav, canon):
                            if src.is_file() and src.stat().st_size > 1024:
                                if src != wav:
                                    wav.write_bytes(src.read_bytes())
                                stats["skip"] += 1
                                skipped = True
                                break
                    if not skipped:
                        ok, detail = synthesize_teacher_retry(
                            args.ref, text, wav, args.api, args.use_cli, teacher, meta,
                        )
                        if not ok:
                            print(f"{tag}FAIL {teacher}/{cid}_a{ti:02d}: {detail}", flush=True)
                            ok_all = False
                            break
                    wav_paths.append(wav)
                if not ok_all:
                    stats["fail"] += 1
                    stats["by_teacher"][teacher]["fail"] += 1
                    for p in wav_paths:
                        p.unlink(missing_ok=True)
                    continue

            # QC
            for wav, text in zip(wav_paths, texts):
                good, dur, peak = audio_ok(wav, args.min_dur, args.max_dur)
                if not good:
                    print(f"QC fail {wav.name} dur={dur:.2f} peak={peak:.3f}", flush=True)
                    ok_all = False
                    break
                if not args.no_stt:
                    try:
                        hyp = transcribe(wav)
                        wer = word_error_rate(text, hyp)
                        if wer > args.max_wer:
                            print(f"WER fail {wav.name} wer={wer:.2f} hyp={hyp[:60]!r}", flush=True)
                            ok_all = False
                            break
                    except requests.RequestException as exc:
                        print(f"STT skip {wav.name}: {exc}", flush=True)
                if voice_gate is not None:
                    cos_r, cos_t, _, vreasons = voice_gate.check(wav, text)
                    if vreasons:
                        ct = f"{cos_t:.3f}" if cos_t is not None else "—"
                        print(
                            f"Voice QC fail {wav.name} ref={cos_r:.3f} tchr={ct} "
                            f"{','.join(vreasons)}",
                            flush=True,
                        )
                        ok_all = False
                        break

            if not ok_all:
                stats["fail"] += 1
                stats["by_teacher"][teacher]["fail"] += 1
                for p in wav_paths:
                    p.unlink(missing_ok=True)
                continue

            wav_rels = [
                wav_rel_path(wav_rel_root, teacher, p.name) for p in wav_paths
            ]
            if item["type"] == "single":
                row = build_single_row(cid, teacher, item["text"], wav_rels[0])
            else:
                row = build_multi_row(cid, teacher, item["turns"], wav_rels)
            row_id = row["id"]
            if row_id in written_ids:
                continue
            jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_f.flush()
            os.fsync(jsonl_f.fileno())
            written_ids.add(row_id)
            stats["ok"] += 1
            stats["by_teacher"][teacher]["ok"] += 1
            if stats["ok"] % 25 == 0:
                print(f"{tag}  {stats['ok']} accepted ({teacher} {cid})", flush=True)

    jsonl_f.close()

    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["train_raw"] = str(jsonl_path)
    stats["n_rows"] = len(written_ids)
    stats_path = args.out_dir / f"dataset_stats{suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0 if stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
