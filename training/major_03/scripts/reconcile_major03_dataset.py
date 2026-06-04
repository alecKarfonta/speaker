#!/usr/bin/env python3
"""Merge gap clips into corpus and dedupe train_raw for major_03."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAJOR = ROOT / "training/major_03"
CORPUS = MAJOR / "corpus/texts.jsonl"
GAPS = MAJOR / "corpus/texts_gaps.jsonl"
TRAIN = MAJOR / "train_raw.jsonl"
WAV_DIR = MAJOR / "wavs/v15"
REF = "data/voices/major/major_2_03_cleaned.wav"
WAV_REL_ROOT = Path("training/major_03/wavs")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def merge_corpus(corpus_rows: list[dict], gap_rows: list[dict]) -> list[dict]:
    gap_by_text: dict[str, list[dict]] = {}
    for g in gap_rows:
        gap_by_text.setdefault(g["text"].strip().lower(), []).append(g)
    gap_by_base: dict[str, list[dict]] = {}
    for g in gap_rows:
        base = g["id"].split("_dup")[0]
        gap_by_base.setdefault(base, []).append(g)
    used_gap: set[str] = set()

    def take_gap(row: dict) -> dict:
        key = row["text"].strip().lower()
        for pool in (gap_by_text.get(key, []), gap_by_base.get(row["id"], [])):
            while pool:
                cand = pool.pop(0)
                if cand["id"] not in used_gap:
                    used_gap.add(cand["id"])
                    return cand
        raise RuntimeError(f"No gap row for duplicate corpus id={row['id']}")

    seen: set[str] = set()
    out: list[dict] = []
    for row in corpus_rows:
        rid = row["id"]
        if rid not in seen:
            seen.add(rid)
            out.append(row)
            continue
        repl = take_gap(row)
        seen.add(repl["id"])
        out.append(repl)
    return out


def dedupe_train_raw(rows: list[dict]) -> list[dict]:
    by_wav: dict[str, dict] = {}
    for row in rows:
        conv = row.get("conversations") or []
        if not conv:
            continue
        wav = conv[0].get("wav", "")
        if not wav:
            continue
        prev = by_wav.get(wav)
        if prev is None:
            by_wav[wav] = row
            continue
        # Prefer entry whose id matches _dup wav stem when wav path contains _dup
        stem = Path(wav).stem
        if "_dup" in stem and "_dup" in row["id"]:
            by_wav[wav] = row
    return list(by_wav.values())


def rebuild_missing_train_rows(corpus_rows: list[dict], train_rows: list[dict]) -> list[dict]:
    by_corpus_id = {}
    for row in train_rows:
        tid = row["id"]
        cid = tid[:-4] if tid.endswith("_v15") else tid
        by_corpus_id[cid] = row

    out = []
    for item in corpus_rows:
        cid = item["id"]
        wav_name = f"{cid}.wav"
        wav_rel = str((WAV_REL_ROOT / "v15" / wav_name).as_posix())
        if not (WAV_DIR / wav_name).is_file():
            raise RuntimeError(f"Missing wav for corpus id {cid}: {wav_name}")
        if cid in by_corpus_id:
            row = by_corpus_id[cid]
            conv = row["conversations"][0]
            if conv.get("text") != item["text"] or conv.get("wav") != wav_rel:
                row = {
                    "id": f"{cid}_v15",
                    "ref_wav": REF,
                    "teacher": "v15",
                    "conversations": [
                        {"role": "assistant", "text": item["text"], "wav": wav_rel}
                    ],
                }
            out.append(row)
        else:
            out.append(
                {
                    "id": f"{cid}_v15",
                    "ref_wav": REF,
                    "teacher": "v15",
                    "conversations": [
                        {"role": "assistant", "text": item["text"], "wav": wav_rel}
                    ],
                }
            )
    return out


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corpus_rows = load_jsonl(CORPUS)
    gap_rows = load_jsonl(GAPS) if GAPS.is_file() else []
    train_rows = load_jsonl(TRAIN)

    for p in (CORPUS, TRAIN):
        bak = p.with_suffix(p.suffix + f".bak.{ts}")
        shutil.copy2(p, bak)
        print(f"backup: {bak}")

    merged = merge_corpus(corpus_rows, gap_rows)
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise SystemExit(f"corpus still has duplicate ids: {dupes[:10]}")

    deduped = dedupe_train_raw(train_rows)
    aligned = rebuild_missing_train_rows(merged, deduped)

    write_jsonl(CORPUS, merged)
    write_jsonl(TRAIN, aligned)

    wavs = {p.stem for p in WAV_DIR.glob("*.wav")}
    corpus_set = set(ids)
    stats = {
        "reconciled_at": ts,
        "corpus_lines": len(merged),
        "corpus_unique_ids": len(set(ids)),
        "train_raw_lines": len(aligned),
        "wav_files": len(wavs),
        "corpus_ids_without_wav": sorted(corpus_set - wavs)[:20],
        "wavs_without_corpus_id": len(wavs - corpus_set),
        "train_without_wav": sum(
            1
            for r in aligned
            if not (ROOT / r["conversations"][0]["wav"]).is_file()
        ),
        "corpus_without_train": sorted(
            corpus_set
            - {r["id"][:-4] if r["id"].endswith("_v15") else r["id"] for r in aligned}
        )[:20],
    }
    stats_path = MAJOR / "dataset_reconcile_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    if stats["corpus_ids_without_wav"] or stats["train_without_wav"] or stats["corpus_without_train"]:
        raise SystemExit("Reconcile incomplete — see dataset_reconcile_stats.json")

    print("OK: corpus, wavs, and train_raw are aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
