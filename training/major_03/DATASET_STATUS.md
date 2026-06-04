# major_03 teacher dataset status

Last updated: 2026-06-03 (reconciled)

## Teacher generation: complete

- **7146** MOSS v1.5 teacher WAVs (`wavs/v15/`)
- **7146** corpus lines with **unique** ids (`corpus/texts.jsonl`)
- **7146** finetune rows (`train_raw.jsonl`)
- Reference voice: `data/voices/major/major_2_03_cleaned.wav`

Includes **30** clips that were duplicate-ID corpus rows; each now has a permanent `st_*_dupNN` id and matching WAV.

## Fixes applied

1. Gap-fill run (2× GPU): generated missing `_dup*.wav` files.
2. `scripts/reconcile_major03_dataset.py`: merged gap ids into corpus, removed 25 duplicate `train_raw` rows.

Backups: `corpus/texts.jsonl.bak.*`, `train_raw.jsonl.bak.*`

## QC (trim-only v2, 2026-06-03): complete

| | Count |
|--|------:|
| Scanned | 7146 |
| pass + trim (kept) | 6816 |
| quarantined (WER>0.75 etc.) | 330 |
| `wavs/v15_pruned/` | 6816 |
| `train_raw.noref.jsonl` | 6816 |

## Preprocess: run before SFT

```bash
NUM_GPUS=4 ./training/major_03/scripts/run_preprocess_only.sh
# → prepared/train_with_codes.rank*.jsonl
```

## Not started yet

- LoRA SFT on MOSS-Realtime (needs prepared shards)

## Quick checks

```bash
# Counts should all be 7146
wc -l training/major_03/corpus/texts.jsonl training/major_03/train_raw.jsonl
find training/major_03/wavs/v15 -name '*.wav' | wc -l

python3 training/major_03/scripts/reconcile_major03_dataset.py  # idempotent if already aligned
```

## Resume teacher gen (only if gaps reappear)

```bash
NUM_SHARDS=2 GPUS=0,1 PORTS=8014,8015 \
  ./training/major_03/scripts/run_major03_teacher_gen_parallel.sh
```

Use `--skip-existing` (default) and a gap corpus under `corpus/texts_gaps.jsonl`.
