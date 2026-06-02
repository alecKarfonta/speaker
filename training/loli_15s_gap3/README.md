# loli_15s gap fill (batch 3)

Targeted teacher-data generation to address weaknesses in the current ~5.2k / ~12h corpus:

| Gap | Current (~) | Gap3 target |
|-----|-------------|-------------|
| Long utterances (≥10s audio) | Few (median ~5s) | **40%** long scripts (150–350 chars) |
| Numbers / dates / counts | Low digit coverage | **15%** |
| Names & places | Template-only | **15%** |
| Questions | Some | **15%** explicit `?` prompts |
| Emotion / tone variety | Mostly cozy-neutral | **15%** excited, gentle, curious, etc. |

**Does not touch** `training/loli_15s/wavs/v15/` originals or batch2 trees until merge.

| | Batch 2 | Gap 3 (this) |
|--|---------|----------------|
| Dir | `training/loli_15s_batch2/` | `training/loli_15s_gap3/` |
| Tmpfs | `/dev/shm/loli15s_batch2_wavs` | `/dev/shm/loli15s_gap3_wavs` |
| IDs | `b2_st_*` | `g3_st_*` |
| Default size | 3000 | **2500** (~+5.5h audio) |

## Prep (now)

```bash
./training/loli_15s_gap3/scripts/build_corpus.sh
# → corpus/texts.jsonl, corpus_stats.json, gap_report.json
```

## After current SFT finishes (automatic chain)

```bash
nohup ./training/loli_15s_gap3/scripts/wait_sft_then_run.sh \
  >> training/loli_15s_gap3/logs/wait_sft_then_run.log 2>&1 &
```

This waits for `output/sft_ddp_single` training to exit, builds corpus, runs **4-GPU teacher gen** (`--qc-lenient`, `MIN_DUR=5`), then merges into `training/loli_15s/`.

## Manual steps

```bash
./training/loli_15s_gap3/scripts/build_corpus.sh
./training/loli_15s_gap3/scripts/run_teacher_gen_parallel.sh
./training/loli_15s_gap3/scripts/finish_merge_to_loli15s.sh
```

## After merge → second SFT pass

```bash
./training/loli_15s/scripts/filter_single_turn_train_raw.sh
python3 training/loli_15s/scripts/distill.py qc trim --trim-only
./training/loli_15s/scripts/run_sft_4gpu.sh   # full preprocess + SFT
```

## Env knobs

| Var | Default | Notes |
|-----|---------|-------|
| `TOTAL` | 2500 | Corpus lines |
| `MIX` | long:0.40,... | Category weights |
| `MIN_DUR` | 5.0 | Teacher QC floor (lenient) |
| `TEACHER_GEN_EXTRA_ARGS` | `--qc-lenient` | Keep short number/question clips |
