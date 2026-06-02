# major_2_03 Realtime SFT training data

Voice reference: `data/voices/major/major_2_03_cleaned.wav`  
Teacher: **MOSS-TTS v1.5** (openmoss GGML Q8)

## 1. Build text corpus (~3000 clips, 10–30s target)

```bash
./scripts/build_major03_corpus.sh
# → training/major_03/corpus/texts.jsonl
```

## 2. Generate teacher WAVs (4× GPU)

Ensure GPUs are free (`./training/moss-realtime/scripts/legacy/teardown_openmoss.sh`).

```bash
./scripts/run_major03_teacher_gen_parallel.sh
```

Outputs:

| Path | Contents |
|------|----------|
| `wavs/v15/` | v1.5 teacher WAVs |
| `train_raw.jsonl` | Finetune JSONL |
| `teacher_gen.shard*.log` | Per-GPU logs |

Env knobs: `OPENMOSS_MAX_SEC=32`, QC `MIN_DUR=9` `MAX_DUR=32`, staging `/dev/shm/major03_wavs`.

## 3. Later (same as loli pipeline)

- QC/prune → preprocess codes → LoRA SFT → eval/deploy
