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

## 2b. Grow corpus (optional, no ID overlap)

```bash
COUNT=7000 ./scripts/build_major03_supplement.sh
# appends maj2_st_* lines → corpus/texts.jsonl (~10k total)
```

## 2c. Reconcile corpus + train_raw (after gap fill)

Duplicate `st_*` ids in an older corpus were filled with `*_dup*.wav` clips. Run once to align `texts.jsonl` and dedupe `train_raw.jsonl`:

```bash
python3 training/major_03/scripts/reconcile_major03_dataset.py
# → dataset_reconcile_stats.json (expect 7146 corpus / wav / train_raw rows)
```

## 3. Dataset status (teacher gen complete)

| Artifact | Count |
|----------|------:|
| `corpus/texts.jsonl` | 7146 unique lines |
| `wavs/v15/*.wav` | 7146 |
| `train_raw.jsonl` | 7146 |

Next: QC/prune → preprocess codes → LoRA SFT (same as loli).

## Production decode (warm_092_072)

Native MOSS-RT defaults when the client omits sampling params:

| Param | Value |
|-------|------:|
| `audio_temperature` | 0.92 |
| `audio_top_p` | 0.72 |
| `audio_top_k` | 40 |
| `audio_repetition_penalty` | 1.05 |

Set via `MOSS_RT_AUDIO_*` in `scripts/start-moss-realtime.sh` / `app/moss_api.py`. Old 0.8/0.6 preset: `--preset legacy_08_06` in eval scripts.

## 4. Later (same as loli pipeline)

- QC/prune (`--end-buffer-ms 500` default — avoids clipping final words):

```bash
MOSS_RT_TRAIN_DIR=$PWD/training/major_03 \
  python training/moss-realtime/scripts/distill.py qc prune --end-buffer-ms 500
```

- Then preprocess codes → LoRA SFT → eval/deploy

## 5. Voice similarity benchmark (ECAPA)

Scores native eval WAVs vs reference enrollment and nearest teacher clip (text match from `train_raw.jsonl`).

```bash
chmod +x training/major_03/scripts/run_voice_similarity_bench.sh
./training/major_03/scripts/run_voice_similarity_bench.sh
# → eval/bench/epoch11_native/scores.json + report.html

GEN_DIR=eval/listen/epoch11_default OUT_DIR=eval/bench/epoch11_default \
  ./training/major_03/scripts/run_voice_similarity_bench.sh

NO_STT=1 ./training/major_03/scripts/run_voice_similarity_bench.sh   # embeddings only

**Pick decode params** (after `generate_eval_samples.py --sweep`):

```bash
SWEEP=1 NO_STT=1 ./training/major_03/scripts/run_voice_similarity_bench.sh
# → eval/bench/epoch11_sampling_sweep/report.html with preset ranking table
```

Rank presets by median **cos(ref)** across the 4 anchor clips; use listening to confirm (ECAPA ≠ subjective quality).

Full sweep regen + scored listen page:

```bash
./training/major_03/scripts/run_sweep_eval.sh
# → eval/listen/epoch11_sampling_sweep/index.html (audio + cos ref/teacher per cell)
```

**Hot-zone sweep** (fine grid around T=1.0 p=0.75 k=50, story-heavy clips):

```bash
./training/major_03/scripts/run_hot_zone_eval.sh
# → eval/listen/epoch11_hot_zone_sweep/index.html
```

**Target sweep** (T≈0.92 p≈0.72 k=40, story-only, 2 runs per cell for consistency):

```bash
./training/major_03/scripts/run_target_eval.sh
# → eval/listen/epoch11_target_sweep/index.html (preset sections + ref/tchr scores)
```
