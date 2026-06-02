# loli_15s teacher batch 2 (3000 clips)

Second MOSS v1.5 teacher dataset for `data/voices/loli_15s/loli_15s.wav`.

**Does not touch the first run.** The original ~3.2k clips stay in `training/loli_15s/wavs/v15/` (`st_*.wav`). This batch uses:

| | Original run | Batch 2 (this) |
|--|--|--|
| Output dir | `training/loli_15s/` | `training/loli_15s_batch2/` |
| Tmpfs staging | `/dev/shm/loli15s_wavs` | `/dev/shm/loli15s_batch2_wavs` |
| Clip IDs | `st_*` | `b2_st_*` |
| train_raw | `training/loli_15s/train_raw.jsonl` | `training/loli_15s_batch2/train_raw.jsonl` |

## Manual run

```bash
./training/loli_15s_batch2/scripts/build_corpus.sh
./training/loli_15s_batch2/scripts/run_teacher_gen_parallel.sh
```

## Chained after major_03

```bash
nohup ./training/major_03/scripts/chain_loli15s_after_major03.sh \
  >> training/major_03/chain_loli15s.log 2>&1 &
```

Outputs: `corpus/texts.jsonl`, `wavs/v15/`, `train_raw.jsonl`, `teacher_gen.shard*.log`.
