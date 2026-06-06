# Loli v2 pipeline

Supplement ~2k emotion-heavy teachers, trim-only QC (750ms buffer), resume SFT from epoch-7, serve with warm_092.

## Quick start

```bash
# Phase A (no GPU): warm decode is default in start-moss-realtime.sh
MOSS_RT_API=http://127.0.0.1:8016 python training/loli_15s/scripts/verify_streaming_stack.py

# Phase B–C: corpus already at training/loli_15s_batch3/corpus/texts.jsonl
training/loli_15s/scripts/build_loli_batch3.sh          # rebuild if needed
training/loli_15s/scripts/run_loli_batch3_teacher_gen.sh  # 4× GPU, hours
training/loli_15s/scripts/merge_batch3_into_loli15s.sh
training/loli_15s/scripts/run_loli_v2_qc.sh  # STT trim + ECAPA cos(ref/tchr) ≥ 0.5

# Phase D
training/loli_15s/scripts/run_sft_v2.sh

# Phase E (server on v2 merged)
MOSS_RT_MODEL_ID=training/loli_15s/exports/loli15s-v2-merged ./scripts/start-moss-realtime.sh
training/loli_15s/scripts/run_eval_v2.sh
```

Or run end-to-end (GPU): `training/loli_15s/scripts/run_loli_v2_pipeline.sh`

## Artifacts

| Path | Role |
|------|------|
| `training/loli_15s_batch3/` | Supplement corpus + teacher WAVs |
| `exports/loli15s-v2-merged` | Production candidate after v2 SFT |
| `eval/bench/verify_streaming_*.json` | Last-word / tail-gap metrics |
