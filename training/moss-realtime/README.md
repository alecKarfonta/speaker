# MOSS-TTS-Realtime distillation

Distill a cloned voice into **MOSS-TTS-Realtime** (LoRA SFT → merge → serve).  
Inference: [docs/moss-realtime.md](../../docs/moss-realtime.md).

## Quick start

```bash
cd /path/to/speaker
export MOSS_RT_TRAIN_DIR=$PWD/training/moss-realtime   # or training/loli_15s for existing runs
cp training/moss-realtime/configs/experiment.yaml.example training/moss-realtime/configs/experiment.yaml

python training/moss-realtime/scripts/distill.py env setup
python training/moss-realtime/scripts/distill.py corpus build
python training/moss-realtime/scripts/distill.py teacher gen --parallel
python training/moss-realtime/scripts/distill.py qc prune
python training/moss-realtime/scripts/distill.py train preprocess --noref
python training/moss-realtime/scripts/distill.py train sft --noref
python training/moss-realtime/scripts/distill.py export merge
python training/moss-realtime/scripts/distill.py export onnx
python training/moss-realtime/scripts/distill.py serve
```

## Layout

```
training/moss-realtime/
  configs/experiment.yaml      # your run (gitignored if under experiment dir data)
  configs/experiment.yaml.example
  scripts/distill.py           # single CLI entry
  scripts/lib/                 # paths, config, runner
  scripts/legacy/              # corpus, teacher, QC, openmoss helpers
  scripts/legacy/finetune/     # preprocess, SFT, merge
  scripts/legacy/bench/        # RTF / sample generation
  corpus/ wavs/ prepared/ ...  # gitignored artifacts
```

Existing **loli_15s** data under `training/loli_15s/` still works — set `MOSS_RT_TRAIN_DIR=$PWD/training/loli_15s`.

## Commands

| Command | Purpose |
|---------|---------|
| `teacher gen --parallel` | MOSS v1.5 teacher WAVs → `train_raw.jsonl` |
| `teacher resume` | Continue after interrupt |
| `teacher teardown` | Kill orphan openmoss servers |
| `qc report` / `qc prune` | STT tail-trim + bad-clip filter |
| `train sft --noref` | Native-voice LoRA (no ref WAV at inference) |
| `export merge` | LoRA → bf16 for ~1.7× RTF serve |
| `eval samples` | Listening matrix under `eval/listen/` |

Requires **openmoss** GGML locally (`openmoss/`, gitignored) for teacher generation.
