# MOSS-RT speed notes

See [training/moss-realtime/README.md](../training/moss-realtime/README.md) and:

```bash
python training/moss-realtime/scripts/distill.py bench rtf --api-url http://127.0.0.1:8016
python training/moss-realtime/scripts/distill.py eval samples
```

## Reference (RTX 3090 Ti, merged LoRA, warm)

| Path | Sustained RTF | TTFA |
|------|---------------|------|
| `/tts/stream` (torch codec) | ~2× | ~1.15s |
| `/tts` batch (ONNX codec) | ~1.7× | n/a |
