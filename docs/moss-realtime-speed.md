# MOSS-RT speed notes

See [training/loli_15s/docs/serve-production.md](../training/loli_15s/docs/serve-production.md) for production deploy.

```bash
python training/moss-realtime/scripts/distill.py bench rtf --api-url http://127.0.0.1:8016
python training/loli_15s/scripts/generate_eval_samples.py
```

## Reference (RTX 3090 Ti, loli15s epoch-7 merged, warm, ttfa_fast env)

| Path | Short TTFA | Sustained RTF (long) |
|------|------------|----------------------|
| `/tts/stream` (torch codec, ttfa_fast) | **~370 ms** | ~1.4–1.6× |
| `/tts/stream` (old steady=24 defaults) | ~1215 ms | ~1.5× |
| `/tts` batch (ONNX codec) | n/a (no streaming) | ~1.7× |

`torch.compile` backbone: **not viable** on current Realtime streaming stack (see `training/loli_15s/eval/bench/compile_attempt_epoch7.json`).
