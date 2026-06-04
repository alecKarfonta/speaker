#!/usr/bin/env python3
"""Local MOSS-TTS-Realtime distillation pipeline (single entry point).

All artifacts live under `$MOSS_RT_TRAIN_DIR` (default: `training/moss-realtime/`). Corpus WAVs and checkpoints are gitignored.

Usage (from repo root):
  python training/moss-realtime/scripts/distill.py env setup
  python training/moss-realtime/scripts/distill.py corpus build
  python training/moss-realtime/scripts/distill.py teacher gen --parallel
  python training/loli_15s/scripts/distill.py qc prune
  python training/loli_15s/scripts/distill.py train preprocess
  python training/loli_15s/scripts/distill.py train sft --noref
  python training/loli_15s/scripts/distill.py export merge
  python training/loli_15s/scripts/distill.py eval samples
  python training/loli_15s/scripts/distill.py bench rtf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.config import ExperimentConfig
from lib.paths import Paths
from lib.runner import run_cmd, run_python, run_shell


def _paths(args: argparse.Namespace) -> Paths:
    train = Path(args.train_dir).resolve() if args.train_dir else None
    return Paths.load(train)


def _cfg(paths: Paths) -> ExperimentConfig:
    return ExperimentConfig.load(paths.configs_dir / "experiment.yaml").resolve(paths.train_dir)


def cmd_env_setup(paths: Paths, _args: argparse.Namespace) -> int:
    return run_shell(paths, paths.legacy_dir / "setup_moss_tts_finetune_env.sh")


def cmd_corpus_build(paths: Paths, args: argparse.Namespace) -> int:
    rc = run_python(paths, paths.legacy_dir / "build_loli15s_corpus.py", [])
    if rc != 0:
        return rc
    if args.enrich:
        rc = run_python(paths, paths.legacy_dir / "enrich_loli15s_corpus_styles.py", [])
        if rc != 0:
            return rc
    return run_python(paths, paths.legacy_dir / "split_finetune_corpus.py", [])


def cmd_teacher_gen(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    if args.teardown_first:
        run_shell(paths, paths.legacy_dir / "teardown_openmoss.sh")
    if args.light_host:
        run_shell(paths, paths.legacy_dir / "lighten_host_for_teacher_gen.sh", ["stop"])

    if args.parallel:
        extra = {
            "NUM_SHARDS": str(args.shards),
            "GPUS": args.gpus,
            "PORTS": args.ports,
            "TEACHERS": cfg.teacher,
        }
        return run_shell(
            paths,
            paths.legacy_dir / "run_loli15s_teacher_gen_parallel.sh",
            extra_env=extra,
        )

    ref = Path(str(cfg.voice_ref))
    if not ref.is_absolute():
        ref = paths.repo_root / ref
    corpus = Path(cfg.corpus)
    out_args = [
        "--corpus", str(corpus),
        "--ref", str(ref),
        "--teachers", cfg.teacher,
        "--out-dir", str(paths.train_dir),
        "--skip-existing",
    ]
    return run_python(paths, paths.legacy_dir / "build_realtime_finetune_dataset.py", out_args)


def cmd_teacher_teardown(paths: Paths, _args: argparse.Namespace) -> int:
    return run_shell(paths, paths.legacy_dir / "teardown_openmoss.sh")


def cmd_teacher_resume(paths: Paths, args: argparse.Namespace) -> int:
    extra: dict[str, str] = {}
    if args.no_reuse_moss:
        extra["REUSE_MOSS"] = "0"
    return run_shell(paths, paths.legacy_dir / "resume_loli15s_teacher_gen.sh", extra_env=extra)


def cmd_qc_report(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    wav_dir = Path(cfg.qc.get("wav_dir", paths.train_dir / "wavs/v15"))
    if not wav_dir.is_absolute():
        wav_dir = paths.train_dir / wav_dir
    extra = ["--limit", str(args.limit), "--wav-dir", str(wav_dir)]
    return run_python(paths, paths.legacy_dir / "prune_loli15s_teacher_dataset.py", extra)


def cmd_qc_prune(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    wav_dir = Path(cfg.qc.get("wav_dir", "wavs/v15"))
    out_dir = Path(cfg.qc.get("wav_dir_pruned", "wavs/v15_pruned"))
    if not wav_dir.is_absolute():
        wav_dir = paths.train_dir / wav_dir
    if not out_dir.is_absolute():
        out_dir = paths.train_dir / out_dir
    extra = ["--apply", "--wav-dir", str(wav_dir), "--out-dir", str(out_dir)]
    if args.limit:
        extra.extend(["--limit", str(args.limit)])
    if args.end_buffer_ms is not None:
        extra.extend(["--end-buffer-ms", str(args.end_buffer_ms)])
    return run_python(paths, paths.legacy_dir / "prune_loli15s_teacher_dataset.py", extra)


def cmd_train_preprocess(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    train_raw = Path(cfg.train_raw)
    if not train_raw.is_absolute():
        train_raw = paths.train_dir / train_raw
    extra = {
        "TRAIN_RAW": str(train_raw),
        "NUM_GPUS": str(args.gpus),
    }
    if args.noref:
        noref = paths.train_dir / "train_raw.noref.jsonl"
        if not noref.is_file():
            run_python(paths, paths.finetune_dir / "build_moss_rt_train_raw_noref.py", [])
        extra["TRAIN_RAW"] = str(noref)
    return run_shell(paths, paths.finetune_dir / "run_moss_rt_finetune_preprocess.sh", extra_env=extra)


def cmd_train_sft(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    sft = cfg.sft
    extra = {
        "NUM_GPUS": str(args.gpus),
        "NUM_EPOCHS": str(args.epochs or sft.get("epochs", 12)),
        "LEARNING_RATE": str(args.lr or sft.get("lr", 3e-6)),
        "GRAD_ACCUM": str(sft.get("grad_accum", 4)),
        "WARMUP_RATIO": str(sft.get("warmup_ratio", 0)),
    }
    if args.noref or sft.get("noref"):
        return run_shell(paths, paths.finetune_dir / "run_moss_rt_finetune_noref.sh", extra_env=extra)
    return run_shell(paths, paths.finetune_dir / "run_moss_rt_finetune_train.sh", extra_env=extra)


def cmd_export_merge(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    ckpt = Path(cfg.export.get("checkpoint", paths.train_dir / "checkpoints/latest"))
    out = Path(cfg.export.get("merged", paths.train_dir / "exports/merged"))
    if not ckpt.is_absolute():
        ckpt = paths.train_dir / ckpt
    if not out.is_absolute():
        out = paths.train_dir / out
    merge_args = ["--checkpoint", str(ckpt), "--output", str(out)]
    if args.base:
        merge_args.extend(["--base", args.base])
    return run_python(paths, paths.finetune_dir / "merge_moss_rt_lora.py", merge_args)


def cmd_export_onnx(paths: Paths, _args: argparse.Namespace) -> int:
    return run_shell(paths, paths.finetune_dir / "download_moss_onnx_codec.sh")


def cmd_eval_samples(paths: Paths, args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    api = args.api_url or cfg.sft.get("serve_api") or "http://127.0.0.1:8016"
    if hasattr(cfg, "serve"):
        pass
    exp_path = paths.configs_dir / "experiment.yaml"
    if exp_path.is_file():
        import yaml

        raw = yaml.safe_load(exp_path.read_text()) or {}
        api = args.api_url or raw.get("serve", {}).get("api_url", api)
    out = paths.train_dir / "eval/listen/rt_merged_demo"
    out.mkdir(parents=True, exist_ok=True)
    return run_python(
        paths,
        paths.bench_dir / "generate_rt_stream_samples.py",
        ["--api-url", api, "--out", str(out)],
    )


def cmd_eval_listen(paths: Paths, args: argparse.Namespace) -> int:
    api = args.api_url or "http://127.0.0.1:8016"
    return run_python(
        paths,
        paths.legacy_dir / "eval_loli15s_finetune.py",
        ["--baseline-url", api, "--out-dir", str(paths.train_dir / "eval")],
    )


def cmd_bench_rtf(paths: Paths, args: argparse.Namespace) -> int:
    api = args.api_url or "http://127.0.0.1:8016"
    return run_python(paths, paths.bench_dir / "benchmark_rt_tts.py", ["--url", api])


def cmd_bench_sweep(paths: Paths, args: argparse.Namespace) -> int:
    api = args.api_url or "http://127.0.0.1:8016"
    return run_python(
        paths,
        paths.bench_dir / "sweep_rt_chunk_sizes.py",
        ["--api-url", api, *([] if not args.quick else ["--quick"])],
    )


def cmd_serve_hint(paths: Paths, _args: argparse.Namespace) -> int:
    cfg = _cfg(paths)
    merged = Path(cfg.export.get("merged", paths.train_dir / "exports/merged"))
    if not merged.is_absolute():
        merged = paths.train_dir / merged
    native = cfg.sft.get("native_voice", True)
    print("Serve merged checkpoint:")
    print(f"  MOSS_RT_MODEL_ID={merged} MOSS_RT_NATIVE_VOICE={'true' if native else 'false'} \\")
    print("    ./scripts/start-moss-realtime.sh")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--train-dir",
        default=os.environ.get("MOSS_RT_TRAIN_DIR", ""),
        help="Experiment dir (default: training/loli_15s)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="Bootstrap MOSS-TTS finetune venv")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    env_sub.add_parser("setup", help="Clone MOSS-TTS + pip install")

    corpus = sub.add_parser("corpus", help="Text corpus for teacher generation")
    corpus_sub = corpus.add_subparsers(dest="corpus_cmd", required=True)
    cb = corpus_sub.add_parser("build", help="Build + split corpus")
    cb.add_argument("--enrich", action="store_true", default=True, help="Add v1.5 style tags (default)")
    cb.add_argument("--no-enrich", action="store_false", dest="enrich")

    teacher = sub.add_parser("teacher", help="MOSS v1.5 teacher / distillation")
    teacher_sub = teacher.add_subparsers(dest="teacher_cmd", required=True)
    tg = teacher_sub.add_parser("gen", help="Generate teacher WAVs + train_raw.jsonl")
    tg.add_argument("--parallel", action="store_true", help="4-GPU parallel openmoss shards")
    tg.add_argument("--teardown-first", action="store_true", default=True)
    tg.add_argument("--no-teardown-first", action="store_false", dest="teardown_first")
    tg.add_argument("--light-host", action="store_true", help="Pause heavy Docker during gen")
    tg.add_argument("--shards", type=int, default=4)
    tg.add_argument("--gpus", default="0,1,2,3")
    tg.add_argument("--ports", default="8014,8015,8016,8017")
    teacher_sub.add_parser("teardown", help="Kill orphan openmoss servers")
    tr = teacher_sub.add_parser("resume", help="Resume teacher gen (reuse live moss)")
    tr.add_argument("--no-reuse-moss", action="store_true", help="Restart moss if unhealthy")

    qc = sub.add_parser("qc", help="Teacher WAV cleanup")
    qc_sub = qc.add_subparsers(dest="qc_cmd", required=True)
    qr = qc_sub.add_parser("report", help="QC report only")
    qr.add_argument("--limit", type=int, default=100)
    qp = qc_sub.add_parser("prune", help="STT tail-prune + quarantine bad clips")
    qp.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)
    qp.add_argument("--limit", type=int, default=0)
    qp.add_argument(
        "--end-buffer-ms",
        type=int,
        default=None,
        help="Tail trim buffer after last STT word (default: prune script default)",
    )

    train = sub.add_parser("train", help="LoRA SFT on Realtime model")
    train_sub = train.add_subparsers(dest="train_cmd", required=True)
    tp = train_sub.add_parser("preprocess", help="Audio codes (prepare_data.py)")
    tp.add_argument("--gpus", type=int, default=4)
    tp.add_argument("--noref", action="store_true", help="Use train_raw.noref.jsonl")
    ts = train_sub.add_parser("sft", help="LoRA fine-tune")
    ts.add_argument("--noref", action="store_true", default=True)
    ts.add_argument("--gpus", type=int, default=4)
    ts.add_argument("--epochs", type=int, default=None)
    ts.add_argument("--lr", type=float, default=None)

    export = sub.add_parser("export", help="Post-train artifacts")
    export_sub = export.add_subparsers(dest="export_cmd", required=True)
    export_sub.add_parser("merge", help="Merge LoRA into bf16 weights").add_argument("--base", default="")
    export_sub.add_parser("onnx", help="Download ONNX codec weights")

    eval_p = sub.add_parser("eval", help="Listening tests")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_sub.add_parser("samples", help="Generate batch+stream WAV matrix").add_argument("--api-url", default="")
    el = eval_sub.add_parser("listen", help="Baseline vs finetuned eval")
    el.add_argument("--api-url", default="")

    bench = sub.add_parser("bench", help="Performance regression")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)
    bench_sub.add_parser("rtf", help="POST /tts RTF").add_argument("--api-url", default="")
    bs = bench_sub.add_parser("sweep", help="Chunk size sweep")
    bs.add_argument("--api-url", default="")
    bs.add_argument("--quick", action="store_true")

    sub.add_parser("serve", help="Print serve command for merged checkpoint")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = _paths(args)

    handlers: dict[tuple[str, str | None], object] = {
        ("env", "setup"): cmd_env_setup,
        ("corpus", "build"): cmd_corpus_build,
        ("teacher", "gen"): cmd_teacher_gen,
        ("teacher", "teardown"): cmd_teacher_teardown,
        ("teacher", "resume"): cmd_teacher_resume,
        ("qc", "report"): cmd_qc_report,
        ("qc", "prune"): cmd_qc_prune,
        ("train", "preprocess"): cmd_train_preprocess,
        ("train", "sft"): cmd_train_sft,
        ("export", "merge"): cmd_export_merge,
        ("export", "onnx"): cmd_export_onnx,
        ("eval", "samples"): cmd_eval_samples,
        ("eval", "listen"): cmd_eval_listen,
        ("bench", "rtf"): cmd_bench_rtf,
        ("bench", "sweep"): cmd_bench_sweep,
        ("serve", None): cmd_serve_hint,
    }

    sub_cmd = getattr(args, f"{args.command}_cmd", None)
    key = (args.command, sub_cmd)
    fn = handlers.get(key)
    if fn is None:
        parser.print_help()
        return 1
    return int(fn(paths, args))


if __name__ == "__main__":
    raise SystemExit(main())
