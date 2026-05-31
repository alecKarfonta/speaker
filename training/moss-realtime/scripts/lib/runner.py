"""Subprocess helpers for distill CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lib.paths import Paths


def run_cmd(
    paths: Paths,
    cmd: list[str],
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = paths.env()
    if extra_env:
        env.update(extra_env)
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or paths.repo_root, env=env).returncode


def run_python(paths: Paths, script: Path, args: list[str], *, extra_env: dict[str, str] | None = None) -> int:
    py = os.environ.get("MOSS_DISTILL_PYTHON", sys.executable)
    return run_cmd(paths, [py, str(script), *args], extra_env=extra_env)


def run_shell(paths: Paths, script: Path, args: list[str] | None = None, *, extra_env: dict[str, str] | None = None) -> int:
    cmd = [str(script), *(args or [])]
    return run_cmd(paths, cmd, extra_env=extra_env)
