#!/usr/bin/env python3
"""
Patch torchaudio's _torchcodec.py so that load_with_torchcodec() falls back
to soundfile-based loading when the torchcodec C++ library is not available.

This fix is needed because:
- torchaudio >= 2.9.1 defaults to torchcodec for torchaudio.load()
- The vLLM base image ships torchcodec but the .so files have an undefined
  symbol (ABI mismatch: libtorchcodec_core*.so)
- torchcodec was uninstalled (pip uninstall) but torchaudio still tries to
  import it and crashes with ImportError in the vLLM EngineCore subprocess
- Setting TORCHAUDIO_BACKEND=soundfile in env is not enough because the
  EngineCore subprocess is spawned fresh via Python 'spawn' and torchaudio
  does its backend selection at import time before any user code runs

The patch replaces the hard ImportError with a soundfile fallback.
"""

import re
from pathlib import Path

TORCHCODEC_PATH = Path("/usr/local/lib/python3.12/dist-packages/torchaudio/_torchcodec.py")

FALLBACK_CODE = '''
def load_with_torchcodec(
    uri,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format=None,
    buffer_size: int = 4096,
    backend=None,
):
    """
    Patched: falls back to soundfile when torchcodec C library is unavailable.
    This patch was applied by scripts/patch_torchaudio.py to avoid crashes
    caused by a broken torchcodec .so after uninstallation.
    """
    import soundfile as _sf
    import numpy as _np
    import torch as _torch

    # Handle file-like objects and paths
    data, sample_rate = _sf.read(str(uri), dtype="float32", always_2d=True)
    # data shape: (frames, channels)
    waveform = _torch.from_numpy(data.T)  # -> (channels, frames)
    if not channels_first:
        waveform = waveform.T
    # Apply frame slicing
    if frame_offset > 0:
        waveform = waveform[..., frame_offset:]
    if num_frames > 0:
        waveform = waveform[..., :num_frames]
    return waveform, sample_rate


def save_with_torchcodec(
    uri,
    src,
    sample_rate: int,
    format=None,
    encoding=None,
    bits_per_sample=None,
    buffer_size: int = 4096,
    backend=None,
):
    """
    Patched: falls back to soundfile when torchcodec C library is unavailable.
    """
    import soundfile as _sf
    import numpy as _np

    data = src.numpy()
    if data.ndim == 2:
        data = data.T  # soundfile expects (frames, channels)
    _sf.write(str(uri), data, sample_rate)
'''


def patch():
    if not TORCHCODEC_PATH.exists():
        print(f"torchaudio _torchcodec.py not found at {TORCHCODEC_PATH}, skipping patch")
        return

    original = TORCHCODEC_PATH.read_text()

    if "Patched: falls back to soundfile" in original:
        print("torchaudio _torchcodec.py already patched, skipping")
        return

    # Write the patched version: keep the imports at the top, replace the functions
    patched = FALLBACK_CODE

    TORCHCODEC_PATH.write_text(patched)
    print(f"✅ Patched {TORCHCODEC_PATH}")


if __name__ == "__main__":
    patch()
