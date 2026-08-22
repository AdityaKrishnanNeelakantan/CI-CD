"""safetensors used ONLY for real tensors (unused by the statistical backend)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def save_tensors(tensors: dict, path: str | Path) -> Path:
    save_file({k: np.asarray(v) for k, v in tensors.items()}, str(path))
    return Path(path)


def load_tensors(path: str | Path) -> dict:
    return load_file(str(path))
