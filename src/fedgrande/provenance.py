"""Run metadata attached to every Phase 2 result: commits, environment, versions, platform."""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys

from .paths import GRANDE_WORKTREES, ROOT

KEY_PACKAGES = ["GRANDE", "torch", "numpy", "pandas", "scipy", "scikit-learn", "category-encoders", "openml"]


def _git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_metadata(impl: str | None = None, **extra) -> dict:
    """Commit of the project and of the GRANDE worktree in use, environment, package versions, platform."""
    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_commit": _git(ROOT, "rev-parse", "HEAD"),
        "project_tracked_changes": _git(ROOT, "status", "--porcelain", "--untracked-files=no").splitlines(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {n: _version(n) for n in KEY_PACKAGES},
    }
    if impl is not None:
        wt, expected = GRANDE_WORKTREES[impl]
        head = _git(wt, "rev-parse", "HEAD")
        meta["grande"] = {
            "impl": impl,
            "worktree": str(wt),
            "expected_commit": expected,
            "head_commit": head,
            "commit_matches": head == expected,
            "tracked_changes": _git(wt, "status", "--porcelain", "--untracked-files=no").splitlines(),
        }
    try:
        import torch

        meta["torch_threads"] = torch.get_num_threads()
        meta["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:  # noqa: BLE001
        pass
    meta.update(extra)
    return meta


def write_json(path, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def _jsonable(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
