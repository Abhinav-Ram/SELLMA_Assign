"""Seeding and run manifest (everything needed to reproduce a run)."""
from __future__ import annotations
import json, os, platform, random, subprocess, sys
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU matmuls are deterministic enough; keep simple
    except ImportError:
        pass


def sub_seed(base: int, *parts) -> int:
    """Deterministic derived seed, e.g. sub_seed(cfg_seed, 'random_edit', i)."""
    h = base
    for p in parts:
        for ch in str(p):
            h = (h * 1000003 + ord(ch)) % (2**31 - 1)
    return h


def manifest(cfg: dict, model_info: dict | None = None) -> dict:
    import torch, transformers
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = None
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "git_sha": sha,
        "threads": torch.get_num_threads(),
        "seed": cfg["seed"],
    }
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    if model_info:
        info["model"] = model_info
    return info


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
