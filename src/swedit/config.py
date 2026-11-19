"""Config loading. One YAML file fully determines a run (plus the code version)."""
from __future__ import annotations
import copy, hashlib, json
import yaml

DEFAULTS = {
    "seed": 1234,
    "out_dir": "results/run1",
    "model": {
        "name": "AI-Sweden-Models/gpt-sw3-126m",  # or "tiny-random" for plumbing tests
        "revision": None,        # PIN a commit hash here for the real run
        "device": "cpu",
        "threads": None,
        "batch_size": 32,
    },
    "data": {
        "source": "builtin",     # "builtin" or "ud"
        "ud_paths": [],          # CoNLL-U files when source == "ud"
        "ud_min_count": 3,
        "n_dev": 60,             # items per type in the dev (selection) split
        "n_test": 100,           # items per type in the test split
        "ppl_text_path": None,   # optional text file, one sentence per line (even lines DEV, odd lines TEST)
        "english_control": False,  # only for models that know English
    },
    "edit": {
        "target_type": "long_distance",
        "k": 5,
        "alpha_up": 1.5,
        "top_m": 30,             # candidates verified in stage 2 of selection
        "penalty_lambda": 1.0,   # weight of the Swedish-fluency regression penalty
        "n_random": 20,
        "random_mode": "uniform",  # "uniform" or "layer_matched"
    },
    "prompting": {"n_shots": 8, "shot_types": ["lexical", "local", "long_distance"], "also_target_only": True},
    "finetune": {
        "enabled": True, "epochs": 10, "lr": 2e-4, "rank": 8, "alpha": 16,
        "batch_size": 8, "seeds": [0, 1, 2], "max_steps": None,
    },
    "stats": {"n_boot": 2000},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    user = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    return _merge(_merge(DEFAULTS, user), overrides or {})


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
