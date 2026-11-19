"""Evaluate one model state (an 'arm') on a split. Returns per-item margins and the fluency check."""
from __future__ import annotations
import numpy as np
from .scoring import pair_margins, mean_nll

TYPES = ["lexical", "local", "long_distance", "en_sv_agree"]


def evaluate(model, tok, items_split: dict, ppl_sentences: list[str], prefix_ids=None, bs: int = 32) -> dict:
    out = {"margins": {}, "genders": {}, "ids": {}}
    for t in items_split:
        its = items_split[t]
        out["margins"][t] = pair_margins(model, tok, its, prefix_ids, bs).tolist()
        out["genders"][t] = [i.meta["gold_gender"] for i in its]
        out["ids"][t] = [i.id for i in its]
    # fluency regression: per-token NLL on held-out Swedish sentences (prefix applies only to pairs above)
    out["nll"] = mean_nll(model, tok, ppl_sentences, bs)
    return out


def correct(arm_eval: dict, t: str) -> np.ndarray:
    return np.asarray(arm_eval["margins"][t]) > 0
