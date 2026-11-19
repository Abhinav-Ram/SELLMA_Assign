"""Few-shot prompting arm: prepend correct DEV sentences as context; no weights change.
The caller chooses which DEV item types supply the shots (mixed Swedish types, or the target type only)."""
from __future__ import annotations
import random


def build_prefix_ids(tok, dev_items, n_shots: int, seed: int) -> tuple[list[int], list[str]]:
    rng = random.Random(seed)
    by_g = {}
    for it in dev_items:
        by_g.setdefault(it.meta["gold_gender"], []).append(it)
    shots = []
    per = max(1, n_shots // 2)
    for g in sorted(by_g):
        pool = by_g[g][:]
        rng.shuffle(pool)
        shots += pool[:per]
    rng.shuffle(shots)
    sents = [s.gold for s in shots]
    ids = []
    for s in sents:
        ids += tok.encode(s + "\n")
    return ids, sents
