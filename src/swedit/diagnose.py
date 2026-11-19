"""Diagnostics (uses DEV items only): which heads does the target skill depend on?

Stage 1  zero-ablate each head, record the change in mean margin on the target items (and on the
         Swedish regression sentences). This is *model feedback* (forward passes), not weight reading.
Stage 2  for the top-M heads by |effect|, propose an op (ablate if ablation helped, amplify by
         alpha_up if ablation hurt), apply that op alone, and measure its real dev effect.
Selection greedy top-k by  gain_target - lambda * harm_swedish_fluency  (only positive scores kept).
"""
from __future__ import annotations
import numpy as np
from .edits import EditSpec, HeadOp, applied
from .scoring import pair_margins, mean_sentence_logprob


def ablation_sweep(model, tok, arch, target_items, reg_sents, bs=32, log=print) -> list[dict]:
    base_t = pair_margins(model, tok, target_items, batch_size=bs).mean()
    base_e = mean_sentence_logprob(model, tok, reg_sents, bs) if reg_sents else 0.0
    rows = []
    for l in range(arch.n_layers):
        for h in range(arch.n_heads):
            with applied(model, arch, EditSpec([HeadOp(l, h, 0.0)])):
                dt = pair_margins(model, tok, target_items, batch_size=bs).mean() - base_t
                de = (mean_sentence_logprob(model, tok, reg_sents, bs) - base_e) if reg_sents else 0.0
            rows.append({"layer": l, "head": h, "d_target": float(dt), "d_reg": float(de)})
        log(f"  ablation sweep: layer {l + 1}/{arch.n_layers} done")
    return rows


def select_spec(model, tok, arch, rows, target_items, reg_sents, k, alpha_up, top_m, lam,
                bs=32, log=print) -> tuple[EditSpec, list[dict]]:
    base_t = pair_margins(model, tok, target_items, batch_size=bs).mean()
    base_e = mean_sentence_logprob(model, tok, reg_sents, bs) if reg_sents else 0.0
    cands = sorted(rows, key=lambda r: -abs(r["d_target"]))[:top_m]
    verified = []
    for r in cands:
        alpha = 0.0 if r["d_target"] > 0 else alpha_up
        op = HeadOp(r["layer"], r["head"], alpha)
        with applied(model, arch, EditSpec([op])):
            gt = pair_margins(model, tok, target_items, batch_size=bs).mean() - base_t
            ge = (mean_sentence_logprob(model, tok, reg_sents, bs) - base_e) if reg_sents else 0.0
        score = float(gt - lam * max(0.0, -ge))
        verified.append({**r, "alpha": alpha, "gain_target": float(gt), "gain_reg": float(ge), "score": score})
    verified.sort(key=lambda v: -v["score"])
    chosen = [v for v in verified if v["score"] > 0][:k]
    spec = EditSpec([HeadOp(v["layer"], v["head"], v["alpha"]) for v in chosen])
    if len(spec):
        with applied(model, arch, spec):
            joint = float(pair_margins(model, tok, target_items, batch_size=bs).mean() - base_t)
    else:
        joint = 0.0
    log(f"  selected {len(spec)} heads; joint dev margin gain {joint:+.3f}")
    return spec, [{**v, "selected": v in chosen, "joint_dev_gain": joint} for v in verified]
