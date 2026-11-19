"""Turn per-arm evaluations into the summary table / plot."""
from __future__ import annotations
import os
import numpy as np
from . import stats as S
from .evaluate import correct


def _ba(arm, items_split, t):
    c = correct(arm, t)
    return S.balanced_accuracy(c, arm["genders"][t]), c


def summarise(arms: dict, items_test: dict, target_type: str, n_boot: int, seed: int) -> dict:
    base = arms["baseline"]
    TYPES = list(base["margins"].keys())
    res = {"target_type": target_type, "types": TYPES, "arms": {}, "random": {}, "finetune": {}}

    def row(name):
        r = {}
        for t in TYPES:
            ba, c = _ba(arms[name], items_test, t)
            lo, hi = S.bootstrap_ci(c, arms[name]["genders"][t], n_boot, seed)
            bc = correct(base, t)
            d, dlo, dhi = S.paired_diff_ci(c, bc, arms[name]["genders"][t], n_boot, seed)
            g = np.array(arms[name]["genders"][t])
            r[t] = {"bal_acc": ba, "ci": [lo, hi],
                    "acc_by_group": {str(v): float(c[g == v].mean()) for v in np.unique(g)}, "delta_vs_base": d, "delta_ci": [dlo, dhi],
                    "mcnemar_p": S.mcnemar_exact_p(c, bc)}
        r["nll"] = arms[name]["nll"]; r["d_nll"] = arms[name]["nll"] - base["nll"]
        return r

    for name in ["baseline", "targeted"] + sorted(n for n in arms if n.startswith("prompt")):
        res["arms"][name] = row(name)
    res["baseline_nll"] = base["nll"]
    rnames = sorted(n for n in arms if n.startswith("random_"))
    if rnames:
        for t in TYPES:
            vals = [_ba(arms[n], items_test, t)[0] for n in rnames]
            tgt = res["arms"]["targeted"][t]["bal_acc"]
            res["random"][t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                "p95": float(np.quantile(vals, 0.95)), "n": len(vals), "values": vals,
                                "targeted_vs_random_p": S.empirical_p(tgt, vals)}
        dn = [arms[n]["nll"] - base["nll"] for n in rnames]
        res["random"]["d_nll_mean"] = float(np.mean(dn))
        res["random"]["d_nll_p95"] = float(np.quantile(dn, 0.95))
    fnames = sorted(n for n in arms if n.startswith("finetune_"))
    if fnames:
        rows = [row(n) for n in fnames]
        for t in TYPES:
            v = [r[t]["bal_acc"] for r in rows]
            res["finetune"][t] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
        res["finetune"]["d_nll_mean"] = float(np.mean([r["d_nll"] for r in rows]))
    return res


def write_report(s: dict, out_dir: str) -> None:
    L = ["# Results summary (test split)\n",
         f"Target type: `{s['target_type']}`. Accuracy is *balanced* (mean of en-gold and ett-gold items).\n",
         f"Baseline per-token NLL on held-out ordinary Swedish text: {s['baseline_nll']:.3f} nats (dNLL below is change vs this).\n",
         "| arm | " + " | ".join(s["types"]) + " | dNLL (Swedish text) |", "|---|" + "---|" * (len(s["types"]) + 1)]
    for name, r in s["arms"].items():
        cells = [f"{r[t]['bal_acc']:.3f} ({r[t]['delta_vs_base']:+.3f})" for t in s["types"]]
        L.append(f"| {name} | " + " | ".join(cells) + f" | {r['d_nll']:+.4f} |")
    if s["random"]:
        cells = [f"{s['random'][t]['mean']:.3f} (p95 {s['random'][t]['p95']:.3f})" for t in s["types"]]
        L.append(f"| random edits (n={s['random'][s['types'][0]]['n']}) | " + " | ".join(cells) + f" | {s['random']['d_nll_mean']:+.4f} (p95 {s['random']['d_nll_p95']:+.4f}) |")
    if s["finetune"]:
        cells = [f"{s['finetune'][t]['mean']:.3f} [{s['finetune'][t]['min']:.3f},{s['finetune'][t]['max']:.3f}]" for t in s["types"]]
        L.append(f"| finetune (mean [min,max]) | " + " | ".join(cells) + f" | {s['finetune']['d_nll_mean']:+.4f} |")
    L += ["", "Per-group accuracy (Swedish: com/neut; English control if enabled: pl/sg), to expose a model that simply prefers one form:"]
    for name, r in s["arms"].items():
        L.append(f"- {name}: " + "; ".join(f"{t} " + "/".join(f"{v:.2f}" for _, v in sorted(r[t]["acc_by_group"].items())) for t in s["types"]))
    L += ["", "Targeted vs random edits (one-sided empirical p; lower = targeted better than chance placement):"]
    for t in s["types"]:
        if s["random"]:
            L.append(f"- {t}: p = {s['random'][t]['targeted_vs_random_p']:.3f}")
    L += ["", f"Items per type (target): {s['n_test_target']}. Rough minimum detectable accuracy difference: ~{s['mde_balanced_acc']:.2f} "
              "(conservative; use the CIs in summary.json). A null result smaller than this is NOT evidence of no effect.",
          "", f"Edit: {s['edit']['n_heads']} heads, {s['edit']['n_weights_changed']} weights changed.", ""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = s["target_type"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(s["random"][t]["values"], bins=10, color="#bbb", label="random edits")
        ax.axvline(s["arms"]["targeted"][t]["bal_acc"], color="C3", label="targeted")
        ax.axvline(s["arms"]["baseline"][t]["bal_acc"], color="k", ls="--", label="baseline")
        ax.set_xlabel(f"balanced accuracy ({t}, test)"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "random_vs_targeted.png"), dpi=150)
    except Exception:
        pass
