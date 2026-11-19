"""Collect per-seed summary.json files into one table: python -m swedit.aggregate results/swe_gpt_wiki/seed_*"""
from __future__ import annotations
import json, os, sys
import numpy as np


def main(dirs):
    rows = []
    for d in dirs:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            print("missing", p); continue
        s = json.load(open(p)); t = s["target_type"]
        base = s["arms"]["baseline"][t]["bal_acc"]
        r = {"dir": os.path.basename(d.rstrip("/")), "n": s["n_test_target"], "base": base,
             "targeted": s["arms"]["targeted"][t]["bal_acc"], "rand_mean": s["random"][t]["mean"],
             "rand_p95": s["random"][t]["p95"], "p": s["random"][t]["targeted_vs_random_p"],
             "prompt": s["arms"]["prompt"][t]["bal_acc"], "finetune": s["finetune"][t]["mean"] if s["finetune"] else float("nan"),
             "heads": ",".join(f"{o['layer']}.{o['head']}" for o in s["edit"]["ops"]),
             "lex_t": s["arms"]["targeted"]["lexical"]["bal_acc"], "dnll_t": s["arms"]["targeted"]["d_nll"]}
        rows.append(r)
    if not rows:
        return
    hdr = "run | n | base | targeted | rand mean (p95) | p | prompt | finetune | lexical(targ) | dNLL(targ) | heads"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dir']} | {r['n']} | {r['base']:.3f} | {r['targeted']:.3f} | {r['rand_mean']:.3f} ({r['rand_p95']:.3f}) | "
              f"{r['p']:.3f} | {r['prompt']:.3f} | {r['finetune']:.3f} | {r['lex_t']:.3f} | {r['dnll_t']:+.3f} | {r['heads']}")
    g = np.array([r["targeted"] - r["base"] for r in rows])
    print(f"\ntargeted gain over baseline across {len(rows)} runs: mean {g.mean():+.3f}, min {g.min():+.3f}, max {g.max():+.3f}")
    allheads = [set(r["heads"].split(",")) for r in rows]
    common = set.intersection(*allheads)
    print("heads chosen in every run:", sorted(common) if common else "none")


if __name__ == "__main__":
    main(sys.argv[1:])
