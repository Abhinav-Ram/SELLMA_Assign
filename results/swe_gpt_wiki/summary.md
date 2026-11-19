# Results summary (test split)

Target type: `long_distance`. Accuracy is *balanced* (mean of en-gold and ett-gold items).

Baseline per-token NLL on held-out ordinary Swedish text: 6.600 nats (dNLL below is change vs this).

| arm | lexical | local | long_distance | dNLL (Swedish text) |
|---|---|---|---|---|
| baseline | 0.980 (+0.000) | 0.980 (+0.000) | 0.670 (+0.000) | +0.0000 |
| targeted | 0.960 (-0.020) | 0.990 (+0.010) | 0.750 (+0.080) | -0.1928 |
| prompt | 0.970 (-0.010) | 0.970 (-0.010) | 0.680 (+0.010) | +0.0000 |
| prompt_target | 0.360 (-0.620) | 0.580 (-0.400) | 0.620 (-0.050) | +0.0000 |
| random edits (n=20) | 0.955 (p95 0.990) | 0.960 (p95 0.981) | 0.674 (p95 0.711) | +0.4702 (p95 +1.3213) |
| finetune (mean [min,max]) | 0.973 [0.970,0.980] | 0.987 [0.980,0.990] | 0.803 [0.790,0.820] | -0.7570 |

Per-group accuracy (Swedish: com/neut; English control if enabled: pl/sg), to expose a model that simply prefers one form:
- baseline: lexical 0.98/0.98; local 1.00/0.96; long_distance 0.80/0.54
- targeted: lexical 0.94/0.98; local 1.00/0.98; long_distance 0.84/0.66
- prompt: lexical 1.00/0.94; local 1.00/0.94; long_distance 0.84/0.52
- prompt_target: lexical 0.38/0.34; local 0.64/0.52; long_distance 0.70/0.54

Targeted vs random edits (one-sided empirical p; lower = targeted better than chance placement):
- lexical: p = 0.857
- local: p = 0.095
- long_distance: p = 0.048

Items per type (target): 100. Rough minimum detectable accuracy difference: ~0.20 (conservative; use the CIs in summary.json). A null result smaller than this is NOT evidence of no effect.

Edit: 5 heads, 245760 weights changed.
