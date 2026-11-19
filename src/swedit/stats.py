"""Statistics: balanced accuracy, bootstrap CIs, paired differences, random-edit comparison."""
from __future__ import annotations
import math
import numpy as np


def balanced_accuracy(correct: np.ndarray, genders: list[str]) -> float:
    g = np.array(genders)
    accs = [correct[g == v].mean() for v in np.unique(g) if (g == v).any()]
    return float(np.mean(accs))


def bootstrap_ci(correct, genders, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(correct); g = np.array(genders); c = np.asarray(correct, dtype=float)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(balanced_accuracy(c[i], list(g[i])))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff_ci(correct_a, correct_b, genders, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap CI for balanced_acc(a) - balanced_acc(b) over the same items (resampled jointly)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(correct_a, dtype=float); b = np.asarray(correct_b, dtype=float)
    g = np.array(genders); n = len(a)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(balanced_accuracy(a[i], list(g[i])) - balanced_accuracy(b[i], list(g[i])))
    point = balanced_accuracy(a, list(g)) - balanced_accuracy(b, list(g))
    return float(point), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def mcnemar_exact_p(correct_a, correct_b) -> float:
    a = np.asarray(correct_a, bool); b = np.asarray(correct_b, bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum()); n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def empirical_p(targeted: float, random_vals: list[float]) -> float:
    """One-sided: fraction of random edits at least as good as the targeted edit (with +1 smoothing)."""
    r = np.asarray(random_vals)
    return float((1 + (r >= targeted).sum()) / (1 + len(r)))


def mde(n: int, p: float = 0.5, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Rough, conservative minimum detectable accuracy difference for two independent samples of size n.
    (Paired designs can detect somewhat smaller effects; this is a planning heuristic.)"""
    return float((z_alpha + z_power) * math.sqrt(2 * p * (1 - p) / n))
