"""Statistics used in the evaluation (derivations Sec. 10, proposal Table I)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, spearmanr


def accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(pred == label))


def ece(probs: np.ndarray, label: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins.

    ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)|
    """
    return ece_from_confidence(probs.max(axis=1), probs.argmax(axis=1) == label, n_bins)


def ece_from_confidence(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE from the confidence of each prediction and whether it was right."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        idx = bins == b
        if idx.any():
            total += idx.mean() * abs(correct[idx].mean() - conf[idx].mean())
    return float(total)


def confident_wrong_rate(probs: np.ndarray, label: np.ndarray, threshold: float = 0.9) -> float:
    """Share of clips predicted with confidence >= threshold that are wrong (failure mode 2)."""
    conf = probs.max(axis=1)
    sure = conf >= threshold
    if not sure.any():
        return 0.0
    return float(np.mean(probs[sure].argmax(axis=1) != label[sure]))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact paired McNemar test on the same clips.

    b counts clips right only under A and c clips right only under B. The test is a
    two-sided binomial test of b against Bin(b + c, 1/2).
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return dict(b=b, c=c, p=float(p))


def camera_bootstrap(values: np.ndarray, cameras: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-clip `values` with a bootstrap interval that resamples cameras.

    Clips from one camera are correlated, so the resampled unit is the camera instead of
    the clip. Returns (mean, low, high).
    """
    rng = np.random.default_rng(seed)
    cams, inv = np.unique(cameras, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(cams))
    counts = np.bincount(inv, minlength=len(cams)).astype(float)
    draws = rng.integers(0, len(cams), size=(n_boot, len(cams)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def macro_f1(pred: np.ndarray, label: np.ndarray) -> float:
    """Macro-F1 over the classes present in the labels (iWildCam's official metric)."""
    f1s = []
    for c in np.unique(label):
        tp = np.sum((pred == c) & (label == c))
        fp = np.sum((pred == c) & (label != c))
        fn = np.sum((pred != c) & (label == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def quadratic_kappa(pred: np.ndarray, label: np.ndarray, n_classes: int) -> float:
    """Quadratic weighted kappa (DDR's official metric for grades 0-4)."""
    O = np.zeros((n_classes, n_classes))
    np.add.at(O, (label, pred), 1)
    i, j = np.meshgrid(np.arange(n_classes), np.arange(n_classes), indexing="ij")
    Wt = (i - j) ** 2 / (n_classes - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (Wt * O).sum() / (Wt * E).sum())
