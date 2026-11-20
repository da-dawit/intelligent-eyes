"""Choosing the few scalars that are not derived: shrinkage lam, stopping threshold eps.

Both are chosen on held-out clips only (proposal Sec. 7: "10% held out for eps").
"""
from __future__ import annotations

import numpy as np

from .belief import posterior
from .controller import Clip, Model, glance_indices, look_indices, run
from .correlation import shrink


def fused_nll(clips: list[Clip], model: Model, n_random: int = 2, seed: int = 0) -> float:
    """Mean negative log-likelihood of the true species under eq. (1), averaged over two
    observation sets per clip: glances + `n_random` random looks, and glances + every look.

    The first is the regime the eyes work in; the second stresses the within-frame table A.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for c in clips:
        R = model.R(c.T)
        K = c.votes.shape[-1]
        votes = c.votes.reshape(-1, K)
        G, Ls = glance_indices(c.T), look_indices(c.T)
        some = list(rng.choice(Ls, size=min(n_random, len(Ls)), replace=False))
        for S in (G + some, G + Ls):
            p = posterior(votes[S], R[np.ix_(S, S)], model.log_prior, model.ridge)
            total += -np.log(p[c.label] + 1e-12) / 2
    return total / len(clips)


def choose_shrinkage(clips: list[Clip], A: np.ndarray, rho: float, log_prior: np.ndarray, sigma2: float,
                     lambdas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0), seed: int = 0):
    """Pick lam for A_lam = (1 - lam) A + lam I by held-out NLL. Returns (lam, {lam: nll})."""
    nll = {lam: fused_nll(clips, Model(shrink(A, lam), rho, log_prior, sigma2), seed=seed) for lam in lambdas}
    return min(nll, key=nll.get), nll


def choose_eps(clips: list[Clip], model: Model, budget: int, grid=None, tol: float = 0.005, seed: int = 0,
               rule: str = "eig"):
    """Largest eps whose held-out accuracy is within `tol` of never stopping early.

    Returns (eps, table) with table[eps] = (accuracy, mean passes).
    """
    grid = grid if grid is not None else [0.0, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.2]
    labels = np.array([c.label for c in clips])
    table = {}
    for eps in grid:
        rng = np.random.default_rng(seed)
        out = [run(c, model, rule, budget=budget, eps=eps, rng=rng) for c in clips]
        table[eps] = (float(np.mean([o.pred for o in out] == labels)), float(np.mean([o.passes for o in out])))
    base = table[grid[0]][0]
    ok = [e for e in grid if table[e][0] >= base - tol]
    return max(ok), table
