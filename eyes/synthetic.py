"""Synthetic clips with a known R, to test the controller before real data exists.

Each clip has one species y (A1). An animal sits under one crop and may move between
frames. Observation (f, k) sees a feature

    x_fk = a_fk * onehot(y) + n_fk,     n ~ N(0, R) per species component, R = T_rho (x) A

where a_fk is how visible the animal is from position k (glance: small; the crop that
holds the animal: full; half-overlapping neighbours: a quarter; others: 0).

The vote is the calibrated log-likelihood ratio of that feature on its own,
    l_fk(y) = log N(x_fk; a_fk e_y, I) + const = a_fk * x_fk(y) + const
            = a_fk^2 * onehot(y) + a_fk * n_fk      (centred across species)
so an empty crop votes exactly 0, as a calibrated classifier would (A3).

The look predictor knows roughly *where* the animal is (a noisy preview of a_fk) but
not *what* it is, so it predicts mu_fk = a_hat_fk^2 * (p_f - 1/K): the belief after
frame f's glance, made louder where the preview sees something.
"""
from __future__ import annotations

import numpy as np

from .belief import softmax
from .controller import Clip
from .correlation import clip_correlation
from .geometry import N_POS, overlap_correlation


def make_clips(n_clips: int, K: int = 8, T_max: int = 10, rho: float = 0.8, A: np.ndarray | None = None,
               amp: float = 2.0, glance_amp: float = 0.6, move_prob: float = 0.3,
               preview_noise: float = 0.3, seed: int = 0):
    """Return (clips, log_prior). Species frequencies follow a mild Zipf law."""
    rng = np.random.default_rng(seed)
    A = overlap_correlation() if A is None else A
    pi = 1.0 / np.arange(1, K + 1) ** 0.7
    pi /= pi.sum()
    log_pi = np.log(pi)
    neighbour = overlap_correlation()[1:, 1:] ** 2          # 1 on the animal's crop, 0.25 beside it

    clips = []
    for i in range(n_clips):
        y = rng.choice(K, p=pi)
        T = int(rng.integers(1, T_max + 1))
        pos = int(rng.integers(0, 9))
        a = np.zeros((T, N_POS))
        for f in range(T):
            if f > 0 and rng.random() < move_prob:
                pos = int(rng.integers(0, 9))
            a[f, 0] = glance_amp
            a[f, 1:] = amp * neighbour[pos]
        R = clip_correlation(T, A, rho)
        Lc = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
        n = (Lc @ rng.standard_normal((T * N_POS, K))).reshape(T, N_POS, K)
        votes = a[..., None] ** 2 * np.eye(K)[y] + a[..., None] * n
        votes -= votes.mean(axis=2, keepdims=True)

        a_hat = np.clip(a + preview_noise * amp * rng.standard_normal(a.shape), 0, None)
        p_glance = softmax(log_pi + votes[:, 0])               # (T, K)
        mu = a_hat[..., None] ** 2 * (p_glance - 1.0 / K)[:, None, :]
        mu[:, 0] = votes[:, 0]
        clips.append(Clip(votes=votes, mu=mu, label=int(y), clip_id=f"syn{i}", camera=i % 50))
    return clips, log_pi


def predictor_sigma2(clips) -> float:
    """Centred mean squared error of the synthetic predictor over all looks."""
    r = np.concatenate([(c.votes[:, 1:] - c.mu[:, 1:]).reshape(-1, c.votes.shape[-1]) for c in clips])
    r = r - r.mean(axis=1, keepdims=True)
    return float(np.sum(r ** 2) / (r.shape[0] * (r.shape[1] - 1)))
