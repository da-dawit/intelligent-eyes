"""Belief over species from correlated votes (proposal eq. 1).

The vote of observation j is l_j(y) = log q(y | f_j) - log pi(y). It says how much more
likely species y is after seeing j than before (pi = training class frequencies).
With correlated votes, Bayes' rule gives (derivations, Prop. 3.1)

    p(y)  propto  pi(y) * exp( sum_{j in S} w_j l_j(y) ),     w = R_S^{-1} 1
"""
from __future__ import annotations

import numpy as np

from .correlation import evidence_weights


def softmax(s: np.ndarray, axis: int = -1) -> np.ndarray:
    z = s - s.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats."""
    return -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0.0), axis=axis)


def votes_from_log_probs(log_q: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """l(y) = log q(y | f) - log pi(y)."""
    return log_q - log_prior


def scores(votes_S: np.ndarray, R_S: np.ndarray, log_prior: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Belief scores s(y) = log pi(y) + sum_j w_j l_j(y), with w = R_S^{-1} 1.

    votes_S is (|S|, K), the votes of the observations taken so far. R_S is (|S|, |S|).
    """
    w = evidence_weights(R_S, ridge)
    return log_prior + w @ votes_S


def posterior(votes_S: np.ndarray, R_S: np.ndarray, log_prior: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Posterior p = softmax(scores) of eq. (1)."""
    return softmax(scores(votes_S, R_S, log_prior, ridge))


def add_one_look_terms(R_SS: np.ndarray, R_SC: np.ndarray, ridge: float = 1e-6, R_CC_diag: np.ndarray | None = None):
    """Terms for adding one candidate look, from the block inverse (derivations, Prop. 4.1).

    For each candidate j (a column of R_SC = R[S, candidates]):
        a_j = R_SS^{-1} R_Sj                  what S already says about j
        s_j = 1 - R_jS R_SS^{-1} R_Sj         Schur complement, the part of j that is new
        c_j = (1 - R_jS R_SS^{-1} 1) / s_j    turns j's innovation into evidence
    Adding j changes the total evidence by c_j * (l_j - a_j^T l_S).

    Returns a (|S|, C), s (C,) and c (C,), all from one linear solve.
    R_CC_diag is each candidate's own variance R_jj. The default 1 is for a correlation
    matrix. With a shared crop error it is 1 + tau.
    """
    n = R_SS.shape[0]
    a = np.linalg.solve(R_SS + ridge * np.eye(n), R_SC)
    own = 1.0 if R_CC_diag is None else R_CC_diag
    s = np.clip(own - np.sum(R_SC * a, axis=0), 1e-9, None)
    c = (1.0 - a.sum(axis=0)) / s
    return a, s, c
