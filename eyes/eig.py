"""Expected information gain of a look (proposal eq. 3, derivations Sec. 6).

If look j would shift the species scores by Delta ~ N(m_j, v_j I), then

    EIG(j) = E[ KL(p_new || p) ]  ~=  1/2 ( m_j^T F m_j + v_j tr F ),   F = diag(p) - p p^T

    m_j^T F m_j = Var_{y~p}[ m_j(y) ]    expected signal (Lemma 6.4)
    v_j tr F    = v_j (1 - sum_y p_y^2)  surprise

Both terms vanish as p becomes one-hot, which is the stopping rule (Cor. 7.1).
"""
from __future__ import annotations

import numpy as np

from .belief import entropy, softmax


def fisher(p: np.ndarray) -> np.ndarray:
    """F = diag(p) - p p^T, the Hessian of log-sum-exp (Lemma 6.2)."""
    return np.diag(p) - np.outer(p, p)


def var_under_p(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Delta^T F Delta = Var_{y~p}[Delta(y)], for each row Delta of M (Lemma 6.4).

    Costs O(K) per row because it never forms the K x K matrix F.
    """
    mean = M @ p
    return (M ** 2) @ p - mean ** 2


def trace_fisher(p: np.ndarray) -> float:
    """tr F = 1 - sum_y p_y^2 (the Gini impurity of the belief)."""
    return float(1.0 - np.sum(p ** 2))


def eig_closed_form(p: np.ndarray, m: np.ndarray, v: np.ndarray, cap: bool = True) -> np.ndarray:
    """Closed-form EIG of eq. (3) for C candidates (Theorem 6.5).

    p (K,) is the current belief, m (C, K) the expected score shifts and v (C,) their variances.
    With cap=True the result is clipped at H(p). Mutual information can never be larger than
    the current entropy, but the quadratic form can be (derivations, 'Limits').
    """
    eig = 0.5 * (var_under_p(p, m) + v * trace_fisher(p))
    if cap:
        eig = np.minimum(eig, entropy(p))
    return eig


def kl(p_new: np.ndarray, p: np.ndarray) -> np.ndarray:
    """KL(p_new || p) along the last axis, in nats."""
    ratio = np.log(np.clip(p_new, 1e-300, None)) - np.log(np.clip(p, 1e-300, None))
    return np.sum(p_new * ratio, axis=-1)


def eig_monte_carlo(s: np.ndarray, m: np.ndarray, v: np.ndarray, n_samples: int = 64,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """EIG with the exact KL, by Monte Carlo (derivations, 'Limits', (ii)).

    s (K,) is the current scores (p = softmax(s)), m is (C, K) and v is (C,).
    Draws Delta^(i) ~ N(m_j, v_j I) and averages KL(softmax(s + Delta) || p).
    Returns the mean and its standard error, each (C,).
    """
    rng = rng or np.random.default_rng(0)
    p = softmax(s)
    C, K = m.shape
    noise = rng.standard_normal((n_samples, C, K)) * np.sqrt(v)[None, :, None]
    delta = m[None] + noise                                  # (M, C, K)
    p_new = softmax(s[None, None, :] + delta)                # (M, C, K)
    kls = kl(p_new, p[None, None, :])                        # (M, C)
    return kls.mean(axis=0), kls.std(axis=0, ddof=1) / np.sqrt(n_samples)
