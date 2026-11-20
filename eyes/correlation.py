"""The repetition matrix R and the evidence weights w = R^{-1} 1 (proposal eq. 1).

Observation j = (f, k) is frame f seen at position k (see geometry.py). Two observations
repeat each other by

    R[(f, k), (f', k')] = A[k, k'] * rho ** |f - f'|          (derivations, Sec. 3.1)

i.e. R = T_rho (x) A in frame-major order, with A the 10x10 table of how much two
positions repeat each other inside one frame and T_rho the AR(1) correlation in time.
Observation (f, k) sits at row f * 10 + k.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import N_POS


# ----------------------------------------------------------------------------- structure

def frame_correlation(T: int, rho: float) -> np.ndarray:
    """T_rho: the T x T AR(1) correlation matrix, T_rho[f, f'] = rho ** |f - f'|."""
    f = np.arange(T)
    return rho ** np.abs(f[:, None] - f[None, :])


def clip_correlation(T: int, A: np.ndarray, rho: float) -> np.ndarray:
    """R = T_rho (x) A for a clip of T frames: a (10T x 10T) matrix."""
    return np.kron(frame_correlation(T, rho), A)


def obs_index(f: int, k: int) -> int:
    """Row of observation (frame f, position k) in R."""
    return f * N_POS + k


def nearest_correlation(A: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Make a symmetric matrix a valid correlation matrix.

    Clip negative eigenvalues to `floor`, then rescale so the diagonal is 1
    (derivations, Sec. 3.1). An estimated A can be slightly indefinite because each
    entry is estimated separately.
    """
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    A = (vecs * np.clip(vals, floor, None)) @ vecs.T
    d = np.sqrt(np.diag(A))
    return A / d[:, None] / d[None, :]


def shrink(A: np.ndarray, lam: float) -> np.ndarray:
    """A_lam = (1 - lam) A + lam I: trust the within-frame repetition table less.

    Why this is needed (found on synthetic clips before any real data): the common-mean
    model behind eq. (1) -- every vote is a noisy copy of one L -- holds for repeats in
    time (same crop, consecutive frames), but not for crops that see *different* parts of
    the frame. With the overlap table the four corner crops tile the frame, so A is
    singular and w = A^{-1} 1 gives the centre and edge crops weight 0. Shrinking A toward
    I makes R positive definite (condition number < 10 for lam >= 0.3 with the overlap
    table); the centre crop's weight turns positive at lam ~ 0.35 and every weight at
    lam ~ 0.65. A negative glance weight is GLS cancelling noise the glance shares with
    the crops, not an error. lam = 0 is the full model, lam = 1 treats positions in one
    frame as independent. Time (rho) is not shrunk. lam is chosen by held-out
    log-likelihood (fit.choose_shrinkage).
    """
    return (1.0 - lam) * A + lam * np.eye(A.shape[0])


# ----------------------------------------------------------------------------- eq. (1)

def evidence_weights(R_S: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """w = (R_S + ridge I)^{-1} 1: how much each observation counts (Prop. 3.1).

    The ridge keeps R_S invertible when two observations nearly repeat each other
    (failure mode 5); ridge -> 0 recovers the pseudo-inverse limit of Prop. 3.2.
    """
    n = R_S.shape[0]
    return np.linalg.solve(R_S + ridge * np.eye(n), np.ones(n))


def effective_count(R_S: np.ndarray, ridge: float = 1e-6) -> float:
    """n_eff = 1^T R_S^{-1} 1: how many independent views the observations are worth."""
    return float(evidence_weights(R_S, ridge).sum())


def ar1_effective_frames(T: int, rho: float) -> float:
    """Closed form of n_eff for T glances alone (Prop. 3.3): (T - (T-2) rho) / (1 + rho)."""
    if T == 1:
        return 1.0
    return (T - (T - 2) * rho) / (1 + rho)


# ----------------------------------------------------------------------------- fitting

def class_conditional_residuals(votes: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Residuals e = vote - E[vote | position, true species], centred across species.

    votes : (N, 10, K) votes of N frames at the 10 positions
    labels: (N,) true species index of each frame's clip

    Assumption (A2) says that *given the species* the votes are Gaussian with
    correlation R, so R is estimated from what is left after removing the
    species-and-position mean. Each residual vector is then centred across species,
    because adding the same amount to every species changes nothing (F 1 = 0, Lemma 6.4a).
    """
    resid = np.empty_like(votes, dtype=np.float64)
    for c in range(n_classes):
        idx = labels == c
        if idx.any():
            resid[idx] = votes[idx] - votes[idx].mean(axis=0, keepdims=True)
    return resid - resid.mean(axis=2, keepdims=True)


def fit_A(resid: np.ndarray) -> np.ndarray:
    """A[k, k'] = pooled correlation of residuals at positions k, k' in the same frame.

    resid: (N, 10, K). Pooled over frames and species components.
    """
    N, P, K = resid.shape
    flat = resid.transpose(1, 0, 2).reshape(P, N * K)       # one long vector per position
    cov = flat @ flat.T
    d = np.sqrt(np.diag(cov))
    return nearest_correlation(cov / d[:, None] / d[None, :])


def fit_rho(resid: np.ndarray, clip_ids: np.ndarray, frame_idx: np.ndarray, max_lag: int = 4):
    """Fit rho so that same-position residuals d frames apart correlate like rho ** d.

    resid    : (N, 10, K) residuals of N frames
    clip_ids : (N,) clip of each frame;  frame_idx: (N,) position of the frame in its clip
    Returns (rho_hat, c_hat) where c_hat[d-1] is the measured pooled correlation at lag d.
    rho_hat minimises sum_d n_d (c_hat_d - rho^d)^2, weighting each lag by its pair count.
    """
    order = np.lexsort((frame_idx, clip_ids))
    resid, clip_ids, frame_idx = resid[order], clip_ids[order], frame_idx[order]
    c_hat, n_pairs = [], []
    for d in range(1, max_lag + 1):
        same = (clip_ids[d:] == clip_ids[:-d]) & (frame_idx[d:] - frame_idx[:-d] == d)
        if same.sum() < 10:
            break
        a = resid[d:][same].ravel()
        b = resid[:-d][same].ravel()
        c_hat.append(float(a @ b / np.sqrt((a @ a) * (b @ b))))
        n_pairs.append(int(same.sum()))
    if not c_hat:
        return 0.0, np.array([])
    c_hat, n_pairs = np.array(c_hat), np.array(n_pairs, dtype=float)
    lags = np.arange(1, len(c_hat) + 1)
    loss = lambda r: float(np.sum(n_pairs * (c_hat - r ** lags) ** 2))
    rho = minimize_scalar(loss, bounds=(0.0, 0.999), method="bounded").x
    return float(rho), c_hat
