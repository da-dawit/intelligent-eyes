"""The repetition matrix R and the evidence weights w = R^{-1} 1 (proposal eq. 1).

Observation j = (f, k) is frame f seen at position k (see geometry.py). Two observations
repeat each other by

    R[(f, k), (f', k')] = A[k, k'] * rho ** |f - f'|          (derivations, Sec. 3.1)

So R = T_rho kron A in frame-major order. A is the 10x10 table of how much two positions
repeat each other inside one frame, and T_rho is the AR(1) correlation in time.
Observation (f, k) sits at row f * 10 + k.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import N_POS


def frame_correlation(T: int, rho: float) -> np.ndarray:
    """T_rho, the T x T AR(1) correlation matrix with T_rho[f, f'] = rho ** |f - f'|."""
    f = np.arange(T)
    return rho ** np.abs(f[:, None] - f[None, :])


def clip_correlation(T: int, A: np.ndarray, rho: float, tau: float = 0.0) -> np.ndarray:
    """R = T_rho kron A + tau 1_c 1_c^T for a clip of T frames, a (10T, 10T) matrix.

    tau adds one error shared by every crop vote in the clip (1_c marks the crop positions,
    k >= 1). For example the crop classifier can misread this camera's light or background.
    This error does not decay with time or shrink with more looks, so the total weight of n
    crop votes levels off near 1/tau instead of growing like n. tau = 0 is the model of Sec. 3.1.
    """
    R = np.kron(frame_correlation(T, rho), A)
    if tau > 0:
        crop = (np.arange(10 * T) % A.shape[0]) != 0
        R = R + tau * np.outer(crop, crop)
    return R


def obs_index(f: int, k: int) -> int:
    """Row of observation (frame f, position k) in R."""
    return f * N_POS + k


def nearest_correlation(A: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Make a symmetric matrix a valid correlation matrix.

    Clips negative eigenvalues to `floor`, then rescales so the diagonal is 1
    (derivations, Sec. 3.1). An estimated A can be slightly indefinite because each
    entry is estimated separately.
    """
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    A = (vecs * np.clip(vals, floor, None)) @ vecs.T
    d = np.sqrt(np.diag(A))
    return A / d[:, None] / d[None, :]


def shrink(A: np.ndarray, lam: float) -> np.ndarray:
    """A_lam = (1 - lam) A + lam I, which trusts the within-frame repetition table less.

    This came up on synthetic clips, before any real data. The common-mean model behind
    eq. (1) says every vote is a noisy copy of one L. That holds for repeats in time (same
    crop, consecutive frames) but not for crops that see different parts of the frame.
    With the overlap table the four corner crops tile the frame, so A is singular and
    w = A^{-1} 1 gives the centre and edge crops weight 0. Shrinking A toward I makes R
    positive definite (condition number < 10 for lam >= 0.3 with the overlap table). The
    centre crop's weight turns positive at lam ~ 0.35 and every weight at lam ~ 0.65.
    A negative glance weight is fine. It is GLS cancelling noise that the glance shares
    with the crops.

    lam = 0 is the full model and lam = 1 treats positions in one frame as independent.
    Time (rho) is not shrunk. lam is chosen by held-out log-likelihood (fit.choose_shrinkage).
    """
    return (1.0 - lam) * A + lam * np.eye(A.shape[0])


# evidence weights for eq. (1)

def evidence_weights(R_S: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Weights w = (R_S + ridge I)^{-1} 1, how much each observation counts (Prop. 3.1).

    The ridge keeps R_S invertible when two observations nearly repeat each other (failure
    mode 5). As ridge -> 0 this gives the pseudo-inverse limit of Prop. 3.2.
    """
    n = R_S.shape[0]
    return np.linalg.solve(R_S + ridge * np.eye(n), np.ones(n))


def effective_count(R_S: np.ndarray, ridge: float = 1e-6) -> float:
    """n_eff = 1^T R_S^{-1} 1, the number of independent views the observations are worth."""
    return float(evidence_weights(R_S, ridge).sum())


def ar1_effective_frames(T: int, rho: float) -> float:
    """n_eff of T glances alone in closed form, (T - (T-2) rho) / (1 + rho) (Prop. 3.3)."""
    if T == 1:
        return 1.0
    return (T - (T - 2) * rho) / (1 + rho)


# fitting A and rho from residuals

def class_conditional_residuals(votes: np.ndarray, labels: np.ndarray, n_classes: int,
                                mask: np.ndarray | None = None) -> np.ndarray:
    """Residuals e = vote - E[vote | position, true species], centred across species.

    votes   (N, 10, K) votes of N frames at the 10 positions
    labels  (N,) true species index of each frame's clip
    mask    optional (N, 10) bool, the observations that enter eq. (1). With gated fusion,
            crops that show only background are left out. The means use only these
            observations and the other residuals are set to 0.

    Assumption (A2) says that given the species the votes are Gaussian with correlation R,
    so R is estimated from what is left after removing the species-and-position mean.
    Each residual vector is then centred across species, because adding the same amount
    to every species changes nothing (F 1 = 0, Lemma 6.4a).
    """
    resid = np.zeros_like(votes, dtype=np.float64)
    m = np.ones(votes.shape[:2], bool) if mask is None else mask
    for c in range(n_classes):
        idx = labels == c
        if idx.any():
            w = m[idx][..., None].astype(np.float64)
            mean = (votes[idx] * w).sum(axis=0, keepdims=True) / np.clip(w.sum(axis=0, keepdims=True), 1, None)
            resid[idx] = (votes[idx] - mean) * w
    return (resid - resid.mean(axis=2, keepdims=True)) * m[..., None]


def fit_A(resid: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """A[k, k'] = pooled correlation of residuals at positions k and k' of the same frame.

    resid is (N, 10, K) and is pooled over frames and species components. With a mask
    ((N, 10) bool), each pair (k, k') uses only the frames where both observations enter eq. (1).
    """
    N, P, K = resid.shape
    if mask is None:
        flat = resid.transpose(1, 0, 2).reshape(P, N * K)       # one long vector per position
        cov = flat @ flat.T
        d = np.sqrt(np.diag(cov))
        return nearest_correlation(cov / d[:, None] / d[None, :])
    m = mask.astype(np.float64)
    r = resid * m[..., None]
    cross = np.einsum("npk,nqk->pq", r, r)                    # sum over frames where both are present
    sq = (r ** 2).sum(axis=2)                                 # (N, P)
    a = (sq.T @ m)                                            # a[k, l] = sum_n m_nl |r_nk|^2
    corr = cross / np.sqrt(np.clip(a * a.T, 1e-300, None))
    np.fill_diagonal(corr, 1.0)
    return nearest_correlation(corr)


def fit_rho(resid: np.ndarray, clip_ids: np.ndarray, frame_idx: np.ndarray, max_lag: int = 4,
            mask: np.ndarray | None = None):
    """Fit rho so that same-position residuals d frames apart correlate like rho ** d.

    resid      (N, 10, K) residuals of N frames
    clip_ids   (N,) clip of each frame
    frame_idx  (N,) position of the frame in its clip
    mask       optional (N, 10) bool, only pairs where both observations enter eq. (1) count

    rho_hat minimises sum_d n_d (c_hat_d - rho^d)^2, so each lag is weighted by its number
    of pairs. Returns (rho_hat, c_hat), with c_hat[d-1] the measured pooled correlation at lag d.
    """
    order = np.lexsort((frame_idx, clip_ids))
    resid, clip_ids, frame_idx = resid[order], clip_ids[order], frame_idx[order]
    m = np.ones(resid.shape[:2], bool) if mask is None else mask[order]
    c_hat, n_pairs = [], []
    for d in range(1, max_lag + 1):
        same = (clip_ids[d:] == clip_ids[:-d]) & (frame_idx[d:] - frame_idx[:-d] == d)
        if same.sum() < 10:
            break
        both = (m[d:] & m[:-d])[same][..., None]
        a = (resid[d:][same] * both).ravel()
        b = (resid[:-d][same] * both).ravel()
        c_hat.append(float(a @ b / np.sqrt((a @ a) * (b @ b))))
        n_pairs.append(int(same.sum()))
    if not c_hat:
        return 0.0, np.array([])
    c_hat, n_pairs = np.array(c_hat), np.array(n_pairs, dtype=float)
    lags = np.arange(1, len(c_hat) + 1)
    loss = lambda r: float(np.sum(n_pairs * (c_hat - r ** lags) ** 2))
    rho = minimize_scalar(loss, bounds=(0.0, 0.999), method="bounded").x
    return float(rho), c_hat
