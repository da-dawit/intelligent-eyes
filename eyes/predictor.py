"""Predict a look before paying for it (proposal eq. 2, derivations Sec. 5).

The prior prediction is a ridge regression (least squares)
    mu_j = W [u_bar_fk ; g_f] + b
where u_bar_fk is the mean of the glance's patch tokens inside crop k (a cheap
low-resolution preview) and g_f is frame f's glance feature.

Once real looks L are taken, the other predictions are corrected by Gaussian
conditioning (Prop. 5.1)
    mu_{j|L}     = mu_j + R_jL R_LL^{-1} (l_L - mu_L)
    sigma2_{j|L} = sigma2 * (1 - R_jL R_LL^{-1} R_Lj)

Only the looks are conditioned on. The predictor already takes the glance as input, so
conditioning on the glances again would count them twice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgePredictor:
    x_mean: np.ndarray        # (D,)
    x_scale: np.ndarray       # (D,)
    y_mean: np.ndarray        # (K,)
    W: np.ndarray             # (D, K)
    lam: float                # chosen ridge strength
    sigma2: float             # held-out error variance per species component
    cv_mse: dict              # lambda -> cross-validated error, for the record

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.x_mean) / self.x_scale
        return Z @ self.W + self.y_mean


def _centred_mse(resid: np.ndarray) -> float:
    """Mean squared residual per species component after centring across species.

    Only differences between species change the belief (F 1 = 0), so the part of
    the error shared by all species is not counted.
    """
    r = resid - resid.mean(axis=-1, keepdims=True)
    K = r.shape[-1]
    return float(np.sum(r ** 2) / (r.shape[0] * (K - 1)))


def fit_ridge(X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
              lambdas=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), n_folds: int = 5,
              seed: int = 0) -> RidgePredictor:
    """Fit W = (Z^T Z + lam N I)^{-1} Z^T Y on standardised inputs Z.

    lam is chosen by grouped K-fold cross-validation. The groups are cameras, so the error
    is measured on unseen cameras. sigma2 is the cross-validated error at that lam.
    One eigendecomposition of Z^T Z per fold covers every lam.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    fold_of_group = {g: i % n_folds for i, g in enumerate(uniq)}
    fold = np.array([fold_of_group[g] for g in groups])

    sse = {lam: 0.0 for lam in lambdas}
    count = 0
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
        my = Y[tr].mean(0)
        Ztr, Zte = (X[tr] - mx) / sx, (X[te] - mx) / sx
        G = Ztr.T @ Ztr
        B = Ztr.T @ (Y[tr] - my)
        evals, Q = np.linalg.eigh(G)
        QtB = Q.T @ B
        n = tr.sum()
        for lam in lambdas:
            W = Q @ (QtB / (evals + lam * n)[:, None])
            resid = Y[te] - (Zte @ W + my)
            r = resid - resid.mean(axis=1, keepdims=True)
            sse[lam] += float(np.sum(r ** 2))
        count += te.sum()

    K = Y.shape[1]
    cv_mse = {lam: sse[lam] / (count * (K - 1)) for lam in lambdas}
    best = min(cv_mse, key=cv_mse.get)

    mx, sx, my = X.mean(0), X.std(0) + 1e-6, Y.mean(0)
    Z = (X - mx) / sx
    W = np.linalg.solve(Z.T @ Z + best * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y - my))
    return RidgePredictor(mx, sx, my, W, best, cv_mse[best], cv_mse)


def out_of_fold(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lam: float, n_folds: int = 5,
                seed: int = 0) -> np.ndarray:
    """Predictions for every row from a ridge fit that never saw that row's group.

    Used for the held-out clips that choose eps, so their predictions are as honest as
    the test clips'.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    fold_of_group = {g: i % n_folds for i, g in enumerate(uniq)}
    fold = np.array([fold_of_group[g] for g in groups])
    out = np.zeros_like(Y)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0:
            continue
        mx, sx, my = X[tr].mean(0), X[tr].std(0) + 1e-6, Y[tr].mean(0)
        Z = (X[tr] - mx) / sx
        W = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y[tr] - my))
        out[te] = ((X[te] - mx) / sx) @ W + my
    return out


def condition_on_looks(mu_C: np.ndarray, sigma2: float, R_LC: np.ndarray, R_LL: np.ndarray,
                       votes_L: np.ndarray, mu_L: np.ndarray, ridge: float = 1e-6):
    """Gaussian conditioning of candidate predictions on the looks taken (Prop. 5.1).

    mu_C           (C, K) prior predictions for the candidates
    R_LC           (|L|, C) correlations between taken looks and candidates
    R_LL           (|L|, |L|)
    votes_L, mu_L  (|L|, K) actual votes and prior predictions of the taken looks
    Returns mu_cond (C, K) and var_cond (C,).
    """
    if R_LL.shape[0] == 0:
        return mu_C, np.full(mu_C.shape[0], sigma2)
    b = np.linalg.solve(R_LL + ridge * np.eye(R_LL.shape[0]), R_LC)    # (|L|, C)
    mu_cond = mu_C + b.T @ (votes_L - mu_L)
    var_cond = sigma2 * np.clip(1.0 - np.sum(R_LC * b, axis=0), 0.0, None)
    return mu_cond, var_cond
