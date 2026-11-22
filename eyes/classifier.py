"""Linear classifiers on frozen features (logistic regression, L26-27) plus a temperature.

Each encoder gets two classifiers, one for glances and one for crops. By default crops are
labelled with their clip's species even when the animal is elsewhere, so a calibrated crop
classifier learns to fall back toward the prior on empty crops. That gives a vote near 0.
Temperature scaling on held-out cameras makes assumption A3 (calibration) hold better on
unseen cameras.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize_scalar


@dataclass
class LinearProbe:
    mean: np.ndarray            # (D,) feature standardisation
    std: np.ndarray
    W: np.ndarray               # (D, K)
    b: np.ndarray               # (K,)
    temperature: float = 1.0

    def logits(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float32) - self.mean) / self.std
        return Z @ self.W + self.b

    def log_probs(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X) / self.temperature
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def train_probe(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 25, lr: float = 2e-3,
                weight_decay: float = 1e-4, batch: int = 4096, seed: int = 0,
                device: str | None = None) -> LinearProbe:
    """Multinomial logistic regression by AdamW on standardised features."""
    torch.manual_seed(seed)
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    mean = X.mean(0).astype(np.float32)
    std = (X.std(0) + 1e-6).astype(np.float32)
    Xt = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    lin = torch.nn.Linear(X.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g).to(dev)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(lin(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    W = lin.weight.detach().cpu().numpy().T.copy()
    b = lin.bias.detach().cpu().numpy().copy()
    return LinearProbe(mean, std, W, b)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T that minimises the held-out NLL of softmax(logits / T)."""
    def nll(logT):
        z = logits / np.exp(logT)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
