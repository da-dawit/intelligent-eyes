"""Read the cached encoder outputs written by scripts/cache_features.py."""
from pathlib import Path

import numpy as np


def load_features(feat_dir, n_rows: int) -> dict:
    """Join all chunk_*.npz files into one dict of arrays with n_rows rows each.

    Keys are g (N, D), crops (N, 9, D), ubar (N, 9, D), thumb (N, 48, 48) and ok (N,).
    Raises an error if caching has not finished.
    """
    files = sorted(Path(feat_dir).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached features in {feat_dir}")
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    if len(out["g"]) != n_rows:
        raise RuntimeError(f"{feat_dir}: {len(out['g'])} cached rows, expected {n_rows}. Caching is incomplete.")
    return out
