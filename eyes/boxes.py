"""Crop labels from animal boxes, and the vote of a crop that may show only background.

The boxes are only used for training labels. MegaDetector v4 boxes (iWildCam 2022 metadata
on LILA) say which of the 9 crops hold the animal. At test time only the trained classifier
is used.

Why crops need an 'empty' class. If every crop is labelled with the clip's species, the crop
classifier learns "this scenery -> this species" (two thirds of crops show no animal), and
that fails on new cameras. So each training crop gets the species if it holds the animal and
'empty' if it shows none of it. Let Y be the species and A whether the crop shows the animal,
and assume background looks the same whatever the species. Then

    p(c | y) = P(A=1) p(c | A=1, y) + P(A=0) p(c | A=0)

and Bayes' rule on the classifier's q(s | c), s in {species, empty}, gives the vote

    l(y) = log p(c | y) / p(c) = log( q(y | c) / pi(y) + q(empty | c) )

with pi the species frequencies of animal crops. P(A=1) cancels. A background crop
(q(empty) ~ 1) votes 0 for every species. An animal crop votes log q(y|c) - log pi(y).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import boxes

EMPTY = -1          # marker used before the class index is known


def load_detections(md_file: str | Path) -> dict:
    """File name (no folder) -> list of its animal boxes (conf, x, y, w, h), normalised coordinates."""
    md = json.load(open(md_file))
    out = {}
    for im in md["images"]:
        out[im["file"].split("/")[-1]] = [(d["conf"], *d["bbox"]) for d in im.get("detections", [])
                                          if d["category"] == "1"]
    return out


def crop_labels(frames: pd.DataFrame, det: dict, cover: float = 0.5, conf: float = 0.5,
                free_conf: float = 0.2) -> np.ndarray:
    """(N, 9) crop labels from the animal boxes.

    1 if the crop holds >= `cover` of an animal box with conf >= `conf`, 0 if it touches no
    animal box with conf >= `free_conf`, and -1 if unusable (a partial animal, or no boxes in
    a frame that should have an animal). Frames labelled empty (y_frame == 0) get all 0.
    """
    crops = boxes()[1:]  # (9, 4) x0 y0 x1 y1
    out = np.full((len(frames), 9), -1, dtype=np.int8)
    names = frames["path"].str.split("/").str[-1].values
    empty = frames["y_frame"].values == 0
    for n, name in enumerate(names):
        if empty[n]:
            out[n] = 0
            continue
        dets = det.get(name)
        if not dets or max(d[0] for d in dets) < conf:
            continue  # animal not found, skip the frame
        holds, touches = np.zeros(9, bool), np.zeros(9, bool)
        for c, x, y, w, h in dets:
            ix = np.minimum(x + w, crops[:, 2]) - np.maximum(x, crops[:, 0])
            iy = np.minimum(y + h, crops[:, 3]) - np.maximum(y, crops[:, 1])
            inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
            if c >= conf:
                holds |= inter >= cover * max(w * h, 1e-9)
            if c >= free_conf:
                touches |= (ix > 0) & (iy > 0)
        out[n] = np.where(holds, 1, np.where(touches, -1, 0))
    return out


def mixture_votes(log_q: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Crop votes l(y) = log(q(y|c)/pi(y) + q(empty|c)), derived in the module docstring.

    log_q is log q(s|c) over the K species plus 'empty' as the last column.
    """
    K = log_q.shape[-1] - 1
    return np.logaddexp(log_q[..., :K] - log_pi, log_q[..., K:])
