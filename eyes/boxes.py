"""Crop labels from animal boxes, and the vote of a crop that may show only background.

Training labels only: MegaDetector v4 boxes (iWildCam 2022 metadata on LILA) say which of
the 9 crops hold the animal. At test time nothing here is used except the trained classifier.

Why crops need an 'empty' class. Labelling every crop with the clip's species teaches the
crop classifier "this scenery -> this species" (two thirds of crops show no animal), which
fails on new cameras. Instead each training crop gets the species if it holds the animal and
'empty' if it shows none of it. Then, with Y the species and A whether the crop shows the
animal, and assuming background looks the same whatever the species,

    p(c | y) = P(A=1) p(c | A=1, y) + P(A=0) p(c | A=0)

and Bayes' rule on the classifier's q(s | c), s in {species, empty}, gives the vote

    l(y) = log p(c | y) / p(c) = log( q(y | c) / pi(y) + q(empty | c) )

with pi the species frequencies of animal crops. P(A=1) cancels. A background crop
(q(empty) ~ 1) votes 0 for every species; an animal crop votes log q(y|c) - log pi(y).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import boxes

EMPTY = -1          # marker used before the class index is known


def load_detections(md_file: str | Path) -> dict:
    """file name (no folder) -> list of (conf, x, y, w, h) animal boxes, normalised coordinates."""
    md = json.load(open(md_file))
    out = {}
    for im in md["images"]:
        out[im["file"].split("/")[-1]] = [(d["conf"], *d["bbox"]) for d in im.get("detections", [])
                                          if d["category"] == "1"]
    return out


def crop_labels(frames: pd.DataFrame, det: dict, cover: float = 0.5, conf: float = 0.5,
                free_conf: float = 0.2) -> np.ndarray:
    """(N, 9) crop content: 1 = holds >= `cover` of an animal box (conf >= `conf`),
    0 = touches no animal box at all (conf >= `free_conf`), -1 = unusable (partial, or no
    boxes in a frame that should have an animal). Frames labelled empty (y_frame == 0) are all 0.
    """
    crops = boxes()[1:]                                                     # (9, 4) x0 y0 x1 y1
    out = np.full((len(frames), 9), -1, dtype=np.int8)
    names = frames["path"].str.split("/").str[-1].values
    empty = frames["y_frame"].values == 0
    for n, name in enumerate(names):
        if empty[n]:
            out[n] = 0
            continue
        dets = det.get(name)
        if not dets or max(d[0] for d in dets) < conf:
            continue                                                         # animal not found: skip frame
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
    """log q(s|c) over K species + 'empty' (last) -> votes l(y) = log(q(y|c)/pi(y) + q(empty|c))."""
    K = log_q.shape[-1] - 1
    return np.logaddexp(log_q[..., :K] - log_pi, log_q[..., K:])
