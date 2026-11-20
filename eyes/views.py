"""Turn a frame into its 10 views: one glance and nine 2x crops.

The frame is resized once to 448 x 448 (squashed, not cropped, so nothing at the edges
is lost; every view shares the same aspect distortion). The glance is that image
downsampled to 224; each crop is an exact 224 x 224 slice of it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .geometry import BASE, VIEW, pixel_slices

THUMB = 48   # grayscale thumbnail kept for the motion baseline (B4)


def load_frame(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an image -> (448x448x3 uint8 frame, 48x48 uint8 grayscale thumbnail)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        frame = im.resize((BASE, BASE), Image.BICUBIC)
    thumb = frame.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    return np.asarray(frame), np.asarray(thumb)


def make_views(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """frames: (B, 3, 448, 448) floats in [0, 1].

    Returns glances (B, 3, 224, 224) and crops (B, 9, 3, 224, 224).
    """
    glances = F.interpolate(frames, size=(VIEW, VIEW), mode="bilinear", antialias=True, align_corners=False)
    crops = torch.stack([frames[:, :, rs, cs] for rs, cs in pixel_slices()], dim=1)
    return glances, crops


def crop_previews(patches: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Mean glance patch token inside each crop: u_bar (B, 9, D).

    patches: (B, n, n, D) glance patch tokens;  masks: (9, n, n) bool (geometry.token_masks).
    """
    m = masks.to(patches.dtype)
    return torch.einsum("bijd,kij->bkd", patches, m) / m.sum(dim=(1, 2))[None, :, None]


def motion_scores(thumbs: np.ndarray) -> np.ndarray:
    """B4 baseline: mean absolute change of each crop region from the previous frame.

    thumbs: (T, 48, 48) uint8 thumbnails of one clip in time order -> (T, 9) scores.
    The first frame is compared with the second (a one-frame clip scores 0).
    """
    T = len(thumbs)
    x = thumbs.astype(np.float32) / 255.0
    out = np.zeros((T, 9), dtype=np.float32)
    if T == 1:
        return out
    diff = np.abs(np.diff(x, axis=0))                 # (T-1, 48, 48)
    diff = np.concatenate([diff[:1], diff], axis=0)    # frame 0 uses the 0->1 change
    for k, (rs, cs) in enumerate(pixel_slices(THUMB)):
        out[:, k] = diff[:, rs, cs].mean(axis=(1, 2))
    return out
