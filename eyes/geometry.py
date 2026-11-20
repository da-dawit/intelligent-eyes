"""Where the eyes can look.

Every frame has 10 *positions*:
    k = 0      the glance: the whole frame at 224 px
    k = 1..9   the looks: a 3x3 grid of crops, each half the frame's side (2x zoom)

Crops start at 0, 1/4 and 1/2 of the side, so neighbouring crops overlap by half.
Frames are stored at 448x448, so a crop is an exact 224x224 slice and the glance is
the 448 image downsampled to 224: the zoom is a true 2x with no extra resampling.
"""
from __future__ import annotations

import numpy as np

N_POS = 10            # glance + 9 crops
N_CROPS = 9
CROP_FRAC = 0.5       # crop side / frame side  -> 2x zoom
OFFSETS = (0.0, 0.25, 0.5)
BASE = 448            # stored frame size (pixels)
VIEW = 224            # encoder input size (pixels)


def boxes() -> np.ndarray:
    """(10, 4) array of boxes (x0, y0, x1, y1) in [0, 1] frame coordinates.

    Row 0 is the glance; rows 1..9 are the crops in reading order
    (top-left, top-middle, ..., bottom-right).
    """
    out = [(0.0, 0.0, 1.0, 1.0)]
    for y0 in OFFSETS:
        for x0 in OFFSETS:
            out.append((x0, y0, x0 + CROP_FRAC, y0 + CROP_FRAC))
    return np.asarray(out, dtype=np.float64)


def pixel_slices(base: int = BASE):
    """Pixel slices (rows, cols) of the 9 crops in a base x base frame."""
    side = int(round(CROP_FRAC * base))
    out = []
    for y0 in OFFSETS:
        for x0 in OFFSETS:
            r, c = int(round(y0 * base)), int(round(x0 * base))
            out.append((slice(r, r + side), slice(c, c + side)))
    return out


def overlap_correlation() -> np.ndarray:
    """Geometric repetition table A (10 x 10) from how much two positions overlap.

        A[k, k'] = |box_k  ∩  box_k'| / sqrt(|box_k| |box_k'|)

    This is the cosine similarity between the two regions' indicator functions, so A
    is a Gram matrix of unit vectors: symmetric, positive semidefinite, ones on the
    diagonal -- a valid correlation matrix. Examples: glance vs crop = 0.5, side
    neighbours = 0.5, diagonal neighbours = 0.25, crops two apart = 0.

    This is the fallback for R (proposal, risk table: "R from crop overlap only").
    """
    b = boxes()
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    return (ix * iy) / np.sqrt(area[:, None] * area[None, :])


def token_masks(grid: int) -> np.ndarray:
    """(9, grid, grid) boolean masks: which glance patch tokens fall inside each crop.

    A token belongs to a crop if its centre lies inside the crop's box. Averaging the
    glance's patch tokens under mask k gives u_bar[f, k], the cheap low-resolution
    preview of crop k that the look predictor (eq. 2) uses.
    """
    centres = (np.arange(grid) + 0.5) / grid
    masks = np.zeros((N_CROPS, grid, grid), dtype=bool)
    for k, (x0, y0, x1, y1) in enumerate(boxes()[1:]):
        rows = (centres >= y0) & (centres < y1)
        cols = (centres >= x0) & (centres < x1)
        masks[k] = rows[:, None] & cols[None, :]
    return masks
