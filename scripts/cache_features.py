"""Step 2: encode every frame once -- glance, 9 crops, crop previews -- and cache it.

    python scripts/cache_features.py --frames $EYES_DATA/processed/iwildcam/frames.csv \
        --images $EYES_DATA/raw/iwildcam_v2.0 --encoder dinov2 \
        --out $EYES_DATA/features/iwildcam/dinov2

Writes chunk_00000.npz, chunk_00001.npz, ... each holding `chunk` consecutive rows of
frames.csv. Finished chunks are skipped, so an interrupted run resumes where it stopped.
Per frame: g (D) glance feature, crops (9, D) crop features, ubar (9, D) crop previews
from the glance's patch tokens, thumb (48, 48) grayscale thumbnail, attn (9,) share of the
glance's CLS attention inside each crop (B3). Stored as float16.

With --hires (baseline B1) only g is stored: the whole 448 x 448 frame in one pass at 2x
resolution (4x the tokens). Give it its own --out, e.g. .../features/iwildcam/dinov2_hr.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from eyes.encoders import FrozenEncoder, pick_device
from eyes.geometry import token_masks
from eyes.views import crop_previews, load_frame, make_views


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            frame, thumb = load_frame(self.paths[i])
            ok = True
        except Exception:                      # unreadable file: zeros, flagged in `ok`
            frame, thumb, ok = np.zeros((448, 448, 3), np.uint8), np.zeros((48, 48), np.uint8), False
        return torch.from_numpy(frame).permute(2, 0, 1), torch.from_numpy(thumb), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--images", required=True, help="dataset root that frames.csv paths are relative to")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32, help="frames per encoder call (x10 views)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hires", action="store_true", help="B1: encode the 448 frame itself, nothing else")
    args = ap.parse_args()

    frames = pd.read_csv(args.frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_chunks = int(np.ceil(len(frames) / args.chunk))
    todo = [c for c in range(n_chunks) if not (out / f"chunk_{c:05d}.npz").exists()]
    print(f"{len(frames)} frames, {n_chunks} chunks, {len(todo)} to do")
    if not todo:
        return

    dev = pick_device()
    enc = FrozenEncoder(args.encoder, dev)
    masks = torch.from_numpy(token_masks(enc.grid)).to(dev)
    with_attn = args.encoder != "siglip2" and not args.hires
    root = Path(args.images)

    # One loader over every pending row (workers start once), flushed chunk by chunk.
    bounds = [(c, c * args.chunk, min((c + 1) * args.chunk, len(frames))) for c in todo]
    rows = np.concatenate([np.arange(lo, hi) for _, lo, hi in bounds])
    paths = [str(root / p) for p in frames["path"].values[rows]]
    loader = DataLoader(Frames(paths), batch_size=args.batch, num_workers=args.workers,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    buf = {k: [] for k in ("g", "crops", "ubar", "thumb", "ok", "attn")}
    n_buf, b_i, t0 = 0, 0, time.time()
    for x, thumb, ok in loader:
        x = x.to(dev).float() / 255.0
        buf["ok"].append(ok.numpy())
        if args.hires:
            g, _ = enc(x)
            buf["g"].append(g.cpu().numpy().astype(np.float16))
        else:
            glances, crops = make_views(x)
            g, patches = enc(glances)
            cf, _ = enc(crops.flatten(0, 1))
            buf["g"].append(g.cpu().numpy().astype(np.float16))
            buf["ubar"].append(crop_previews(patches, masks).cpu().numpy().astype(np.float16))
            buf["crops"].append(cf.reshape(len(x), 9, -1).cpu().numpy().astype(np.float16))
            buf["thumb"].append(thumb.numpy())
        if with_attn:
            a = enc.cls_attention(glances)                                       # (B, n, n)
            mass = torch.einsum("bij,kij->bk", a, masks.float()) / a.sum(dim=(1, 2))[:, None]
            buf["attn"].append(mass.cpu().numpy().astype(np.float16))
        n_buf += len(x)
        # flush every chunk that is now complete (batches may straddle a chunk boundary)
        while b_i < len(bounds) and n_buf >= bounds[b_i][2] - bounds[b_i][1]:
            c, lo, hi = bounds[b_i]
            n = hi - lo
            arrays = {k: np.concatenate(v) for k, v in buf.items() if v}
            np.savez(out / f"chunk_{c:05d}.npz", **{k: v[:n] for k, v in arrays.items()})
            buf = {k: [arrays[k][n:]] if k in arrays else [] for k in buf}       # keep the overflow
            n_buf -= n
            b_i += 1
            dt = time.time() - t0
            print(f"chunk {c + 1}/{n_chunks}: {n} frames in {dt:.0f}s ({n / dt:.1f} frames/s)", flush=True)
            t0 = time.time()


if __name__ == "__main__":
    main()
