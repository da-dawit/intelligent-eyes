"""Step 2: encode every frame once -- glance, 9 crops, crop previews -- and cache it.

    python scripts/cache_features.py --frames $EYES_DATA/processed/iwildcam/frames.csv \
        --images $EYES_DATA/raw/iwildcam_v2.0 --encoder dinov2 \
        --out $EYES_DATA/features/iwildcam/dinov2

Writes chunk_00000.npz, chunk_00001.npz, ... each holding `chunk` consecutive rows of
frames.csv. Finished chunks are skipped, so an interrupted run resumes where it stopped.
Per frame: g (D) glance feature, crops (9, D) crop features, ubar (9, D) crop previews
from the glance's patch tokens, thumb (48, 48) grayscale thumbnail. Stored as float16.
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
    root = Path(args.images)

    for c in todo:
        t0 = time.time()
        lo, hi = c * args.chunk, min((c + 1) * args.chunk, len(frames))
        paths = [str(root / p) for p in frames["path"].iloc[lo:hi]]
        loader = DataLoader(Frames(paths), batch_size=args.batch, num_workers=args.workers)
        G, C, U, TH, OK = [], [], [], [], []
        for x, thumb, ok in loader:
            x = x.to(dev).float() / 255.0
            glances, crops = make_views(x)
            g, patches = enc(glances)
            ubar = crop_previews(patches, masks)
            cf, _ = enc(crops.flatten(0, 1))
            G.append(g.cpu().numpy().astype(np.float16))
            U.append(ubar.cpu().numpy().astype(np.float16))
            C.append(cf.reshape(len(x), 9, -1).cpu().numpy().astype(np.float16))
            TH.append(thumb.numpy())
            OK.append(ok.numpy())
        np.savez(out / f"chunk_{c:05d}.npz", g=np.concatenate(G), crops=np.concatenate(C),
                 ubar=np.concatenate(U), thumb=np.concatenate(TH), ok=np.concatenate(OK))
        dt = time.time() - t0
        print(f"chunk {c + 1}/{n_chunks}: {hi - lo} frames in {dt:.0f}s ({(hi - lo) / dt:.1f} frames/s)", flush=True)


if __name__ == "__main__":
    main()
