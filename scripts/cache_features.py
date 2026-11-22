"""Step 2: encode every frame once (glance, 9 crops, crop previews) and cache it.

    python scripts/cache_features.py --frames $EYES_DATA/processed/iwildcam/frames.csv \
        --images $EYES_DATA/raw/iwildcam_v2.0 --encoders dinov2 clip --hires \
        --out-root $EYES_DATA/features/iwildcam

Each frame is decoded once and fed to every encoder listed. For encoder E it writes
<out-root>/E/chunk_00000.npz, chunk_00001.npz, ..., each holding `chunk` consecutive rows
of frames.csv. Per frame:
  g (D)           glance feature
  crops (9, D)    crop features
  ubar (9, D)     crop previews from the glance's patch tokens
  thumb (48, 48)  grayscale thumbnail
  ok              False if the image could not be read
  attn (9,)       share of the glance's CLS attention inside each crop (B3, not for SigLIP 2)
With --hires it also writes <out-root>/E_hr/, with g of the whole 448 x 448 frame in one
pass at 2x resolution (baseline B1). Features are stored as float16. Finished chunks are
skipped, so an interrupted run resumes where it stopped.

Outputs stay on the GPU until a chunk is complete, which keeps a large GPU busy. Then one
copy moves the chunk to the CPU and a background thread writes it.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
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


@torch.inference_mode()
def encode(enc: FrozenEncoder, masks, x, glances, crops, hires: bool) -> tuple[dict, dict | None]:
    """All outputs of one encoder for one batch, as float16 tensors left on the GPU."""
    g, patches = enc(glances)
    cf, _ = enc(crops.flatten(0, 1))
    views = {"g": g.half(), "crops": cf.reshape(len(x), 9, -1).half(),
             "ubar": crop_previews(patches, masks).half()}
    if enc.name != "siglip2":
        a = enc.cls_attention(glances)                                           # (B, n, n)
        views["attn"] = (torch.einsum("bij,kij->bk", a, masks.float()) / a.sum(dim=(1, 2))[:, None]).half()
    hr = {"g": enc(x)[0].half()} if hires else None
    return views, hr


def split(values: list, n: int):
    """Concatenate a buffer, return (first n rows as numpy, the rest kept in the same form)."""
    if torch.is_tensor(values[0]):
        t = torch.cat(values)
        return t[:n].cpu().numpy(), [t[n:]]
    a = np.concatenate(values)
    return a[:n], [a[n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--images", required=True, help="dataset root that frames.csv paths are relative to")
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--hires", action="store_true", help="also encode the 448 frame itself (B1)")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32, help="frames per step (x10 views per encoder)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frames = pd.read_csv(args.frames)
    root = Path(args.out_root)
    dirs = {e: root / e for e in args.encoders}
    if args.hires:
        dirs.update({f"{e}_hr": root / f"{e}_hr" for e in args.encoders})
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    n_chunks = int(np.ceil(len(frames) / args.chunk))
    todo = [c for c in range(n_chunks) if not all((d / f"chunk_{c:05d}.npz").exists() for d in dirs.values())]
    print(f"{len(frames)} frames, {n_chunks} chunks, {len(todo)} to do, outputs: {', '.join(dirs)}", flush=True)
    if not todo:
        return

    dev = pick_device()
    encs = {e: FrozenEncoder(e, dev) for e in args.encoders}
    masks = {e: torch.from_numpy(token_masks(enc.grid)).to(dev) for e, enc in encs.items()}
    img_root = Path(args.images)

    # One loader over every pending row (workers start once), flushed chunk by chunk.
    bounds = [(c, c * args.chunk, min((c + 1) * args.chunk, len(frames))) for c in todo]
    rows = np.concatenate([np.arange(lo, hi) for _, lo, hi in bounds])
    paths = [str(img_root / p) for p in frames["path"].values[rows]]
    loader = DataLoader(Frames(paths), batch_size=args.batch, num_workers=args.workers, pin_memory=dev.type == "cuda",
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    buf = {key: {} for key in dirs}       # output -> array name -> list of GPU tensors / CPU arrays
    writer, pending = ThreadPoolExecutor(max_workers=2), []
    n_buf, b_i, n_done, t0 = 0, 0, 0, time.time()

    def add(key, name, value):
        buf[key].setdefault(name, []).append(value)

    for x, thumb, ok in loader:
        x = x.to(dev, non_blocking=True).float() / 255.0
        glances, crops = make_views(x)
        for e, enc in encs.items():
            views, hr = encode(enc, masks[e], x, glances, crops, args.hires)
            for k, v in views.items():
                add(e, k, v)
            add(e, "thumb", thumb.numpy())
            add(e, "ok", ok.numpy())
            if hr is not None:
                add(f"{e}_hr", "g", hr["g"])
                add(f"{e}_hr", "ok", ok.numpy())
        n_buf += len(x)
        # flush every chunk that is now complete (batches may straddle a chunk boundary)
        while b_i < len(bounds) and n_buf >= bounds[b_i][2] - bounds[b_i][1]:
            c, lo, hi = bounds[b_i]
            n = hi - lo
            for key, d in dirs.items():
                out = {}
                for name, values in buf[key].items():
                    out[name], buf[key][name] = split(values, n)              # keep the overflow
                pending.append(writer.submit(np.savez, d / f"chunk_{c:05d}.npz", **out))
            n_buf -= n
            b_i += 1
            n_done += n
            dt = time.time() - t0
            print(f"chunk {c + 1}/{n_chunks}: {n_done} frames in {dt:.0f}s ({n_done / dt:.1f} frames/s, "
                  f"{len(encs)} encoder(s){' + hires' if args.hires else ''})", flush=True)
    for p in pending:
        p.result()                         # re-raise any write error
    writer.shutdown()


if __name__ == "__main__":
    main()
