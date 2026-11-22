"""Recompute only the per-crop CLS attention (baseline B3) of an existing cache.

    python scripts/fix_attention.py --frames ... --images ... --encoder dinov3 --out attn_dinov3.npy
    python scripts/fix_attention.py --frames ... --apply attn_dinov3.npy --feature-dir .../features/iwildcam/dinov3

The first form runs the glances through the encoder's eager copy in float32 and saves the
(N, 9) attention shares. The second writes them into every chunk of a cache, in place.
Used once, after float16 eager attention turned DINOv3's maps into NaN.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eyes.encoders import FrozenEncoder, pick_device
from eyes.geometry import token_masks
from eyes.views import make_views

from cache_features import Frames  # noqa: E402  (scripts/ is on sys.path when run as a script)


def compute(args):
    frames = pd.read_csv(args.frames)
    dev = pick_device()
    enc = FrozenEncoder(args.encoder, dev)
    masks = torch.from_numpy(token_masks(enc.grid)).to(dev).float()
    paths = [str(Path(args.images) / p) for p in frames["path"].values]
    loader = DataLoader(Frames(paths), batch_size=args.batch, num_workers=args.workers, pin_memory=dev.type == "cuda")
    out = []
    with torch.inference_mode():
        for i, (x, _, _) in enumerate(loader):
            glances, _ = make_views(x.to(dev, non_blocking=True).float() / 255.0)
            a = enc.cls_attention(glances)
            out.append((torch.einsum("bij,kij->bk", a, masks) / a.sum(dim=(1, 2))[:, None]).half().cpu())
            if i % 100 == 0:
                print(f"{(i + 1) * args.batch} / {len(paths)}", flush=True)
    attn = torch.cat(out).numpy()
    print("NaN:", int(np.isnan(attn.astype(np.float32)).sum()), "range", attn.min(), attn.max())
    np.save(args.out, attn)


def apply(args):
    attn = np.load(args.apply)
    files = sorted(Path(args.feature_dir).glob("chunk_*.npz"))
    lo = 0
    for f in files:
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        n = len(data["g"])
        data["attn"] = attn[lo:lo + n]
        np.savez(f, **data)
        lo += n
    assert lo == len(attn), (lo, len(attn))
    print(f"wrote attention into {len(files)} chunks ({lo} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--images")
    ap.add_argument("--encoder")
    ap.add_argument("--out")
    ap.add_argument("--apply")
    ap.add_argument("--feature-dir")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    apply(args) if args.apply else compute(args)


if __name__ == "__main__":
    main()
