"""Check an encoder loads, fp16 agrees with fp32, and measure views per second.

    python scripts/benchmark_encoder.py --encoder dinov2 --images path/to/folder
"""
import argparse
import glob
import time

import numpy as np
import torch

from eyes.encoders import FrozenEncoder, pick_device
from eyes.geometry import token_masks
from eyes.views import crop_previews, load_frame, make_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=64, help="frames to time")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.images}/**/*.jpg", recursive=True))[: args.n]
    t0 = time.time()
    frames = np.stack([load_frame(p)[0] for p in paths])
    t_load = time.time() - t0
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    dev = pick_device()
    enc16 = FrozenEncoder(args.encoder, dev, torch.float16)
    enc32 = FrozenEncoder(args.encoder, dev, torch.float32)
    glances, crops = make_views(x[:8].to(dev))
    a, pa = enc16(glances)
    b, pb = enc32(glances)
    cos = torch.nn.functional.cosine_similarity(a, b).min().item()
    print(f"{args.encoder}: device={dev} D={a.shape[1]} grid={pa.shape[1]} min cos(fp16, fp32)={cos:.5f}")

    masks = torch.from_numpy(token_masks(enc16.grid)).to(dev)
    torch.mps.synchronize() if dev.type == "mps" else None
    t0 = time.time()
    for i in range(0, len(x), 16):
        g, c = make_views(x[i:i + 16].to(dev))
        pooled, patches = enc16(g)
        ubar = crop_previews(patches, masks)
        cp, _ = enc16(c.flatten(0, 1))
    torch.mps.synchronize() if dev.type == "mps" else None
    dt = time.time() - t0
    print(f"decode+resize: {len(x) / t_load:.0f} frames/s (1 process) | encode: {10 * len(x) / dt:.0f} views/s"
          f" = {len(x) / dt:.1f} frames/s (glance + 9 crops)")


if __name__ == "__main__":
    main()
