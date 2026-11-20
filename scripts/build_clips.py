"""Step 1: build the frame table of clips.

    python scripts/build_clips.py --dataset iwildcam --root $EYES_DATA/raw/iwildcam_v2.0 \
        --out $EYES_DATA/processed/iwildcam [--max-train-clips 20000 --max-test-clips 6000]
"""
import argparse
import json
from pathlib import Path

from eyes.data import build_ddr, build_iwildcam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iwildcam", "ddr"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train-clips", type=int, default=None)
    ap.add_argument("--max-test-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    caps = {"train": args.max_train_clips, "test": args.max_test_clips}
    if args.dataset == "iwildcam":
        frames, stats = build_iwildcam(args.root, max_clips=caps, seed=args.seed)
    else:
        frames, stats = build_ddr(args.root, max_clips=caps, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out / "frames.csv", index=False)
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
