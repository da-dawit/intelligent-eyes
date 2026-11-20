"""Videos of the eyes at work, for the project page.

    python scripts/run_eval.py ... --seeds 0 --dump proposed.pkl --dump-clips ID,ID,...   (method as proposed)
    python scripts/run_eval.py ... --seeds 0 --dump final.pkl --dump-clips ID,ID,...      (final method flags)
    python scripts/make_videos.py --proposed proposed.pkl --final final.pkl --out docs/project_page/static/videos

For each clip the controller is replayed step by step (glance, look 1, look 2) with the fitted
model, so the belief shown after every step is the one the method actually holds. Writes
compare_<clip>.mp4 (as proposed | final, side by side), teaser.mp4 (final method only) and a
poster image for each. Needs ffmpeg.
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyes import paths  # noqa: E402
from eyes.controller import run  # noqa: E402
from eyes.geometry import boxes  # noqa: E402

FPS = 12
BLUE, ORANGE = (31, 119, 180), (255, 127, 14)
FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_FILE, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def sci(name):
    """Binomial style: capitalise the genus ('urocyon cinereoargenteus' -> 'Urocyon cinereoargenteus')."""
    return name[:1].upper() + name[1:]


def trajectory(clip, model, rule):
    """Beliefs after the glance and after each of 2 looks, as the controller computes them."""
    o0, o1, o2 = (run(clip, model, rule, budget=b) for b in (0, 1, 2))
    looks = o2.looks
    entered = [clip.gate is None or bool(clip.gate[f, k - 1]) for f, k in looks]
    return dict(p=[o0.p, o1.p, o2.p], looks=looks, entered=entered, pred=o2.pred)


def crop_px(k, size):
    W, H = size
    b = boxes()[k]
    return (b[0] * W, b[1] * H, b[2] * W, b[3] * H)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


class Panel:
    """One version of the method on one clip: the footage with a corner tag and a caption strip."""

    def __init__(self, name, colour, traj, frames, names, label, w, h, scale=1.0):
        self.name, self.colour, self.t, self.frames, self.names, self.label = name, colour, traj, frames, names, label
        self.w, self.h, self.s = w, h, scale

    def prob_text(self, p):
        top = int(np.argmax(p))
        if top == self.label:
            return f"{sci(self.names[top])}  {p[top]:.2f}  (true species)"
        return f"{sci(self.names[top])}  {p[top]:.2f}    true species {sci(self.names[self.label])}  {p[self.label]:.2f}"

    def render(self, st):
        im = self.frames[st["f"]]
        W, H = im.size
        x0, y0, x1, y1 = st.get("win") or (0, 0, W, H)
        v = im.crop((round(x0), round(y0), round(x1), round(y1))).resize((self.w, self.h), Image.BICUBIC).convert("RGBA")
        over = Image.new("RGBA", v.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        s = self.s
        if st.get("box") is not None:
            sx, sy = self.w / (x1 - x0), self.h / (y1 - y0)
            b = st["box"]
            d.rectangle([(b[0] - x0) * sx + 1, (b[1] - y0) * sy + 1, (b[2] - x0) * sx - 2, (b[3] - y0) * sy - 2],
                        outline=self.colour + (255,), width=max(2, round(3 * s)))
        tag = font(round(15 * s), bold=True)                                   # corner tag
        tw = tag.getlength(self.name)
        pad = round(8 * s)
        d.rectangle([0, 0, tw + 3 * pad + 6 * s, round(30 * s)], fill=(0, 0, 0, 150))
        d.rectangle([pad, round(9 * s), pad + 6 * s, round(21 * s)], fill=self.colour + (255,))
        d.text((2 * pad + 6 * s, round(6 * s)), self.name, font=tag, fill=(255, 255, 255, 255))
        lines = [st["step"]] + ([self.prob_text(st["p"])] if st.get("p") is not None else [])
        f1, f2 = font(round(16 * s), bold=st.get("final", False)), font(round(15 * s))
        strip = round((26 + 22 * (len(lines) - 1)) * s)                         # caption strip
        d.rectangle([0, self.h - strip - pad, self.w, self.h], fill=(0, 0, 0, 160))
        y = self.h - strip - pad // 2
        for i, line in enumerate(lines):
            d.text((pad + 2, y), line, font=f1 if i == 0 else f2, fill=(255, 255, 255, 255))
            y += round(22 * s)
        return Image.alpha_composite(v, over).convert("RGB")


def timeline(panels, T):
    """Per video frame, one state per panel: frame, zoom window, box, step text, belief."""
    for f in range(T):                                                  # glance at every frame
        for _ in range(3):
            yield [dict(f=f, step=f"Glance: frame {f + 1} of {T}") for _ in panels]
    for _ in range(14):
        yield [dict(f=T - 1, step=f"After the glance ({T} frames)", p=p.t["p"][0]) for p in panels]
    for j in range(2):                                                  # two looks
        info = []
        for p in panels:
            f, k = p.t["looks"][j]
            label = f"Look {j + 1}: frame {f + 1}, crop {k}"
            if not p.t["entered"][j]:
                label += " (classified empty, not used)"
            info.append((f, k, crop_px(k, p.frames[f].size), label, p.t["p"][j], p.t["p"][j + 1]))
        for _ in range(8):
            yield [dict(f=f, box=bx, step=lab, p=pv) for f, k, bx, lab, pv, nv in info]
        for t in range(10):
            a = ease((t + 1) / 10)
            out = []
            for p, (f, k, bx, lab, pv, nv) in zip(panels, info):
                W, H = p.frames[f].size
                out.append(dict(f=f, win=(a * bx[0], a * bx[1], W + a * (bx[2] - W), H + a * (bx[3] - H)), box=bx,
                                step=lab, p=pv))
            yield out
        for t in range(12):
            a = ease(min(1, (t + 1) / 6))
            yield [dict(f=f, win=bx, step=lab, p=(1 - a) * pv + a * nv) for f, k, bx, lab, pv, nv in info]
        for _ in range(6):
            yield [dict(f=f, box=bx, step=lab, p=nv) for f, k, bx, lab, pv, nv in info]
    for _ in range(24):                                                 # result
        yield [dict(f=p.t["looks"][-1][0], final=True, p=p.t["p"][2],
                    step=f"Prediction: {sci(p.names[p.t['pred']])} ({'correct' if p.t['pred'] == p.label else 'incorrect'})")
               for p in panels]


def compose(panels, states, gap=6):
    """Panels side by side on black."""
    w, h = panels[0].w, panels[0].h
    canvas = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, h), (0, 0, 0))
    for i, (p, st) in enumerate(zip(panels, states)):
        canvas.paste(p.render(st), (i * (w + gap), 0))
    return canvas


def write_mp4(frames_iter, size, out):
    W, H = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster = None
    for i, im in enumerate(frames_iter):
        if poster is None and i == 80:
            poster = im.copy()
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    proc.wait()
    return poster


def load_clip_frames(frames_df, clip_id, root):
    rows = frames_df[frames_df.clip_id == clip_id].sort_values("frame_idx")
    return [Image.open(root / p).convert("RGB") for p in rows.path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", default=str(paths.REPO / "docs" / "project_page" / "static" / "videos"))
    ap.add_argument("--compare", type=int, default=4, help="side-by-side videos")
    ap.add_argument("--teaser", type=int, default=3, help="clips in the teaser")
    ap.add_argument("--clips", default="", help="comma-separated clip ids, in the order to use them (default: dump order)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = pickle.load(open(args.proposed, "rb")), pickle.load(open(args.final, "rb"))
    cats = pd.read_csv(paths.DATA / "raw" / "iwildcam_v2.0" / "categories.csv")
    name_of = dict(zip(cats["y"], cats["name"]))
    names_a = {i: name_of[int(y)] for i, y in enumerate(A["classes"])}
    names_b = {i: name_of[int(y)] for i, y in enumerate(B["classes"])}
    frames_df = pd.read_csv(paths.processed("iwildcam") / "frames.csv")
    root = paths.DATA / "raw" / "iwildcam_v2.0"
    clips_a = {c.clip_id: c for c in A["clips"]}

    done = []
    for cb in B["clips"]:
        ca = clips_a.get(cb.clip_id)
        if ca is None:
            continue
        ta, tb = trajectory(ca, A["model"], A["rule"]), trajectory(cb, B["model"], B["rule"])
        if not (tb["pred"] == cb.label and ta["pred"] != ca.label):
            continue
        done.append((cb.clip_id, ca, cb, ta, tb))
    print(f"{len(done)} clips where the final eyes are right and the proposed ones are wrong")
    if args.clips:
        order = {cid: i for i, cid in enumerate(args.clips.split(","))}
        done = sorted((d for d in done if d[0] in order), key=lambda d: order[d[0]])

    pw, ph = 640, 360
    made = []
    for cid, ca, cb, ta, tb in done[: args.compare]:
        frames = load_clip_frames(frames_df, cid, root)
        truth = names_b[cb.label]
        pa = Panel("As proposed", ORANGE, ta, frames, names_a, ca.label, pw, ph)
        pb = Panel("Final", BLUE, tb, frames, names_b, cb.label, pw, ph)
        size = (2 * pw + 6, ph)
        poster = write_mp4((compose([pa, pb], st) for st in timeline([pa, pb], len(frames))), size,
                           out / f"compare_{cid}.mp4")
        poster.save(out / f"compare_{cid}.jpg", quality=88)
        print("wrote", out / f"compare_{cid}.mp4")
        made.append(f"compare_{cid}.mp4\t{sci(truth)}\t{len(frames)}")
    (out / "videos.txt").write_text("\n".join(made) + "\n")         # file, species, frames: the page shows these, in order

    def teaser_frames():                                            # the final method alone, larger
        for cid, ca, cb, ta, tb in done[: args.teaser]:
            frames = load_clip_frames(frames_df, cid, root)
            pb = Panel("Final method (DINOv2)", BLUE, tb, frames, names_b, cb.label, 1280, 720, scale=1.6)
            for st in timeline([pb], len(frames)):
                yield compose([pb], st)

    poster = write_mp4(teaser_frames(), (1280, 720), out / "teaser.mp4")
    poster.save(out / "teaser.jpg", quality=88)
    print("wrote", out / "teaser.mp4")


if __name__ == "__main__":
    main()
