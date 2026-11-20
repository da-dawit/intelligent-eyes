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
INK, INK2, MUTED, RULE, BG = (17, 17, 17), (85, 85, 85), (150, 150, 150), (225, 225, 225), (255, 255, 255)
BLUE, ORANGE, GREEN, RED = (42, 120, 214), (235, 104, 52), (20, 130, 60), (200, 50, 45)
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def font(size, style=""):
    name = {"": "Arial.ttf", "b": "Arial Bold.ttf", "i": "Arial Italic.ttf"}[style]
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def trajectory(clip, model, rule):
    """Beliefs after the glance and after each of 2 looks, as the controller computes them."""
    o0, o1, o2 = (run(clip, model, rule, budget=b) for b in (0, 1, 2))
    looks = o2.looks
    entered = [clip.gate is None or bool(clip.gate[f, k - 1]) for f, k in looks]
    return dict(p=[o0.p, o1.p, o2.p], looks=looks, entered=entered, pred=o2.pred)


class Panel:
    """One version of the eyes on one clip: draws the frame view and the belief bars."""

    def __init__(self, name, colour, traj, frames, names, label, w, frame_h, bars_h):
        self.name, self.colour, self.t, self.frames, self.names, self.label = name, colour, traj, frames, names, label
        self.w, self.fh, self.bh = w, frame_h, bars_h
        final = traj["p"][-1]
        top = list(np.argsort(-final)[:3])
        if label not in top:
            top = top[:2] + [label]
        self.top = top

    def view(self, f, window=None, box=None):
        im = self.frames[f]
        W, H = im.size
        x0, y0, x1, y1 = window if window is not None else (0, 0, W, H)
        v = im.crop((round(x0), round(y0), round(x1), round(y1))).resize((self.w, self.fh), Image.BICUBIC)
        d = ImageDraw.Draw(v)
        if window is None:
            for t in (0.25, 0.5, 0.75):
                d.line([(t * self.w, 0), (t * self.w, self.fh)], fill=(255, 255, 255), width=1)
                d.line([(0, t * self.fh), (self.w, t * self.fh)], fill=(255, 255, 255), width=1)
        if box is not None:
            sx, sy = self.w / (x1 - x0), self.fh / (y1 - y0)
            bx = [(box[0] - x0) * sx, (box[1] - y0) * sy, (box[2] - x0) * sx, (box[3] - y0) * sy]
            d.rectangle([bx[0] + 2, bx[1] + 2, bx[2] - 3, bx[3] - 3], outline=self.colour, width=5)
        return v

    def bars(self, p):
        img = Image.new("RGB", (self.w, self.bh), BG)
        d = ImageDraw.Draw(img)
        name_w, row = 230, self.bh // 4
        d.text((0, 2), "belief over species", font=font(15), fill=INK2)
        if p is None:
            return img
        for r, k in enumerate(self.top):
            y = 2 + (r + 1) * row
            sp = self.names[k] + ("  (true)" if k == self.label else "")
            d.text((0, y), sp, font=font(16, "i" if k != self.label else "b"), fill=INK)
            bw = (self.w - name_w - 60) * float(p[k])
            d.rectangle([name_w, y + 3, name_w + max(bw, 1), y + row - 10], fill=self.colour if k == self.label else (190, 190, 186))
            d.text((name_w + bw + 8, y + 1), f"{100 * float(p[k]):.0f}%", font=font(16), fill=INK)
        return img


def crop_px(k, size):
    W, H = size
    b = boxes()[k]
    return (b[0] * W, b[1] * H, b[2] * W, b[3] * H)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def timeline(panel_list, T):
    """Yield, per video frame, one (image, caption, belief) triple per panel."""
    n = len(panel_list)
    for f in range(T):                                                  # glance at every frame
        for _ in range(3):
            yield [(p.view(f), f"Glance: frame {f + 1} of {T} at 224 px", None) for p in panel_list]
    for _ in range(14):
        yield [(p.view(T - 1), "After the glance", p.t["p"][0]) for p in panel_list]
    for j in range(2):                                                  # two looks
        prev, nxt = [p.t["p"][j] for p in panel_list], [p.t["p"][j + 1] for p in panel_list]
        info = []
        for p in panel_list:
            f, k = p.t["looks"][j]
            info.append((f, k, crop_px(k, p.frames[f].size), p.t["entered"][j]))
        for _ in range(8):
            yield [(p.view(f, box=bx), f"Look {j + 1}: frame {f + 1}, crop {k}", pv)
                   for p, (f, k, bx, _), pv in zip(panel_list, info, prev)]
        for s in range(10):
            a = ease((s + 1) / 10)
            out = []
            for p, (f, k, bx, _), pv in zip(panel_list, info, prev):
                W, H = p.frames[f].size
                win = (a * bx[0], a * bx[1], W + a * (bx[2] - W), H + a * (bx[3] - H))
                out.append((p.view(f, window=win, box=bx), f"Look {j + 1}: zooming in 2x", pv))
            yield out
        for s in range(12):
            a = ease(min(1, (s + 1) / 6))
            out = []
            for p, (f, k, bx, ent), pv, nv in zip(panel_list, info, prev, nxt):
                cap = f"Look {j + 1}: frame {f + 1}, crop {k}" + ("" if ent else "  -  no animal, not counted")
                out.append((p.view(f, window=bx), cap, (1 - a) * pv + a * nv))
            yield out
        for _ in range(6):
            yield [(p.view(f, box=bx), f"Look {j + 1}: frame {f + 1}, crop {k}" + ("" if ent else "  -  no animal, not counted"), nv)
                   for p, (f, k, bx, ent), nv in zip(panel_list, info, nxt)]
    for _ in range(24):                                                 # verdict
        out = []
        for p in panel_list:
            ok = p.t["pred"] == p.label
            out.append((p.view(p.t["looks"][-1][0]), ("Correct: " if ok else "Wrong: ") + p.names[p.t["pred"]], p.t["p"][2]))
        yield out


def compose(panels, steps, title, width, height, frame_h, bars_h):
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    d.text((24, 16), title, font=font(22, "b"), fill=INK)
    gap = 28
    pw = panels[0].w
    for i, (p, (view, caption, belief)) in enumerate(zip(panels, steps)):
        x = 24 + i * (pw + gap)
        y = 56
        d.text((x, y), p.name, font=font(18, "b"), fill=p.colour)
        d.line([(x, y + 26), (x + pw, y + 26)], fill=p.colour, width=3)
        canvas.paste(view, (x, y + 34))
        cap_col = INK
        if caption.startswith("Correct"):
            cap_col = GREEN
        elif caption.startswith("Wrong"):
            cap_col = RED
        d.text((x, y + 42 + frame_h), caption, font=font(17, "b" if cap_col != INK else ""), fill=cap_col)
        canvas.paste(p.bars(belief), (x, y + 76 + frame_h))
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

    frame_h, bars_h, pw = 340, 132, 604
    made = []
    for cid, ca, cb, ta, tb in done[: args.compare]:
        frames = load_clip_frames(frames_df, cid, root)
        truth = names_b[cb.label]
        pa = Panel("As proposed", ORANGE, ta, frames, names_a, ca.label, pw, frame_h, bars_h)
        pb = Panel("Final", BLUE, tb, frames, names_b, cb.label, pw, frame_h, bars_h)
        size = (24 * 2 + 2 * pw + 28, 56 + 76 + frame_h + bars_h + 18)
        size = (size[0] + size[0] % 2, size[1] + size[1] % 2)
        steps = timeline([pa, pb], len(frames))
        imgs = (compose([pa, pb], s, f"Test clip, {len(frames)} frames: {truth}", size[0], size[1], frame_h, bars_h)
                for s in steps)
        poster = write_mp4(imgs, size, out / f"compare_{cid}.mp4")
        poster.save(out / f"compare_{cid}.jpg", quality=85)
        print("wrote", out / f"compare_{cid}.mp4")
        made.append(f"compare_{cid}.mp4")
    (out / "videos.txt").write_text("\n".join(made) + "\n")         # the page shows exactly these, in order

    # teaser: the final method alone, a few clips one after another, larger
    tw, th_frame, th_bars = 900, 506, 150
    size = (24 * 2 + tw, 56 + 76 + th_frame + th_bars + 18)
    size = (size[0] + size[0] % 2, size[1] + size[1] % 2)

    def teaser_frames():
        for cid, ca, cb, ta, tb in done[: args.teaser]:
            frames = load_clip_frames(frames_df, cid, root)
            pb = Panel("Intelligent eyes: glance at every frame, then 2 looks chosen by expected information gain",
                       BLUE, tb, frames, names_b, cb.label, tw, th_frame, th_bars)
            for s in timeline([pb], len(frames)):
                yield compose([pb], s, f"Test camera never seen in training  -  {names_b[cb.label]}", size[0], size[1],
                              th_frame, th_bars)

    poster = write_mp4(teaser_frames(), size, out / "teaser.mp4")
    poster.save(out / "teaser.jpg", quality=85)
    print("wrote", out / "teaser.mp4")


if __name__ == "__main__":
    main()
