"""Videos of the eyes at work, for the project page.

    python scripts/run_eval.py ... --seeds 0 --dump proposed.pkl --dump-clips ID,ID,...   (method as proposed)
    python scripts/run_eval.py ... --seeds 0 --dump final.pkl --dump-clips ID,ID,...      (final method flags)
    python scripts/make_videos.py --proposed proposed.pkl --final final.pkl --clips ID,ID,... \
        --out docs/project_page/static/videos

For each clip the controller is replayed step by step (glance, look 1, look 2) with the fitted
model, so the belief shown after every step is the one the method actually holds. Writes
compare_<clip>.mp4 (as proposed | final, side by side), teaser.mp4 (final method only) and a
poster image for each.

Layout: the footage is drawn at its stored size (448 px tall in the teaser) and nothing is written
on it; a step tracker sits above it, a filmstrip below it (faded frames are not glanced yet, boxes
mark where the looks went) and the belief over species next to it. The species rows are fixed for
the whole video, so only the bars move. Needs ffmpeg.
"""
import argparse
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyes import paths  # noqa: E402
from eyes.controller import run  # noqa: E402
from eyes.geometry import boxes  # noqa: E402

FPS = 24
BG, INK, MUTED, FAINT, TRACK = (255, 255, 255), (33, 37, 41), (108, 117, 125), (200, 205, 210), (233, 236, 239)
BLUE, ORANGE = (31, 119, 180), (255, 127, 14)
GOOD, BAD = (25, 135, 84), (200, 35, 51)
STEPS = ("Glance", "Look 1", "Look 2", "Prediction")
WHOLE = (0.0, 0.0, 1.0, 1.0)

FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"
FACES = dict(regular=0, bold=1, italic=2, medium=10)
_fonts = {}


def font(size, face="regular"):
    if (size, face) not in _fonts:
        try:
            _fonts[size, face] = ImageFont.truetype(FONT_FILE, size, index=FACES[face])
        except OSError:
            _fonts[size, face] = ImageFont.load_default(size)
    return _fonts[size, face]


def sci(name):
    """Binomial style: capitalise the genus ('urocyon cinereoargenteus' -> 'Urocyon cinereoargenteus')."""
    return name[:1].upper() + name[1:]


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * float(np.clip(t, 0, 1)))


def lerp(a, b, t):
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


# ------------------------------------------------------------------ what the method did
def trajectory(clip, model, rule):
    """Beliefs after the glance and after each of 2 looks, as the controller computes them."""
    o0, o1, o2 = (run(clip, model, rule, budget=b) for b in (0, 1, 2))
    entered = [clip.gate is None or bool(clip.gate[f, k - 1]) for f, k in o2.looks]
    return dict(p=[o0.p, o1.p, o2.p], looks=o2.looks, entered=entered, pred=o2.pred)


def species_rows(t, label, n=4):
    """The species listed for a panel: the true one, the predicted one and the most probable ones
    at any step, in the order of the glance's belief."""
    best = np.max(np.stack(t["p"]), axis=0)
    rows = list(dict.fromkeys([label, t["pred"]] + [int(k) for k in np.argsort(-best)]))[:n]
    return sorted(rows, key=lambda k: -t["p"][0][k])


class Panel:
    """One version of the method on one clip."""

    def __init__(self, name, colour, traj, frames, names, label):
        self.name, self.colour, self.t, self.frames, self.names, self.label = name, colour, traj, frames, names, label
        self.rows = species_rows(traj, label)
        self._thumbs = {}

    def thumb(self, i, size):
        if (i, size) not in self._thumbs:
            self._thumbs[i, size] = self.frames[i].resize(size, Image.LANCZOS)
        return self._thumbs[i, size]


@dataclass
class State:
    """What one panel shows in one video frame."""
    step: int                          # index into STEPS
    f: int                             # frame on screen
    seen: int                          # frames glanced so far
    looks: int = 0                     # looks taken so far
    win: tuple = WHOLE                 # part of the frame on screen, in frame coordinates
    box: tuple | None = None           # look box on the footage, in frame coordinates
    box_alpha: float = 1.0
    p: np.ndarray | None = None        # belief shown (None until the glance is done)
    zoomed: bool = False
    status: str = ""
    verdict: bool | None = None        # correct or not, once the prediction is shown


# ------------------------------------------------------------------ timing
def glance_hold(T):
    return int(np.clip(round(72 / T), 7, 20))         # video frames per clip frame during the glance


def poster_index(T):
    return glance_hold(T) * T + 40 + 20 + 14 + 24       # look 1, zoomed in


def timeline(panels, T):
    """One State per panel for every video frame; the panels move through the steps together."""
    for f in range(T):                                                   # glance at every frame
        for _ in range(glance_hold(T)):
            yield [State(0, f, f + 1, status=f"Glance: frame {f + 1} of {T}, whole frame at 224 px") for _ in panels]
    for i in range(40):                                                  # belief after the glance
        a = ease(i / 14)
        yield [State(0, T - 1, T, p=a * p.t["p"][0], status=f"After the glance: {T} frames") for p in panels]
    for j in range(2):                                                   # two looks
        def at(p, **kw):
            f, k = p.t["looks"][j]
            note = "" if p.t["entered"][j] else ", background: not used"
            return State(j + 1, f, T, looks=j, status=f"Look {j + 1}: frame {f + 1} of {T}, 2× crop{note}",
                         p=p.t["p"][j], **kw)
        box = [tuple(boxes()[p.t["looks"][j][1]]) for p in panels]
        for i in range(20):                                              # the chosen crop
            yield [at(p, box=b, box_alpha=ease(i / 8)) for p, b in zip(panels, box)]
        for i in range(14):                                              # zoom in
            a = ease((i + 1) / 14)
            yield [at(p, win=lerp(WHOLE, b, a), box=b, box_alpha=1 - a) for p, b in zip(panels, box)]
        for i in range(32):                                              # its vote enters the belief
            a = ease((i - 4) / 14)
            out = []
            for p, b in zip(panels, box):
                st = at(p, win=b, zoomed=True)
                st.p, st.looks = (1 - a) * p.t["p"][j] + a * p.t["p"][j + 1], j + 1
                out.append(st)
            yield out
        for i in range(12):                                              # zoom out
            a = ease((i + 1) / 12)
            out = []
            for p, b in zip(panels, box):
                st = at(p, win=lerp(b, WHOLE, a))
                st.p, st.looks = p.t["p"][j + 1], j + 1
                out.append(st)
            yield out
    for _ in range(56):                                                  # result
        yield [State(3, p.t["looks"][-1][0], T, looks=2, p=p.t["p"][2], verdict=p.t["pred"] == p.label,
                     status=f"Prediction: {sci(p.names[p.t['pred']])}") for p in panels]


# ------------------------------------------------------------------ drawing
def footage(p, st, size):
    """The frame (or the part of it being looked at) fitted into `size`; only the look box on top."""
    im = p.frames[st.f]
    W, H = im.size
    s = min(size[0] / W, size[1] / H)
    dw, dh = round(W * s), round(H * s)
    x0, y0, x1, y1 = st.win
    out = Image.new("RGBA", size, BG + (255,))
    ox, oy = (size[0] - dw) // 2, (size[1] - dh) // 2
    out.paste(im.resize((dw, dh), Image.LANCZOS, box=(x0 * W, y0 * H, x1 * W, y1 * H)), (ox, oy))
    over = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    if st.box is not None and st.box_alpha > 0:
        sx, sy = dw / (x1 - x0), dh / (y1 - y0)
        b = [ox + (st.box[0] - x0) * sx, oy + (st.box[1] - y0) * sy, ox + (st.box[2] - x0) * sx, oy + (st.box[3] - y0) * sy]
        b = [max(b[0], ox + 2), max(b[1], oy + 2), min(b[2], ox + dw - 3), min(b[3], oy + dh - 3)]
        a = int(255 * st.box_alpha)
        d.rectangle([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], outline=(255, 255, 255, a // 2), width=1)
        d.rectangle(b, outline=p.colour + (a,), width=4)
    if st.zoomed:                                                        # a small tag while zoomed in
        tag = "2× crop" if p.t["entered"][st.step - 1] else "2× crop: background"
        fnt = font(19, "medium")
        w = fnt.getlength(tag)
        d.rounded_rectangle([ox + 10, oy + 10, ox + 30 + w, oy + 42], radius=8, fill=(0, 0, 0, 150))
        d.text((ox + 20, oy + 26), tag, font=fnt, fill=(255, 255, 255, 255), anchor="lm")
    return Image.alpha_composite(out, over).convert("RGB")


def filmstrip(canvas, d, p, st, x, y, w, tw_max):
    """The clip's frames; faded = not glanced yet; outlined = on screen; boxes = looks taken."""
    T = len(p.frames)
    W, H = p.frames[0].size
    gap = 5
    tw = int(min(tw_max, (w - gap * (T - 1)) / T))
    th = round(tw * H / W)
    x += (w - T * tw - (T - 1) * gap) // 2                               # centred under the footage
    for i in range(T):
        im = p.thumb(i, (tw, th))
        if i >= st.seen:
            im = Image.blend(im, Image.new("RGB", im.size, BG), 0.72)
        xi = x + i * (tw + gap)
        canvas.paste(im, (xi, y))
        for f, k in p.t["looks"][:st.looks]:
            if f == i:
                b = boxes()[k]
                d.rectangle([xi + b[0] * tw, y + b[1] * th, xi + b[2] * tw - 1, y + b[3] * th - 1], outline=p.colour, width=2)
        if i == st.f and st.step < 3:
            d.rectangle([xi - 3, y - 3, xi + tw + 2, y + th + 2], outline=INK, width=2)
    return th


def tracker(d, step, cx, cy):
    """Glance -- Look 1 -- Look 2 -- Prediction, with the current step dark."""
    fnt, r, link, pad = font(21, "medium"), 6, 34, 9
    widths = [fnt.getlength(s) for s in STEPS]
    total = sum(2 * r + pad + w for w in widths) + (len(STEPS) - 1) * (link + 2 * pad)
    x = cx - total / 2
    for i, (s, w) in enumerate(zip(STEPS, widths)):
        col = INK if i <= step else FAINT
        d.ellipse([x, cy - r, x + 2 * r, cy + r], fill=col if i <= step else BG, outline=col, width=2)
        d.text((x + 2 * r + pad, cy), s, font=fnt, fill=INK if i == step else (MUTED if i < step else FAINT), anchor="lm")
        x += 2 * r + pad + w
        if i < len(STEPS) - 1:
            d.line([x + pad, cy, x + pad + link, cy], fill=INK if i < step else FAINT, width=2)
            x += link + 2 * pad


def mark(d, ok, x, cy, s=14, width=3):
    """A drawn check mark or cross (the font has neither), coloured for correct / wrong."""
    if ok:
        d.line([(x, cy), (x + s * 0.38, cy + s * 0.4), (x + s, cy - s * 0.45)], fill=GOOD, width=width, joint="curve")
    else:
        h = s * 0.42
        d.line([(x, cy - h), (x + 2 * h, cy + h)], fill=BAD, width=width)
        d.line([(x, cy + h), (x + 2 * h, cy - h)], fill=BAD, width=width)


def status_line(d, st, x, cy, size=21, anchor="lm", right=None):
    """The panel's status; at the end, a coloured mark and 'correct' / 'wrong' after the prediction."""
    fnt = font(size)
    text = st.status + ("" if st.verdict is None else ("  correct" if st.verdict else "  wrong"))
    if right is not None:
        x = right - fnt.getlength(text) - (22 if st.verdict is not None else 0)
    if st.verdict is None:
        d.text((x, cy), text, font=fnt, fill=MUTED, anchor="lm")
        return
    d.text((x, cy), st.status, font=fnt, fill=INK, anchor="lm")
    x += fnt.getlength(st.status) + 12
    mark(d, st.verdict, x, cy)
    d.text((x + 22, cy), "correct" if st.verdict else "wrong", font=font(size, "medium"),
           fill=GOOD if st.verdict else BAD, anchor="lm")


def beliefs(d, p, st, x, y, w, row_h, stacked=False):
    """Belief over the panel's species rows: name, bar, value. 'true' marks the true species."""
    nf, vf, tf = font(21, "italic"), font(20), font(14, "bold")
    pill = 12 + tf.getlength("TRUE")
    names = [sci(p.names[k]) for k in p.rows]
    name_w = max(nf.getlength(n) for n in names) + 10 + pill + 18
    for r, (k, name) in enumerate(zip(p.rows, names)):
        cy = y + r * row_h + row_h / 2
        if stacked:                                                       # name above the bar
            ny, by, bx, bw = cy - 12, cy + 13, x, w - 58
        else:                                                             # name, bar, value in one line
            ny, by, bx, bw = cy, cy, x + name_w, w - name_w - 58
        d.text((x, ny), name, font=nf, fill=INK, anchor="lm")
        if k == p.label:
            px = x + nf.getlength(name) + 10
            d.rounded_rectangle([px, ny - 10, px + pill, ny + 10], radius=5, fill=TRACK)
            d.text((px + 6, ny + 1), "TRUE", font=tf, fill=MUTED, anchor="lm")
        d.rounded_rectangle([bx, by - 7, bx + bw, by + 7], radius=7, fill=TRACK)
        v = None if st.p is None else float(st.p[k])
        if v is not None and v * bw >= 2:
            d.rounded_rectangle([bx, by - 7, bx + max(v * bw, 14), by + 7], radius=7, fill=p.colour)
        top = st.verdict is not None and k == p.t["pred"]
        d.text((bx + bw + 10, by), "–" if v is None else f"{v:.2f}", font=font(20, "bold") if top else vf,
               fill=INK if v is not None else FAINT, anchor="lm")


def swatch_title(d, p, x, cy, size=24):
    d.rounded_rectangle([x, cy - 7, x + 14, cy + 7], radius=3, fill=p.colour)
    d.text((x + 24, cy), p.name, font=font(size, "bold"), fill=INK, anchor="lm")


# ------------------------------------------------------------------ the two layouts
CMP_W, CMP_H, CMP_COL, CMP_FOOT, CMP_M, CMP_G = 1360, 720, 640, 360, 24, 32


def compare_frame(panels, states):
    """As proposed | final, side by side: header, footage, filmstrip, beliefs in each column."""
    canvas = Image.new("RGB", (CMP_W, CMP_H), BG)
    d = ImageDraw.Draw(canvas)
    tracker(d, states[0].step, CMP_W / 2, 38)
    for i, (p, st) in enumerate(zip(panels, states)):
        x = CMP_M + i * (CMP_COL + CMP_G)
        swatch_title(d, p, x, 96)
        status_line(d, st, 0, 96, size=20, right=x + CMP_COL)
        canvas.paste(footage(p, st, (CMP_COL, CMP_FOOT)), (x, 118))
        th = filmstrip(canvas, d, p, st, x, 118 + CMP_FOOT + 14, CMP_COL, 64)
        beliefs(d, p, st, x, 118 + CMP_FOOT + 14 + th + 14, CMP_COL, 36)
    return canvas


TSR_W, TSR_H, TSR_FOOT = 1280, 640, (796, 448)


def teaser_frame(p, st, encoder):
    """The final method alone: footage at its stored size, filmstrip below, beliefs and cost on the right."""
    canvas = Image.new("RGB", (TSR_W, TSR_H), BG)
    d = ImageDraw.Draw(canvas)
    tracker(d, st.step, TSR_W / 2, 40)
    canvas.paste(footage(p, st, TSR_FOOT), (32, 84))
    filmstrip(canvas, d, p, st, 32, 84 + TSR_FOOT[1] + 18, TSR_FOOT[0], 76)
    T = len(p.frames)
    x, w = 868, TSR_W - 868 - 32
    swatch_title(d, p, x, 100, size=26)
    d.text((x, 134), f"{encoder}, frozen", font=font(20), fill=MUTED, anchor="lm")
    status_line(d, st, x, 182, size=20)
    beliefs(d, p, st, x, 214, w, 62, stacked=True)
    d.text((x, 500), "encoder passes", font=font(19), fill=MUTED, anchor="lm")
    d.text((x, 540), str(st.seen + st.looks), font=font(38, "bold"), fill=INK, anchor="lm")
    d.text((x, 582), f"every crop of every frame: {10 * T}", font=font(19), fill=MUTED, anchor="lm")
    return canvas


# ------------------------------------------------------------------ output
def write_mp4(frames_iter, size, out, poster_at):
    W, H = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster = None
    for i, im in enumerate(frames_iter):
        if i == poster_at:
            poster = im.copy()
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {out}")
    return poster


def crossfade(a, b, n=8):
    for i in range(1, n + 1):
        yield Image.blend(a, b, i / (n + 1))


def load_clip_frames(frames_df, clip_id, root):
    rows = frames_df[frames_df.clip_id == clip_id].sort_values("frame_idx")
    return [Image.open(root / p).convert("RGB") for p in rows.path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", default=str(paths.REPO / "docs" / "project_page" / "static" / "videos"))
    ap.add_argument("--encoder", default="DINOv2", help="name shown in the teaser")
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

    made = []
    for cid, ca, cb, ta, tb in done[: args.compare]:
        frames = load_clip_frames(frames_df, cid, root)
        panels = [Panel("As proposed", ORANGE, ta, frames, names_a, ca.label),
                  Panel("Final method", BLUE, tb, frames, names_b, cb.label)]
        poster = write_mp4((compare_frame(panels, st) for st in timeline(panels, len(frames))), (CMP_W, CMP_H),
                           out / f"compare_{cid}.mp4", poster_index(len(frames)))
        poster.save(out / f"compare_{cid}.jpg", quality=90)
        print("wrote", out / f"compare_{cid}.mp4")
        made.append(f"compare_{cid}.mp4\t{sci(names_b[cb.label])}\t{len(frames)}")
    (out / "videos.txt").write_text("\n".join(made) + "\n")         # file, species, frames: the page shows these, in order

    def teaser_frames():                                            # the final method alone, clips joined by crossfades
        last = None
        for cid, ca, cb, ta, tb in done[: args.teaser]:
            frames = load_clip_frames(frames_df, cid, root)
            p = Panel("Final method", BLUE, tb, frames, names_b, cb.label)
            for i, st in enumerate(timeline([p], len(frames))):
                im = teaser_frame(p, st[0], args.encoder)
                if i == 0 and last is not None:
                    yield from crossfade(last, im)
                yield im
                last = im

    first_T = len(load_clip_frames(frames_df, done[0][0], root))
    poster = write_mp4(teaser_frames(), (TSR_W, TSR_H), out / "teaser.mp4", poster_index(first_T))
    poster.save(out / "teaser.jpg", quality=90)
    print("wrote", out / "teaser.mp4")


if __name__ == "__main__":
    main()
