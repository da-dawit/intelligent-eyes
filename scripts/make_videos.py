"""Videos of the eyes at work, for the project page.

    python scripts/run_eval.py ... --seeds 0 --dump proposed.pkl --dump-clips ID,ID,...   (method as proposed)
    python scripts/run_eval.py ... --seeds 0 --dump final_<enc>.pkl --dump-clips ID,...   (final method flags)

    python scripts/make_videos.py --proposed proposed.pkl --final final_dinov2.pkl --clips ID,ID,...
    python scripts/make_videos.py --baselines final_dinov2.pkl --baseline-clips ID,ID,...
    python scripts/make_videos.py --encoders clip=final_clip.pkl,siglip2=...,dinov2=...,dinov3=... --encoder-clips ID,...
    python scripts/make_videos.py --gallery final_dinov2.pkl --gallery-clips ID,ID,...

Every video replays the controller step by step with the fitted model, so each look and each belief
shown is the one the method actually computes. Four kinds, each listed (in order) in a text file the
page reads:
    compare_<clip>.mp4   as proposed | final method, side by side                 (videos.txt)
    teaser.mp4           the final method alone, larger
    baselines_<clip>.mp4 random | motion | attention | EIG looks on one clip      (baselines.txt)
    encoders_<clip>.mp4  the final method with each of the four encoders           (encoders.txt)
    gallery_<clip>.mp4   the final method with the stopping rule (0-4 looks)       (gallery.txt)

Layout: nothing is written on the footage; a step tracker sits above it, a filmstrip below it (faded
frames are not glanced yet, boxes mark where the looks went) and the belief over species under or
beside it. The species rows are fixed for a whole video, so only the bars move. Needs ffmpeg.
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
BLUE, ORANGE, GREEN, PURPLE, BROWN = (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189), (140, 86, 75)
GOOD, BAD = (25, 135, 84), (200, 35, 51)
WHOLE = (0.0, 0.0, 1.0, 1.0)
ENCODER_NAMES = dict(clip="CLIP", siglip2="SigLIP 2", dinov2="DINOv2", dinov3="DINOv3")
BASELINES = [("Random looks (B2)", "random", GREEN), ("Motion (B4)", "score:motion", PURPLE),     # name, rule, colour
             ("Attention (B3)", "score:attention", BROWN), ("EIG: final method", None, BLUE)]
SHORT = {"Random looks (B2)": "random", "Motion (B4)": "motion", "Attention (B3)": "attention", "EIG: final method": "EIG"}

FONT_FILE = "/System/Library/Fonts/HelveticaNeue.ttc"
FACES = dict(regular=0, bold=1, italic=2, medium=10)
_fonts = {}


def font(size, face="regular"):
    size = int(round(size))
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
def trajectory(clip, model, rule, budget=2, eps=0.0, seed=0):
    """Beliefs after the glance and after each look, as the controller computes them.

    With eps > 0 the controller may stop before the budget (the stopping rule); the looks it takes
    are the first ones it would take without stopping, so the beliefs come from shorter budgets.
    Random looks use the same seed in every call, so they are the same looks too."""
    final = run(clip, model, rule, budget=budget, eps=eps, rng=np.random.default_rng(seed))
    n = len(final.looks)
    p = [run(clip, model, rule, budget=b, rng=np.random.default_rng(seed)).p for b in range(n + 1)]
    entered = [clip.gate is None or bool(clip.gate[f, k - 1]) for f, k in final.looks]
    return dict(p=p, looks=final.looks, entered=entered, pred=final.pred, eig=final.best_eig,
                stopped=final.stopped, eps=eps)


def species_rows(t, label, n=4):
    """The species listed for a panel: the true one, the predicted one and the most probable ones
    at any step, in the order of the glance's belief."""
    best = np.max(np.stack(t["p"]), axis=0)
    rows = list(dict.fromkeys([label, t["pred"]] + [int(k) for k in np.argsort(-best)]))[:n]
    return sorted(rows, key=lambda k: -t["p"][0][k])


class Panel:
    """One version of the method on one clip."""

    def __init__(self, name, colour, traj, frames, names, label, rows=4):
        self.name, self.colour, self.t, self.frames, self.names, self.label = name, colour, traj, frames, names, label
        self.rows = species_rows(traj, label, rows)
        self._thumbs = {}

    @property
    def steps(self):
        return ["Glance"] + [f"Look {j + 1}" for j in range(len(self.t["looks"]))] + ["Prediction"]

    def thumb(self, i, size):
        if (i, size) not in self._thumbs:
            self._thumbs[i, size] = self.frames[i].resize(size, Image.LANCZOS)
        return self._thumbs[i, size]


@dataclass
class State:
    """What one panel shows in one video frame."""
    step: int                          # index into Panel.steps
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


def poster_index(T, n_looks=2):
    """A representative video frame: zoomed in on look 1, or the prediction if there is no look."""
    return glance_hold(T) * T + (40 + 20 + 14 + 24 if n_looks else 52)


def look_status(p, j, T):
    f = p.t["looks"][j][0]
    eig = p.t["eig"][j] if j < len(p.t["eig"]) else float("nan")
    text = f"Look {j + 1}: frame {f + 1} of {T}" + ("" if p.t["entered"][j] else ", background: not used")
    return text + ("" if np.isnan(eig) else f", EIG {eig:.2f}")        # narrow panels drop the last parts first


def timeline(panels, T):
    """One State per panel for every video frame; the panels move through the steps together."""
    n = len(panels[0].t["looks"])
    assert all(len(p.t["looks"]) == n for p in panels), "panels side by side take the same number of looks"
    for f in range(T):                                                   # glance at every frame
        for _ in range(glance_hold(T)):
            yield [State(0, f, f + 1, status=f"Glance: frame {f + 1} of {T}, whole frame at 224 px") for _ in panels]
    for i in range(40):                                                  # belief after the glance
        a = ease(i / 14)
        yield [State(0, T - 1, T, p=a * p.t["p"][0], status=f"After the glance: {T} frames") for p in panels]
    for j in range(n):                                                   # the looks
        def at(p, **kw):
            return State(j + 1, p.t["looks"][j][0], T, looks=j, status=look_status(p, j, T), p=p.t["p"][j], **kw)
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
    last = [p.t["looks"][-1][0] if n else T - 1 for p in panels]
    if any(p.t["stopped"] for p in panels):                              # the stopping rule fires
        for _ in range(36):
            yield [State(n, f, T, looks=n, p=p.t["p"][n],
                         status=f"Stop: best EIG {p.t['eig'][-1]:.2f} is below ε = {p.t['eps']:.2f}")
                   for p, f in zip(panels, last)]
    for _ in range(56):                                                  # result
        yield [State(n + 1, f, T, looks=n, p=p.t["p"][n], verdict=p.t["pred"] == p.label,
                     status=f"Prediction: {sci(p.names[p.t['pred']])}") for p, f in zip(panels, last)]


# ------------------------------------------------------------------ drawing
def footage(p, st, size, s=1.0):
    """The frame (or the part of it being looked at) fitted into `size`; only the look box on top."""
    im = p.frames[st.f]
    W, H = im.size
    sc = min(size[0] / W, size[1] / H)
    dw, dh = round(W * sc), round(H * sc)
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
        d.rectangle(b, outline=p.colour + (a,), width=max(3, round(4 * s)))
    if st.zoomed:                                                        # a small tag while zoomed in
        tag = "2× crop" if p.t["entered"][st.step - 1] else "2× crop: background"
        fnt = font(19 * s, "medium")
        w, pad = fnt.getlength(tag), 10 * s
        d.rounded_rectangle([ox + pad, oy + pad, ox + 3 * pad + w, oy + pad + 32 * s], radius=8 * s, fill=(0, 0, 0, 150))
        d.text((ox + 2 * pad, oy + pad + 16 * s), tag, font=fnt, fill=(255, 255, 255, 255), anchor="lm")
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
        if i == st.f and st.verdict is None:
            d.rectangle([xi - 3, y - 3, xi + tw + 2, y + th + 2], outline=INK, width=2)
    return th


def tracker(d, step, names, cx, cy, size=21, max_w=None):
    """Glance -- Look 1 -- ... -- Prediction, with the current step dark; shrinks to fit max_w."""
    while True:
        fnt, r, link, pad = font(size, "medium"), size * 0.29, size * 1.6, size * 0.43
        widths = [fnt.getlength(s) for s in names]
        total = sum(2 * r + pad + w for w in widths) + (len(names) - 1) * (link + 2 * pad)
        if max_w is None or total <= max_w or size <= 12:
            break
        size -= 1
    x = cx - total / 2
    for i, (s, w) in enumerate(zip(names, widths)):
        col = INK if i <= step else FAINT
        d.ellipse([x, cy - r, x + 2 * r, cy + r], fill=col if i <= step else BG, outline=col, width=2)
        d.text((x + 2 * r + pad, cy), s, font=fnt, fill=INK if i == step else (MUTED if i < step else FAINT), anchor="lm")
        x += 2 * r + pad + w
        if i < len(names) - 1:
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


def status_line(d, st, x, cy, size=21, right=None, room=None):
    """The panel's status; after the prediction, a coloured mark and 'correct' / 'wrong'.
    right: right-align at this x; room: the width available (the text is shortened to fit)."""
    fnt, bold = font(size), font(size, "medium")
    word = None if st.verdict is None else ("correct" if st.verdict else "wrong")
    tail = 0 if word is None else 12 + size + bold.getlength(word)
    text = st.status
    if room is not None:                                                 # too long: drop trailing parts, the prefix,
        while ", " in text and fnt.getlength(text) + tail > room:        # then the predicted name
            text = text.rsplit(", ", 1)[0]
        if fnt.getlength(text) + tail > room:
            text = text.replace("Prediction: ", "")
        if fnt.getlength(text) + tail > room and word is not None:
            text = ""
    if right is not None:
        x = right - fnt.getlength(text) - tail
    if word is None:
        d.text((x, cy), text, font=fnt, fill=MUTED, anchor="lm")
        return
    d.text((x, cy), text, font=fnt, fill=INK, anchor="lm")
    x += fnt.getlength(text) + (12 if text else 0)
    mark(d, st.verdict, x, cy, s=size * 0.67, width=max(2, round(size / 7)))
    d.text((x + size, cy), word, font=bold, fill=GOOD if st.verdict else BAD, anchor="lm")


def beliefs(d, p, st, x, y, w, row_h, stacked=False, s=1.0):
    """Belief over the panel's species rows: name, bar, value. 'TRUE' marks the true species."""
    nf, vf, tf = font(21 * s, "italic"), font(20 * s), font(14 * s, "bold")
    pill = 12 * s + tf.getlength("TRUE")
    names = [sci(p.names[k]) for k in p.rows]
    name_w = max(nf.getlength(n) for n in names) + 10 * s + pill + 18 * s
    bh = 7 * s                                                            # half the bar's thickness
    for r, (k, name) in enumerate(zip(p.rows, names)):
        cy = y + r * row_h + row_h / 2
        if stacked:                                                       # name above the bar
            ny, by, bx, bw = cy - 12 * s, cy + 13 * s, x, w - 58 * s
        else:                                                             # name, bar, value in one line
            ny, by, bx, bw = cy, cy, x + name_w, w - name_w - 58 * s
        d.text((x, ny), name, font=nf, fill=INK, anchor="lm")
        if k == p.label:
            px = x + nf.getlength(name) + 10 * s
            d.rounded_rectangle([px, ny - 10 * s, px + pill, ny + 10 * s], radius=5 * s, fill=TRACK)
            d.text((px + 6 * s, ny + s), "TRUE", font=tf, fill=MUTED, anchor="lm")
        d.rounded_rectangle([bx, by - bh, bx + bw, by + bh], radius=bh, fill=TRACK)
        v = None if st.p is None else float(st.p[k])
        if v is not None and v * bw >= 2:
            d.rounded_rectangle([bx, by - bh, bx + max(v * bw, 2 * bh), by + bh], radius=bh, fill=p.colour)
        top = st.verdict is not None and k == p.t["pred"]
        d.text((bx + bw + 10 * s, by), "–" if v is None else f"{v:.2f}", font=font(20 * s, "bold") if top else vf,
               fill=INK if v is not None else FAINT, anchor="lm")


def swatch_title(d, p, x, cy, size=24):
    q = size * 0.3
    d.rounded_rectangle([x, cy - q, x + 2 * q, cy + q], radius=3, fill=p.colour)
    d.text((x + 2 * q + 10, cy), p.name, font=font(size, "bold"), fill=INK, anchor="lm")
    return 2 * q + 10 + font(size, "bold").getlength(p.name)


def cell(canvas, d, p, st, x, y, w, foot_h, s=1.0, strip=True, row_h=36, title_w=None):
    """One panel: name and status, footage, filmstrip, beliefs. Returns the height used.
    title_w: the widest name among the panels, so every panel shortens its status the same way."""
    y0 = y
    name_w = swatch_title(d, p, x, y + 14 * s, size=24 * s)
    status_line(d, st, 0, y + 14 * s, size=20 * s, right=x + w, room=w - (title_w or name_w) - 24 * s)
    y += 36 * s
    canvas.paste(footage(p, st, (w, foot_h), s), (x, round(y)))
    y += foot_h + 14 * s
    if strip:
        y += filmstrip(canvas, d, p, st, x, round(y), w, 64 * s) + 14 * s
    beliefs(d, p, st, x, y, w, row_h * s, s=s)
    y += len(p.rows) * row_h * s
    return y - y0


# ------------------------------------------------------------------ the layouts
CMP_SIZE, GRID_SIZE, TSR_SIZE, CARD_SIZE = (1360, 720), (1280, 1232), (1280, 640), (960, 820)


def compare_frame(panels, states):
    """As proposed | final, side by side."""
    canvas = Image.new("RGB", CMP_SIZE, BG)
    d = ImageDraw.Draw(canvas)
    tracker(d, states[0].step, panels[0].steps, CMP_SIZE[0] / 2, 38)
    for i, (p, st) in enumerate(zip(panels, states)):
        cell(canvas, d, p, st, 24 + i * (640 + 32), 78, 640, 360)
    return canvas


def grid_frame(panels, states):
    """Four panels in a 2 x 2 grid (baselines or encoders) on one clip."""
    canvas = Image.new("RGB", GRID_SIZE, BG)
    d = ImageDraw.Draw(canvas)
    tracker(d, states[0].step, panels[0].steps, GRID_SIZE[0] / 2, 36)
    w, fh, gx, pitch = 604, 340, 24, 584                                 # pitch: one row of panels
    title_w = max(24 * 0.6 + 10 + font(24, "bold").getlength(p.name) for p in panels)
    for i, (p, st) in enumerate(zip(panels, states)):
        r, c = divmod(i, 2)
        cell(canvas, d, p, st, 24 + c * (w + gx), 72 + r * pitch, w, fh, row_h=34, title_w=title_w)
    return canvas


def teaser_frame(p, st, encoder):
    """The final method alone: footage at its stored size, filmstrip below, beliefs and cost on the right."""
    canvas = Image.new("RGB", TSR_SIZE, BG)
    d = ImageDraw.Draw(canvas)
    tracker(d, st.step, p.steps, TSR_SIZE[0] / 2, 40)
    canvas.paste(footage(p, st, (796, 448)), (32, 84))
    filmstrip(canvas, d, p, st, 32, 84 + 448 + 18, 796, 76)
    T = len(p.frames)
    x, w = 868, TSR_SIZE[0] - 868 - 32
    swatch_title(d, p, x, 100, size=26)
    d.text((x, 134), f"{encoder}, frozen", font=font(20), fill=MUTED, anchor="lm")
    status_line(d, st, x, 182, size=20, room=w)
    beliefs(d, p, st, x, 214, w, 62, stacked=True)
    d.text((x, 500), "encoder passes", font=font(19), fill=MUTED, anchor="lm")
    d.text((x, 540), str(st.seen + st.looks), font=font(38, "bold"), fill=INK, anchor="lm")
    d.text((x, 582), f"every crop of every frame: {10 * T}", font=font(19), fill=MUTED, anchor="lm")
    return canvas


def card_frame(p, st):
    """A gallery card, made to be shown at half the page width: large type, footage, three species."""
    canvas = Image.new("RGB", CARD_SIZE, BG)
    d = ImageDraw.Draw(canvas)
    s, x, w = 1.45, 24, CARD_SIZE[0] - 48
    tracker(d, st.step, p.steps, CARD_SIZE[0] / 2, 34, size=28, max_w=w)
    status_line(d, st, x, 84, size=28, room=w)
    canvas.paste(footage(p, st, (w, 513), s), (x, 112))
    beliefs(d, p, st, x, 112 + 513 + 22, w, 50, s=s)
    d.text((CARD_SIZE[0] - x, 84), f"passes {st.seen + st.looks}", font=font(26), fill=MUTED, anchor="rm")
    return canvas


# ------------------------------------------------------------------ output
def write_mp4(frames_iter, size, out, poster_at, crf=20):
    W, H = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster = None
    for i, im in enumerate(frames_iter):
        if i == poster_at or poster is None:
            poster = im.copy()
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {out}")
    poster.save(out.with_suffix(".jpg"), quality=90)
    print("wrote", out)


def crossfade(a, b, n=8):
    for i in range(1, n + 1):
        yield Image.blend(a, b, i / (n + 1))


class Data:
    """Species names and clip frames."""

    def __init__(self):
        cats = pd.read_csv(paths.DATA / "raw" / "iwildcam_v2.0" / "categories.csv")
        self.name_of = dict(zip(cats["y"], cats["name"]))
        self.frames_df = pd.read_csv(paths.processed("iwildcam") / "frames.csv")
        self.root = paths.DATA / "raw" / "iwildcam_v2.0"

    def names(self, dump):
        return {i: self.name_of[int(y)] for i, y in enumerate(dump["classes"])}

    def frames(self, clip_id):
        rows = self.frames_df[self.frames_df.clip_id == clip_id].sort_values("frame_idx")
        return [Image.open(self.root / p).convert("RGB") for p in rows.path]


def ids(arg):
    return [c for c in arg.split(",") if c]


def make_compare(args, data, out):
    """As proposed | final on clips where the final eyes are right and the proposed ones wrong; teaser."""
    A, B = pickle.load(open(args.proposed, "rb")), pickle.load(open(args.final, "rb"))
    names_a, names_b = data.names(A), data.names(B)
    clips_a = {c.clip_id: c for c in A["clips"]}
    done = []
    for cb in B["clips"]:
        ca = clips_a.get(cb.clip_id)
        if ca is None:
            continue
        ta, tb = trajectory(ca, A["model"], A["rule"]), trajectory(cb, B["model"], B["rule"])
        if tb["pred"] == cb.label and ta["pred"] != ca.label:
            done.append((cb.clip_id, ca, cb, ta, tb))
    print(f"{len(done)} clips where the final eyes are right and the proposed ones are wrong")
    if args.clips:
        order = {cid: i for i, cid in enumerate(ids(args.clips))}
        done = sorted((d for d in done if d[0] in order), key=lambda d: order[d[0]])
    made = []
    for cid, ca, cb, ta, tb in done[: args.compare]:
        frames = data.frames(cid)
        panels = [Panel("As proposed", ORANGE, ta, frames, names_a, ca.label),
                  Panel("Final method", BLUE, tb, frames, names_b, cb.label)]
        write_mp4((compare_frame(panels, st) for st in timeline(panels, len(frames))), CMP_SIZE,
                  out / f"compare_{cid}.mp4", poster_index(len(frames)))
        made.append(f"compare_{cid}.mp4\t{sci(names_b[cb.label])}\t{len(frames)}")
    (out / "videos.txt").write_text("\n".join(made) + "\n")          # file, species, frames: the page shows these

    def teaser_frames():                                             # the final method alone, joined by crossfades
        last = None
        for cid, ca, cb, ta, tb in done[: args.teaser]:
            p = Panel("Final method", BLUE, tb, data.frames(cid), names_b, cb.label)
            for i, st in enumerate(timeline([p], len(p.frames))):
                im = teaser_frame(p, st[0], args.encoder)
                if i == 0 and last is not None:
                    yield from crossfade(last, im)
                yield im
                last = im
    write_mp4(teaser_frames(), TSR_SIZE, out / "teaser.mp4", poster_index(len(data.frames(done[0][0]))))


def make_baselines(args, data, out):
    """Random, motion, attention and EIG looks on the same clip, same fitted model, 2 looks each."""
    D = pickle.load(open(args.baselines, "rb"))
    names, clips = data.names(D), {c.clip_id: c for c in D["clips"]}
    made = []
    for cid in ids(args.baseline_clips):
        c, frames = clips[cid], data.frames(cid)
        panels = [Panel(name, col, trajectory(c, D["model"], rule or D["rule"]), frames, names, c.label, rows=3)
                  for name, rule, col in BASELINES]
        write_mp4((grid_frame(panels, st) for st in timeline(panels, len(frames))), GRID_SIZE,
                  out / f"baselines_{cid}.mp4", poster_index(len(frames)), crf=22)
        right = ",".join("1" if p.t["pred"] == c.label else "0" for p in panels)
        made.append(f"baselines_{cid}.mp4\t{sci(names[c.label])}\t{len(frames)}\t{right}\t"
                    + ",".join(SHORT[p.name] for p in panels))
    (out / "baselines.txt").write_text("\n".join(made) + "\n")  # file, species, frames, correct per panel, panels


def make_encoders(args, data, out):
    """The final method with each encoder on the same clip (each encoder has its own fitted model)."""
    dumps = {k: pickle.load(open(v, "rb")) for k, v in (kv.split("=") for kv in args.encoders.split(","))}
    made = []
    for cid in ids(args.encoder_clips):
        frames, panels = data.frames(cid), []
        for enc, D in dumps.items():
            c = next(c for c in D["clips"] if c.clip_id == cid)
            panels.append(Panel(ENCODER_NAMES[enc], BLUE, trajectory(c, D["model"], D["rule"]), frames,
                                data.names(D), c.label, rows=3))
        write_mp4((grid_frame(panels, st) for st in timeline(panels, len(frames))), GRID_SIZE,
                  out / f"encoders_{cid}.mp4", poster_index(len(frames)), crf=22)
        right = ",".join("1" if p.t["pred"] == p.label else "0" for p in panels)
        made.append(f"encoders_{cid}.mp4\t{sci(panels[0].names[panels[0].label])}\t{len(frames)}\t{right}\t"
                    + ",".join(p.name for p in panels))
    (out / "encoders.txt").write_text("\n".join(made) + "\n")   # file, species, frames, correct per panel, panels


def make_gallery(args, data, out):
    """The final method with the stopping rule (up to 4 looks), one card per clip."""
    D = pickle.load(open(args.gallery, "rb"))
    names, clips = data.names(D), {c.clip_id: c for c in D["clips"]}
    made = []
    for cid in ids(args.gallery_clips):
        c, frames = clips[cid], data.frames(cid)
        t = trajectory(c, D["model"], D["rule"], budget=4, eps=D["eps"])
        p = Panel("Final method", BLUE, t, frames, names, c.label, rows=3)
        write_mp4((card_frame(p, st[0]) for st in timeline([p], len(frames))), CARD_SIZE,
                  out / f"gallery_{cid}.mp4", poster_index(len(frames), len(t["looks"])), crf=22)
        made.append(f"gallery_{cid}.mp4\t{sci(names[c.label])}\t{len(frames)}\t{len(t['looks'])}\t"
                    f"{int(t['pred'] == c.label)}\t{sci(names[t['pred']])}")
    (out / "gallery.txt").write_text("\n".join(made) + "\n")  # file, species, frames, looks, correct, predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.REPO / "docs" / "project_page" / "static" / "videos"))
    ap.add_argument("--proposed", default="", help="dump of the method as proposed (with --final: compare + teaser)")
    ap.add_argument("--final", default="", help="dump of the final method")
    ap.add_argument("--clips", default="", help="comma-separated clip ids for the side-by-side videos, in order")
    ap.add_argument("--encoder", default="DINOv2", help="name shown in the teaser")
    ap.add_argument("--compare", type=int, default=4, help="side-by-side videos")
    ap.add_argument("--teaser", type=int, default=3, help="clips in the teaser")
    ap.add_argument("--baselines", default="", help="dump of the final method: baseline grid videos")
    ap.add_argument("--baseline-clips", default="")
    ap.add_argument("--encoders", default="", help="enc=dump,... (final method): encoder grid videos")
    ap.add_argument("--encoder-clips", default="")
    ap.add_argument("--gallery", default="", help="dump of the final method: stopping-rule cards")
    ap.add_argument("--gallery-clips", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data()
    if args.proposed and args.final:
        make_compare(args, data, out)
    if args.baselines:
        make_baselines(args, data, out)
    if args.encoders:
        make_encoders(args, data, out)
    if args.gallery:
        make_gallery(args, data, out)


if __name__ == "__main__":
    main()
