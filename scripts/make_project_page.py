"""Build the project page from the results: docs/project_page/index.html (+ a few example frames).

    python scripts/make_project_page.py

Self-contained: no internet needed to view it. It reads results/iwildcam/<run>/seed*/clips.csv.gz
for the method as proposed (<encoder>) and the final method (<encoder>-boxes-preview-gated-tau),
and draws a few test clips with the looks each version chose. MegaDetector boxes are drawn
only for the reader; the method never sees them at test time.
"""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import heldout_cameras  # noqa: E402

from eyes import paths  # noqa: E402
from eyes.boxes import crop_labels, load_detections  # noqa: E402
from eyes.geometry import boxes  # noqa: E402

ENCODERS = [("clip", "CLIP"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3"), ("siglip2", "SigLIP 2")]
FINAL = "-boxes-preview-gated-tau"
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)


def load(run):
    files = sorted((paths.RESULTS / "iwildcam" / run).glob("seed*/clips.csv.gz"))
    return pd.concat([pd.read_csv(f).assign(seed=f.parent.name) for f in files])


def acc(r, m):
    x = r[r.method == m]
    return None if x.empty else round(100 * float(x.correct.mean()), 1)


def numbers():
    out = {}
    for enc, name in ENCODERS:
        first, final = load(enc), load(enc + FINAL)
        passes = lambda r, m: float(r[r.method == m].passes.mean())
        out[enc] = dict(
            name=name,
            first=dict(glance=acc(first, "glance"), eyes=[acc(first, f"eyes_b{b}") for b in (1, 2, 3, 4)],
                       random=[acc(first, f"random_b{b}") for b in (1, 2, 3, 4)], all9=acc(first, "all9")),
            final=dict(glance=acc(final, "glance"), eyes=[acc(final, f"eyesmix_b{b}") for b in (1, 2, 3, 4)],
                       random=[acc(final, f"random_b{b}") for b in (1, 2, 3, 4)], all9=acc(final, "all9"),
                       stop=acc(final, "eyesmix_stop"),
                       stop_looks=round(passes(final, "eyesmix_stop") - passes(final, "glance"), 2),
                       motion=acc(final, "motion_b2"), attention=acc(final, "attention_b2"),
                       hires=acc(final, "hires"), best_pair=acc(final, "best_pair")))
    return out


def look_hits(frames, encoder="dinov2"):
    """Share of 2 looks (all seeds) that land on a crop holding the animal, among crops that clearly
    hold it or clearly don't (eyes/boxes.py: crop_labels); boxes are used only to measure this."""
    content = crop_labels(frames, load_detections(paths.DATA / "raw" / "iwildcam_mdv4" / "iwildcam2022_mdv4_detections.json"))
    rows_of = {cid: np.sort(idx) for cid, idx in frames.groupby("clip_id").indices.items()}

    def share(r, m):
        r = r[(r.method == m) & (r.draw == 0)]
        hit = tot = 0
        for cid, looks in zip(r.clip_id, r.looks.fillna("")):
            for lk in filter(None, looks.split(";")):
                f, k = map(int, lk.split("-"))
                c = content[rows_of[cid][f], k - 1]
                if c >= 0:
                    tot += 1
                    hit += c == 1
        return 100 * hit / max(tot, 1)

    first, final = load(encoder), load(encoder + FINAL)
    c = content[frames.index[frames.split == "test"].values]
    return dict(chance=100 * (c == 1).sum() / (c >= 0).sum(), random=share(final, "random_b2"),
                eig_first=share(first, "eyes_b2"), motion=share(final, "motion_b2"),
                attention=share(final, "attention_b2"), eig_final=share(final, "eyesmix_b2"))


def class_names(frames, seed):
    """Class index -> species name, exactly as run_eval numbers the classes for this seed."""
    cats = pd.read_csv(paths.DATA / "raw" / "iwildcam_v2.0" / "categories.csv")
    name_of = dict(zip(cats["y"], cats["name"]))
    train = frames[frames.split == "train"]
    held = heldout_cameras(train, 0.10, np.random.default_rng(seed))
    classes = np.sort(train[~train.camera.isin(held)].y_clip.unique())
    return {i: name_of.get(int(y), str(y)) for i, y in enumerate(classes)}


def draw_look(frame_path, look_k, det, colour, width=300):
    """One frame, the 3x3 crop grid, the chosen crop, and (dashed) the detector's animal box."""
    im = Image.open(frame_path).convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)))
    W, H = im.size
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for t in (0.25, 0.5, 0.75):
        d.line([(t * W, 0), (t * W, H)], fill=(255, 255, 255, 70), width=1)
        d.line([(0, t * H), (W, t * H)], fill=(255, 255, 255, 70), width=1)
    for conf, x, y, w, h in det:
        if conf < 0.5:
            continue
        x0, y0, x1, y1 = x * W, y * H, (x + w) * W, (y + h) * H
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // 6), 1)
            for i in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                d.line([p, q], fill=(255, 255, 255, 230), width=2)
    bx = boxes()[look_k]
    d.rectangle([bx[0] * W + 2, bx[1] * H + 2, bx[2] * W - 3, bx[3] * H - 3], outline=colour + (255,), width=4)
    return Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")


def examples(frames, n, seed, encoder, out_dir):
    names = class_names(frames, seed)
    det = load_detections(paths.DATA / "raw" / "iwildcam_mdv4" / "iwildcam2022_mdv4_detections.json")
    first = pd.read_csv(paths.RESULTS / "iwildcam" / encoder / f"seed{seed}" / "clips.csv.gz")
    final = pd.read_csv(paths.RESULTS / "iwildcam" / (encoder + FINAL) / f"seed{seed}" / "clips.csv.gz")
    a = first[first.method == "eyes_b2"].set_index("clip_id")
    g = final[final.method == "glance"].set_index("clip_id")
    b = final[final.method == "eyesmix_b2"].set_index("clip_id")
    cand = b.index[b.correct & ~a.loc[b.index].correct & ~g.loc[b.index].correct & (b["T"] >= 2)]
    rows_of = {cid: np.sort(idx) for cid, idx in frames.groupby("clip_id").indices.items()}
    rng = np.random.default_rng(7)
    picked, seen = [], set()
    for cid in rng.permutation(cand):
        lab = int(b.loc[cid, "label"])
        if lab in seen:
            continue
        looks = {m: [tuple(map(int, s.split("-"))) for s in r.loc[cid, "looks"].split(";")] for m, r in (("a", a), ("b", b))}
        files = {m: [frames.path.values[rows_of[cid][f]].split("/")[-1] for f, _ in looks[m]] for m in looks}
        if not all(det.get(f) for f in files["b"]):
            continue
        seen.add(lab)
        cards = {}
        for m, colour in (("a", ORANGE), ("b", BLUE)):
            imgs = []
            for j, ((f, k), fname) in enumerate(zip(looks[m], files[m])):
                im = draw_look(paths.DATA / "raw" / "iwildcam_v2.0" / "train" / fname, k, det.get(fname, []), colour)
                rel = f"assets/{cid}_{m}{j}.jpg"
                im.save(out_dir / rel, quality=82)
                imgs.append(dict(src=rel, frame=f + 1, crop=k))
            cards[m] = imgs
        picked.append(dict(clip=cid, T=int(b.loc[cid, "T"]), truth=names[lab],
                           glance=names[int(g.loc[cid, "pred"])], glance_conf=float(g.loc[cid, "conf"]),
                           proposed=names[int(a.loc[cid, "pred"])], proposed_conf=float(a.loc[cid, "conf"]),
                           final=names[int(b.loc[cid, "pred"])], final_conf=float(b.loc[cid, "conf"]), **cards))
        if len(picked) == n:
            break
    return picked


def example_html(ex):
    def strip(imgs):
        return "".join(f'<figure><img src="{i["src"]}" alt="frame {i["frame"]}, crop {i["crop"]}" loading="lazy">'
                       f'<figcaption>frame {i["frame"]}, crop {i["crop"]}</figcaption></figure>' for i in imgs)
    e = {k: html.escape(str(v)) for k, v in ex.items() if isinstance(v, str)}
    return f"""
<article class="ex">
  <header><h3><i>{e['truth']}</i></h3><span class="muted">test clip of {ex['T']} frames · glance alone said
  <i>{e['glance']}</i> ({100 * ex['glance_conf']:.0f}%)</span></header>
  <div class="pair">
    <div><div class="tag t-orange">As proposed</div><div class="strip">{strip(ex['a'])}</div>
      <p class="verdict bad">✗ <i>{e['proposed']}</i> ({100 * ex['proposed_conf']:.0f}%)</p></div>
    <div><div class="tag t-blue">Final</div><div class="strip">{strip(ex['b'])}</div>
      <p class="verdict good">✓ <i>{e['final']}</i> ({100 * ex['final_conf']:.0f}%)</p></div>
  </div>
</article>"""


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Eyes</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink2: #52514e; --muted: #8a8984;
  --line: #e6e5e1; --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --neutral: #b9b8b2;
  --good: #0a7d0a; --bad: #c0392b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19; --card: #232321; --ink: #f0efec; --ink2: #c3c2b7; --muted: #8f8e88;
    --line: #3a3a37; --blue: #3987e5; --orange: #d95926; --aqua: #199e70; --neutral: #5f5e5a;
    --good: #4cc24c; --bad: #e66767;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 2.1rem; line-height: 1.15; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 1.3rem; margin: 44px 0 10px; }
h3 { font-size: 1.05rem; margin: 0; }
p { margin: 6px 0 10px; }
.sub { color: var(--ink2); font-size: 1.08rem; margin: 0 0 4px; }
.muted { color: var(--ink2); font-size: 0.9rem; }
a { color: var(--blue); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.stat b { display: block; font-size: 1.6rem; line-height: 1.2; }
.stat span { color: var(--ink2); font-size: 0.9rem; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; padding: 0; list-style: none; counter-reset: s; }
.steps li { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; counter-increment: s; }
.steps li::before { content: counter(s); display: inline-grid; place-items: center; width: 22px; height: 22px; margin-right: 6px;
  border-radius: 50%; background: var(--blue); color: #fff; font-size: 0.8rem; font-weight: 700; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-bottom: 10px; }
.seg button { font: inherit; font-size: 0.9rem; border: 0; background: transparent; color: var(--ink2); padding: 6px 12px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--blue); color: #fff; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.85rem; color: var(--ink2); margin: 4px 0 0; }
.legend i { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.chart-wrap { position: relative; }
svg text { fill: var(--ink2); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 8px; font-size: 0.82rem; box-shadow: 0 2px 8px rgba(0,0,0,.12); opacity: 0; transition: opacity .1s; white-space: nowrap; }
.grid2 { display: grid; grid-template-columns: 1.25fr 1fr; gap: 14px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
.bars { display: grid; gap: 8px; margin-top: 8px; }
.bar { display: grid; grid-template-columns: 130px 1fr 42px; align-items: center; gap: 8px; font-size: 0.88rem; }
.bar .track { position: relative; height: 16px; background: transparent; }
.bar .fill { height: 100%; border-radius: 0 4px 4px 0; }
.bar .chance { position: absolute; top: -4px; bottom: -4px; border-left: 1.5px dashed var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--ink2); font-weight: 600; font-size: 0.82rem; vertical-align: bottom; }
td.best { font-weight: 700; color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
.scroll { overflow-x: auto; }
.ex { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.ex header { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; margin-bottom: 10px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } }
.strip { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.strip figure { margin: 0; }
.strip img { width: 100%; height: auto; display: block; border-radius: 6px; }
.strip figcaption { font-size: 0.75rem; color: var(--ink2); margin-top: 2px; }
.tag { display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; color: #fff; margin-bottom: 6px; }
.t-orange { background: var(--orange); } .t-blue { background: var(--blue); }
.verdict { margin: 6px 0 0; font-weight: 600; }
.verdict.good { color: var(--good); } .verdict.bad { color: var(--bad); }
.fixes { padding-left: 1.2em; } .fixes li { margin: 6px 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; overflow-x: auto; }
footer { margin-top: 48px; color: var(--ink2); font-size: 0.85rem; }
</style>
</head>
<body>
<main>
<h1>Intelligent Eyes</h1>
<p class="sub">Teaching frozen ViT encoders where, and when, to look in video.</p>
<p class="muted">Dawit Chun · DA411 Computational Mathematics, final project (Option A) ·
<a href="../proposal.pdf">proposal</a> · <a href="../derivations.pdf">derivations</a></p>

<div class="stats" id="stats"></div>

<h2>How the eyes work</h2>
<ol class="steps">
  <li><b>Glance</b> at every frame of a camera-trap clip once, at 224&nbsp;px, with a frozen encoder.</li>
  <li><b>Predict</b> what each of the 9T possible 2× zoomed crops would say, from the glance alone.</li>
  <li><b>Look</b> where the expected information gain (EIG) is highest; stop when it falls below ε.</li>
  <li><b>Combine</b> every vote with Bayes' rule, discounting repeated evidence through a fitted correlation R.</li>
</ol>

<h2>Accuracy as looks are added</h2>
<div class="grid2">
  <div class="panel">
    <div class="seg" role="group" aria-label="encoder" id="enc"></div>
    <div class="chart-wrap"><svg id="chart" viewBox="0 0 560 320" width="100%" role="img" aria-label="accuracy versus looks"></svg>
      <div class="tip" id="tip"></div></div>
    <div class="legend">
      <span><i style="background:var(--blue)"></i>eyes, final</span>
      <span><i style="background:var(--aqua)"></i>random looks, final</span>
      <span><i style="background:var(--orange)"></i>eyes, as proposed</span>
      <span><i style="border-top:2px dashed var(--muted);height:0"></i>every crop (10T passes)</span>
    </div>
  </div>
  <div class="panel">
    <h3>Where 2 looks land (DINOv2)</h3>
    <p class="muted">Share of looks on a crop that holds the animal. Detector boxes are used only to measure this.</p>
    <div class="bars" id="bars"></div>
  </div>
</div>

<h2>All methods, all encoders</h2>
<p class="muted">iWildCam test clips from 48 cameras never seen in training (3,930 clips, 3 seeds). Clip accuracy (%); T = frames per clip; looks are 2× crops. Underlined: best method at about T+2 passes.</p>
<div class="panel scroll"><table id="table"></table></div>

<h2>Side by side: the same clip, two versions of the eyes</h2>
<p class="muted">Test clips where glancing alone was wrong. Each version took 2 looks (coloured squares). The dashed white box
is the detector's animal box, drawn for you; the eyes never see it.</p>
__EXAMPLES__

<h2>What went wrong first, and what fixed it</h2>
<p>As proposed, every look <em>lowered</em> accuracy. The cause: each crop was trained with the clip's species,
although two thirds of crops show only background, so the crop classifier learned "this scenery means this species";
and Bayes' rule then read a background crop's empty vote as evidence against the glance. Four changes, each fitted on
held-out cameras:</p>
<ol class="fixes">
  <li><b>Crops get an <i>empty</i> class</b> (labels from animal boxes on training cameras), and the vote becomes
      log(q(y|c)/π(y) + q(empty|c)): about 0 for background, the usual log-ratio for the animal.</li>
  <li><b>Gated fusion:</b> a look whose crop shows no animal costs its pass but stays out of the combination.</li>
  <li><b>A shared crop error</b> τ in R: a clip's crops share one error, so many crops count as about one extra frame.</li>
  <li><b>Two-branch EIG:</b> EIG = P(animal) · EIG(animal branch), with P(animal) read from the glance's own tokens.</li>
</ol>

<h2>Run it</h2>
<pre>uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest                     # 43 tests, no data needed
python scripts/synthetic_study.py              # the method on synthetic clips, no data needed
# full pipeline (data download, encoding, evaluation): see README.md</pre>

<footer>Data: WILDS iWildCam v2.0 and MegaDetector v4 boxes from the iWildCam 2022 metadata (CDLA-Permissive).
Encoders used frozen: CLIP B/16, SigLIP 2 B/16, DINOv2 B/14, DINOv3 B/16.</footer>
</main>

<script>
const DATA = __DATA__;
const HITS = __HITS__;
const ORDER = ["clip", "dinov2", "dinov3", "siglip2"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// headline numbers
(() => {
  const g = ORDER.map(e => DATA[e].final.eyes[1] - DATA[e].final.glance);
  const looks = ORDER.map(e => DATA[e].final.stop_looks);
  const s = [
    [`+${Math.min(...g).toFixed(1)} to +${Math.max(...g).toFixed(1)}`, "points over glancing, with 2 looks"],
    [`${Math.round(HITS.eig_final)}%`, `of looks land on the animal (was ${Math.round(HITS.eig_first)}%)`],
    [`${Math.min(...looks).toFixed(1)}–${Math.max(...looks).toFixed(1)}`, "looks per clip with the stopping rule"],
    ["4 encoders", "CLIP, SigLIP 2, DINOv2, DINOv3, all frozen"]];
  document.getElementById("stats").innerHTML = s.map(([b, t]) => `<div class="stat"><b>${b}</b><span>${t}</span></div>`).join("");
})();

// accuracy vs looks
let current = "dinov2";
function drawChart() {
  const d = DATA[current], svg = document.getElementById("chart"), tip = document.getElementById("tip");
  const W = 560, H = 320, L = 44, R = 150, T = 14, B = 48;
  const series = [
    {name: "eyes, final", c: css("--blue"), y: [d.final.glance, ...d.final.eyes]},
    {name: "random looks, final", c: css("--aqua"), y: [d.final.glance, ...d.final.random]},
    {name: "eyes, as proposed", c: css("--orange"), y: [d.first.glance, ...d.first.eyes]}];
  const vals = series.flatMap(s => s.y).concat([d.final.all9]);
  const lo = Math.floor((Math.min(...vals) - 1) / 4) * 4, hi = Math.ceil((Math.max(...vals) + 1) / 4) * 4;
  const x = i => L + i * (W - L - R) / 4, y = v => T + (hi - v) * (H - T - B) / (hi - lo);
  let s = "";
  for (let v = lo; v <= hi; v += 4)
    s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${css("--line")}"/>` +
         `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  for (let i = 0; i <= 4; i++) s += `<text x="${x(i)}" y="${H - B + 18}" text-anchor="middle">${i}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">looks per clip (0 = glance only)</text>`;
  s += `<line x1="${L}" x2="${x(4)}" y1="${y(d.final.all9)}" y2="${y(d.final.all9)}" stroke="${css("--muted")}" stroke-dasharray="5 4"/>`;
  const labelY = [];
  series.forEach(se => {
    s += `<polyline fill="none" stroke="${se.c}" stroke-width="2.2" stroke-linejoin="round" points="${se.y.map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>`;
    se.y.forEach((v, i) => s += `<circle cx="${x(i)}" cy="${y(v)}" r="4.5" fill="${se.c}" stroke="${css("--card")}" stroke-width="1.5"/>`);
    labelY.push([y(se.y[4]), `${se.name} ${se.y[4].toFixed(1)}`, css("--ink")]);
  });
  labelY.push([y(d.final.all9), `every crop ${d.final.all9.toFixed(1)}`, css("--ink2")]);
  labelY.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < labelY.length; i++) labelY[i][0] = Math.max(labelY[i][0], labelY[i - 1][0] + 14);
  labelY.forEach(([ly, t, c]) => s += `<text x="${x(4) + 10}" y="${ly + 4}" style="fill:${c}">${t}</text>`);
  for (let i = 0; i <= 4; i++)
    s += `<rect x="${x(i) - 25}" y="${T}" width="50" height="${H - T - B}" fill="transparent" data-i="${i}"/>`;
  svg.innerHTML = s;
  svg.querySelectorAll("rect[data-i]").forEach(r => {
    r.addEventListener("mousemove", ev => {
      const i = +r.dataset.i, box = svg.getBoundingClientRect();
      tip.innerHTML = `<b>${i === 0 ? "glance only" : i + (i === 1 ? " look" : " looks")}</b><br>` +
        series.map(se => `<span style="color:${se.c}">●</span> ${se.name}: ${se.y[i].toFixed(1)}%`).join("<br>");
      tip.style.left = Math.min(ev.clientX - box.left + 14, box.width - 190) + "px";
      tip.style.top = (ev.clientY - box.top - 10) + "px";
      tip.style.opacity = 1;
    });
    r.addEventListener("mouseleave", () => tip.style.opacity = 0);
  });
}
const seg = document.getElementById("enc");
seg.innerHTML = ORDER.map(e => `<button type="button" data-e="${e}" aria-pressed="${e === current}">${DATA[e].name}</button>`).join("");
seg.addEventListener("click", ev => {
  const b = ev.target.closest("button"); if (!b) return;
  current = b.dataset.e;
  seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
  drawChart();
});
drawChart();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);

// where looks land
document.getElementById("bars").innerHTML = [
  ["eyes, final", HITS.eig_final, "--blue"], ["attention (B3)", HITS.attention, "--neutral"],
  ["motion (B4)", HITS.motion, "--neutral"], ["eyes, as proposed", HITS.eig_first, "--orange"],
  ["random looks", HITS.random, "--aqua"]].map(([n, v, c]) =>
  `<div class="bar"><span>${n}</span><span class="track"><span class="fill" style="display:block;width:${v}%;background:var(${c})"></span>` +
  `<span class="chance" style="left:${HITS.chance}%" title="chance ${HITS.chance.toFixed(0)}%"></span></span><span>${v.toFixed(0)}%</span></div>`).join("") +
  `<p class="muted" style="margin-top:6px">Dashed line: chance, ${HITS.chance.toFixed(0)}%.</p>`;

// table
(() => {
  const cols = [["glance", "Glance", "T"], ["first_eyes", "Eyes as proposed", "T+2"], ["eyes", "Eyes, final", "T+2"],
    ["stop", "Eyes + stopping", "≈T+1"], ["random", "Random", "T+2"], ["motion", "Motion (B4)", "T+2"],
    ["attention", "Attention (B3)", "T+2"], ["hires", "High-res (B1)", "4T"], ["all9", "Every crop", "10T"]];
  const row = e => { const f = DATA[e].final; return {glance: f.glance, first_eyes: DATA[e].first.eyes[1], eyes: f.eyes[1],
    stop: f.stop, random: f.random[1], motion: f.motion, attention: f.attention, hires: f.hires, all9: f.all9}; };
  let h = "<thead><tr><th>encoder</th>" + cols.map(c => `<th>${c[1]}<br><span class="muted">${c[2]} passes</span></th>`).join("") + "</tr></thead><tbody>";
  ORDER.forEach(e => {
    const r = row(e), cheap = ["eyes", "stop", "random", "motion", "attention", "first_eyes"].filter(k => r[k] != null);
    const best = cheap.reduce((a, k) => r[k] > r[a] ? k : a, cheap[0]);
    h += `<tr><td>${DATA[e].name}</td>` + cols.map(c => `<td class="${c[0] === best ? "best" : ""}">${r[c[0]] == null ? "—" : r[c[0]].toFixed(1)}</td>`).join("") + "</tr>";
  });
  document.getElementById("table").innerHTML = h + "</tbody>";
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.REPO / "docs" / "project_page"))
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(paths.processed("iwildcam") / "frames.csv")
    data = numbers()
    hits = look_hits(frames)
    exs = examples(frames, args.examples, 0, "dinov2", out)
    page = PAGE.replace("__DATA__", json.dumps(data)).replace("__HITS__", json.dumps(hits))
    page = page.replace("__EXAMPLES__", "".join(example_html(e) for e in exs))
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'} with {len(exs)} examples")


if __name__ == "__main__":
    main()
