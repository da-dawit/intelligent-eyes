"""Build the project page: docs/project_page/index.html, its figures, and (with make_videos.py) its videos.

    python scripts/make_project_page.py --final-dump final.pkl

Layout and styles follow the Nerfies project page (CC BY-SA 4.0, credited in the footer); all
files are local, so the page also works offline. Numbers come from results/iwildcam/: the
method as proposed (<encoder>) and the final method (<encoder>-boxes-preview-gated-tau).
MegaDetector boxes are used only to measure where looks land; the method never sees them.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyes import paths  # noqa: E402
from eyes.boxes import crop_labels, load_detections  # noqa: E402
from eyes.controller import run  # noqa: E402
from eyes.geometry import boxes  # noqa: E402

ENCODERS = [("clip", "CLIP"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3"), ("siglip2", "SigLIP 2")]
FINAL = "-boxes-preview-gated-tau"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, NEUTRAL = "#111111", "#555555", "#8a8984", "#e6e5e1", "#bdbcb6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 11.5, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6})


# ------------------------------------------------------------------ numbers
def load(run_name):
    files = sorted((paths.RESULTS / "iwildcam" / run_name).glob("seed*/clips.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in files])


def acc(r, m):
    x = r[r.method == m]
    return None if x.empty else 100 * float(x.correct.mean())


def numbers():
    out = {}
    for enc, name in ENCODERS:
        first, final = load(enc), load(enc + FINAL)
        looks = final[final.method == "eyesmix_stop"].passes.mean() - final[final.method == "glance"].passes.mean()
        out[enc] = dict(
            name=name,
            first=dict(glance=acc(first, "glance"), eyes=[acc(first, f"eyes_b{b}") for b in (1, 2, 3, 4)]),
            final=dict(glance=acc(final, "glance"), eyes=[acc(final, f"eyesmix_b{b}") for b in (1, 2, 3, 4)],
                       random=[acc(final, f"random_b{b}") for b in (1, 2, 3, 4)], all9=acc(final, "all9"),
                       stop=acc(final, "eyesmix_stop"), stop_looks=float(looks),
                       motion=acc(final, "motion_b2"), attention=acc(final, "attention_b2"),
                       hires=acc(final, "hires"), best_pair=acc(final, "best_pair")))
    return out


def look_hits(frames, encoder="dinov2"):
    """Share of 2 looks (3 seeds) on a crop holding the animal, among crops that clearly hold it or
    clearly don't (eyes/boxes.py: crop_labels). Boxes are used only to measure this."""
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


# ------------------------------------------------------------------ figures
def clean(ax, keep=("bottom", "left")):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)


def fig_curves(data, out):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3))
    for ax, (enc, _) in zip(axes, ENCODERS):
        d = data[enc]
        x = [0, 1, 2, 3, 4]
        ax.axhline(d["final"]["all9"], color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.plot(x, [d["first"]["glance"]] + d["first"]["eyes"], color=ORANGE, lw=1.8, marker="D", ms=4)
        ax.plot(x, [d["final"]["glance"]] + d["final"]["random"], color=AQUA, lw=1.8, marker="s", ms=4)
        ax.plot(x, [d["final"]["glance"]] + d["final"]["eyes"], color=BLUE, lw=2.2, marker="o", ms=4.5)
        ax.set_title(d["name"], fontsize=13, color=INK, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        clean(ax)
        ax.set_xlabel("looks per clip")
    axes[0].set_ylabel("clip accuracy (%)")
    handles = [Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=4.5), Line2D([], [], color=AQUA, lw=1.8, marker="s", ms=4),
               Line2D([], [], color=ORANGE, lw=1.8, marker="D", ms=4), Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)))]
    fig.legend(handles, ["eyes (final)", "random looks (final)", "eyes (as proposed)", "every crop, 10T passes (final)"],
               loc="lower center", ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_diagnosis(hits, out):
    """(a) the crop classifier alone, from scripts/diagnose_crops.py (DINOv2, seed 0); (b) where looks land."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a.bar([-0.18, 0.82], [42.9, 17.5], width=0.34, color=ORANGE, label="as proposed")
    a.bar([0.18], [85.5], width=0.34, color=BLUE, label="with an empty class")
    for x, v in ((-0.18, 42.9), (0.82, 17.5), (0.18, 85.5)):
        a.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=11, color=INK)
    a.text(1.18, 6, "votes about 0\n(empty)", ha="center", fontsize=10.5, color=INK2)
    a.set_xticks([0, 1])
    a.set_xticklabels(["crop holds\nthe animal", "background\ncrop"])
    a.set_ylim(0, 100)
    a.set_ylabel("crop classifier top-1 (%)")
    a.set_title("(a) one crop alone (DINOv2, seed 0)", loc="left", fontsize=12.5, color=INK, fontweight="bold")
    a.legend(frameon=False, fontsize=10.5, loc="upper right")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    clean(a)
    rows = [("eyes (final)", hits["eig_final"], BLUE), ("attention (B3)", hits["attention"], NEUTRAL),
            ("motion (B4)", hits["motion"], NEUTRAL), ("eyes (as proposed)", hits["eig_first"], ORANGE),
            ("random looks", hits["random"], AQUA)]
    ys = list(range(len(rows)))[::-1]
    for y, (_, v, c) in zip(ys, rows):
        b.barh(y, v, height=0.56, color=c)
        b.text(max(v, hits["chance"]) + 1.5, y, f"{v:.0f}%", va="center", fontsize=11, color=INK)
    b.vlines(hits["chance"], -0.45, len(rows) - 0.55, color=MUTED, lw=1, ls=(0, (4, 3)))
    b.text(hits["chance"] - 1.2, len(rows) - 0.3, f"chance {hits['chance']:.0f}%", ha="right", fontsize=10.5, color=INK2)
    b.set_yticks(ys)
    b.set_yticklabels([r[0] for r in rows])
    b.set_xlim(0, 108)
    b.set_ylim(-0.6, len(rows))
    b.set_xlabel("looks that land on the animal (%)")
    b.set_title("(b) where 2 looks land (DINOv2, 3 seeds)", loc="left", fontsize=12.5, color=INK, fontweight="bold")
    b.tick_params(axis="y", length=0)
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    clean(b, keep=("bottom",))
    fig.tight_layout(w_pad=3)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_method(dump, frames, out, clip_id=None):
    """Glance -> P(animal) per crop -> 2x look -> belief, drawn from one real test clip."""
    D = pickle.load(open(dump, "rb"))
    cats = pd.read_csv(paths.DATA / "raw" / "iwildcam_v2.0" / "categories.csv")
    name_of = dict(zip(cats["y"], cats["name"]))
    names = {i: name_of[int(y)] for i, y in enumerate(D["classes"])}
    clips = {c.clip_id: c for c in D["clips"]}
    c = clips.get(clip_id, D["clips"][0])
    o0, o2 = run(c, D["model"], D["rule"], budget=0), run(c, D["model"], D["rule"], budget=2)
    rows = frames[frames.clip_id == c.clip_id].sort_values("frame_idx")
    imgs = [Image.open(paths.DATA / "raw" / "iwildcam_v2.0" / p).convert("RGB") for p in rows.path]
    f1, k1 = o2.looks[0]
    W, H = imgs[f1].size

    fig = plt.figure(figsize=(11, 2.5))
    pos = [[0.005, 0.04, 0.235, 0.7], [0.255, 0.04, 0.235, 0.7], [0.505, 0.04, 0.235, 0.7], [0.775, 0.16, 0.2, 0.56]]
    titles = [f"1  Glance at all {len(imgs)} frames", "2  P(animal) per crop",
              "3  Look at the best crop (2x)", "4  Belief update"]
    for (x, *_), t in zip(pos, titles):
        fig.text(x + 0.003, 0.86, t, fontsize=12.5, color=INK, fontweight="bold")
    fig.text(pos[3][0] + 0.003, 0.77, "grey: glance    blue: 2 looks", fontsize=10, color=INK2)
    ax = fig.add_axes(pos[0])                                      # 1: a fanned stack of the frames
    n = len(imgs)
    for i, im in enumerate(imgs[::-1]):
        j = n - 1 - i
        x0, y0 = 0.22 * j / max(n - 1, 1), 0.18 * (n - 1 - j) / max(n - 1, 1)
        ax.imshow(np.asarray(im.resize((320, 180))), extent=(x0, x0 + 1.6, y0, y0 + 0.9), zorder=i, aspect="auto")
        ax.add_patch(plt.Rectangle((x0, y0), 1.6, 0.9, fill=False, ec="white", lw=1, zorder=i + 0.5))
    ax.set_xlim(0, 1.84)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    ax = fig.add_axes(pos[1])                                      # 2: P(animal) at each crop's centre
    ax.imshow(np.asarray(imgs[f1]), aspect="auto")
    b = boxes()[k1]
    ax.add_patch(plt.Rectangle((b[0] * W, b[1] * H), 0.5 * W, 0.5 * H, fill=False, ec=BLUE, lw=2.4))
    for k in range(9):
        bb = boxes()[k + 1]
        cx, cy = (bb[0] + bb[2]) / 2 * W, (bb[1] + bb[3]) / 2 * H
        ax.text(cx, cy, f"{c.p_animal[f1, k]:.2f}", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold" if k + 1 == k1 else "normal",
                bbox=dict(boxstyle="round,pad=0.18", fc=BLUE if k + 1 == k1 else (0, 0, 0, 0.5), ec="none"))
    ax.axis("off")
    ax = fig.add_axes(pos[2])                                      # 3: the look
    ax.imshow(np.asarray(imgs[f1].crop((b[0] * W, b[1] * H, b[2] * W, b[3] * H))), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUE)
        sp.set_linewidth(2.4)
    ax = fig.add_axes(pos[3])                                      # 4: belief before and after
    top = list(np.argsort(-o2.p)[:3])
    ys = np.arange(len(top))[::-1] * 1.25
    ax.barh(ys + 0.0, [o0.p[k] for k in top], height=0.3, color=NEUTRAL, label="after the glance")
    ax.barh(ys - 0.32, [o2.p[k] for k in top], height=0.3, color=BLUE, label="after 2 looks")
    for y, k in zip(ys, top):
        ax.text(0, y + 0.3, names[k] + ("  (true)" if k == c.label else ""), fontsize=11,
                color=INK, fontweight="bold" if k == c.label else "normal", style="normal" if k == c.label else "italic")
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.6, ys.max() + 0.65)
    clean(ax, keep=("bottom",))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ page
def fmt(v):
    return "&ndash;" if v is None else f"{v:.1f}"


def table_html(data):
    head = ("<tr><th>Encoder</th><th>Glance<br><span>T</span></th><th>Eyes, as proposed<br><span>T+2</span></th>"
            "<th>Eyes, final<br><span>T+2</span></th><th>Eyes + stopping<br><span>looks/clip</span></th>"
            "<th>Random<br><span>T+2</span></th><th>Motion (B4)<br><span>T+2</span></th>"
            "<th>Attention (B3)<br><span>T+2</span></th><th>High-res (B1)<br><span>4T</span></th>"
            "<th>Every crop<br><span>10T</span></th></tr>")
    rows = []
    for enc, _ in ENCODERS:
        d, f = data[enc], data[enc]["final"]
        rows.append(f"<tr><td>{d['name']}</td><td>{fmt(f['glance'])}</td><td>{fmt(d['first']['eyes'][1])}</td>"
                    f"<td><b>{fmt(f['eyes'][1])}</b></td><td>{fmt(f['stop'])} ({f['stop_looks']:.1f})</td>"
                    f"<td>{fmt(f['random'][1])}</td><td>{fmt(f['motion'])}</td><td>{fmt(f['attention'])}</td>"
                    f"<td>{fmt(f['hires'])}</td><td>{fmt(f['all9'])}</td></tr>")
    return f"<table class=\"table results-table\"><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def compare_items(video_dir):
    """The side-by-side videos listed (in order) by make_videos.py in videos.txt."""
    items = []
    listed = (video_dir / "videos.txt").read_text().split() if (video_dir / "videos.txt").exists() else []
    for name in listed:
        mp4 = video_dir / name
        rel = f"static/videos/{mp4.name}"
        items.append(f"""
      <div class="item">
        <video poster="{rel.replace('.mp4', '.jpg')}" controls muted loop playsinline preload="metadata" width="100%">
          <source src="{rel}" type="video/mp4">
        </video>
      </div>""")
    return "".join(items)


PAGE_FILE = Path(__file__).with_name("project_page_template.html")


CSS = """body { font-family: 'Noto Sans', 'Helvetica Neue', Arial, sans-serif; }
.publication-title, .publication-authors, .teaser { font-family: 'Google Sans', 'Noto Sans', 'Helvetica Neue', Arial, sans-serif; }
.publication-authors { color: #4a4a4a; }
.author-block { display: inline-block; }
.link-block a { margin-top: 5px; margin-bottom: 5px; }
.teaser .hero-body { padding-top: 0; padding-bottom: 3rem; }
.teaser video { border-radius: 6px; }
.figure { display: block; width: 100%; margin: 0.5rem 0 1.25rem; }
.results-carousel { overflow: hidden; }
.results-carousel .item { margin: 5px; overflow: hidden; border: 1px solid #bbb; border-radius: 10px; padding: 0; font-size: 0; }
.results-carousel video { margin: 0; }
.results-table { width: 100%; font-variant-numeric: tabular-nums; }
.results-table th { font-weight: 600; font-size: 0.85rem; vertical-align: bottom; text-align: right; white-space: nowrap; }
.results-table th span { font-weight: 400; color: #7a7a7a; }
.results-table td { text-align: right; }
.results-table th:first-child, .results-table td:first-child { text-align: left; }
pre { white-space: pre; overflow-x: auto; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.REPO / "docs" / "project_page"))
    ap.add_argument("--final-dump", default="", help="pickle from run_eval --dump (final method) for the method figure")
    ap.add_argument("--method-clip", default="test-22020")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("images", "css"):
        (out / "static" / sub).mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(paths.processed("iwildcam") / "frames.csv")
    data = numbers()
    hits = look_hits(frames)
    fig_curves(data, out / "static" / "images" / "curves.png")
    fig_diagnosis(hits, out / "static" / "images" / "diagnosis.png")
    if args.final_dump:
        fig_method(args.final_dump, frames, out / "static" / "images" / "method.png", args.method_clip)
    gains = [data[e]["final"]["eyes"][1] - data[e]["final"]["glance"] for e, _ in ENCODERS]
    page = (PAGE_FILE.read_text().replace("__TABLE__", table_html(data))
            .replace("__COMPARE__", compare_items(out / "static" / "videos"))
            .replace("__GAINS__", f"{min(gains):.1f}&ndash;{max(gains):.1f}")
            .replace("__HIT__", f"{hits['eig_final']:.0f}"))
    (out / "static" / "css" / "index.css").write_text(CSS)
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
