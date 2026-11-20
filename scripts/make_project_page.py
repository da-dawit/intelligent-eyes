"""Build the project page: docs/project_page/index.html, its figures, and (with make_videos.py) its videos.

    python scripts/make_project_page.py --final-dump final.pkl

Layout and styles follow the Nerfies project page (CC BY-SA 4.0, credited in the footer); all
files are local, so the page also works offline. Numbers come from results/iwildcam/: the
method as proposed (<encoder>) and the final method (<encoder>-boxes-preview-gated-tau).
MegaDetector boxes are used only to measure where looks land; the method never sees them.
Figures are drawn at the size they are shown (10 in = the 960 px page column) in a plain
paper style; their explanations are in the HTML captions.
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
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyes import paths  # noqa: E402
from eyes.boxes import crop_labels, load_detections  # noqa: E402
from eyes.controller import run  # noqa: E402
from eyes.geometry import boxes  # noqa: E402

ENCODERS = [("clip", "CLIP"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3"), ("siglip2", "SigLIP 2")]
FINAL = "-boxes-preview-gated-tau"
C_FINAL, C_FIRST, C_RANDOM, C_REF, C_BASE = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f", "#b0b0b0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04})


# ------------------------------------------------------------------ numbers
def load(run_name):
    files = sorted((paths.RESULTS / "iwildcam" / run_name).glob("seed*/clips.csv.gz"))
    return pd.concat([pd.read_csv(f).assign(seed=f.parent.name) for f in files])


def acc(r, m):
    x = r[r.method == m]
    return None if x.empty else 100 * float(x.correct.mean())


def acc_seeds(r, m):
    """Accuracy (%) per seed (random looks: averaged over draws first)."""
    x = r[r.method == m]
    return 100 * x.groupby("seed").correct.mean().values


def numbers():
    out = {}
    for enc, name in ENCODERS:
        first, final = load(enc), load(enc + FINAL)
        looks = final[final.method == "eyesmix_stop"].passes.mean() - final[final.method == "glance"].passes.mean()
        out[enc] = dict(
            name=name,
            first=dict(glance=acc(first, "glance"), eyes=[acc(first, f"eyes_b{b}") for b in (1, 2, 3, 4)],
                       curve=[acc_seeds(first, "glance")] + [acc_seeds(first, f"eyes_b{b}") for b in (1, 2, 3, 4)]),
            final=dict(glance=acc(final, "glance"), eyes=[acc(final, f"eyesmix_b{b}") for b in (1, 2, 3, 4)],
                       random=[acc(final, f"random_b{b}") for b in (1, 2, 3, 4)], all9=acc(final, "all9"),
                       stop=acc(final, "eyesmix_stop"), stop_looks=float(looks),
                       motion=acc(final, "motion_b2"), attention=acc(final, "attention_b2"),
                       hires=acc(final, "hires"), best_pair=acc(final, "best_pair"),
                       curve=[acc_seeds(final, "glance")] + [acc_seeds(final, f"eyesmix_b{b}") for b in (1, 2, 3, 4)],
                       random_curve=[acc_seeds(final, "glance")] + [acc_seeds(final, f"random_b{b}") for b in (1, 2, 3, 4)]))
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
def letter(ax, ch, x=-0.02, y=1.0):
    """Bold panel letter in the top-left corner, outside the axes (as in journal figures)."""
    ax.text(x, y, ch, transform=ax.transAxes, ha="right", va="bottom", fontsize=14, fontweight="bold")


def fig_curves(data, out):
    """Accuracy against the number of looks; mean and standard deviation over the 3 seeds."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.55), layout="constrained")
    x = np.arange(5)
    for ax, (enc, _) in zip(axes, ENCODERS):
        d = data[enc]
        for ys, col, mk, lab in ((d["first"]["curve"], C_FIRST, "D", "as proposed"),
                                 (d["final"]["random_curve"], C_RANDOM, "s", "random looks"),
                                 (d["final"]["curve"], C_FINAL, "o", "final method")):
            m = np.array([v.mean() for v in ys])
            s = np.array([v.std() for v in ys])
            ax.errorbar(x, m, yerr=s, color=col, marker=mk, capsize=2, label=lab, ms=4.5)
        ax.axhline(d["final"]["all9"], color=C_REF, ls="--", lw=1, label="all crops ($10T$ passes)")
        ax.set_xticks(x)
        ax.set_xlabel("looks per clip")
        ax.set_title(d["name"], fontsize=12)
    axes[0].set_ylabel("accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_diagnosis(hits, out):
    """a: the crop classifier alone (scripts/diagnose_crops.py, DINOv2 seed 0); b: where 2 looks land."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.7]})
    a.bar([0, 1], [42.9, 85.5], width=0.55, color=[C_FIRST, C_FINAL])
    a.axhline(62.5, color=C_REF, ls="--", lw=1, label="glance, whole frame")
    a.legend(loc="upper left")
    a.set_xticks([0, 1])
    a.set_xticklabels(["clip labels", "empty class"])
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(0, 112)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_ylabel("top-1, animal crops (%)")
    letter(a, "a", x=-0.2)
    names = ["random", "proposed", "motion", "attention", "final"]
    vals = [hits["random"], hits["eig_first"], hits["motion"], hits["attention"], hits["eig_final"]]
    b.bar(np.arange(5), vals, width=0.6, color=[C_RANDOM, C_FIRST, C_BASE, C_BASE, C_FINAL])
    b.axhline(hits["chance"], color=C_REF, ls="--", lw=1, label="chance")
    b.legend(loc="upper left")
    b.set_xticks(np.arange(5))
    b.set_xticklabels(names)
    b.set_xlim(-0.6, 4.6)
    b.set_ylim(0, 112)
    b.set_yticks([0, 20, 40, 60, 80, 100])
    b.set_ylabel("looks on the animal (%)")
    letter(b, "b", x=-0.12)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_method(dump, frames, out, clip_id=None):
    """a: the glance frames, b: P(animal) per crop, c: the selected crop at 2x, d: the posterior."""
    D = pickle.load(open(dump, "rb"))
    cats = pd.read_csv(paths.DATA / "raw" / "iwildcam_v2.0" / "categories.csv")
    name_of = dict(zip(cats["y"], cats["name"]))
    names = {i: (lambda n: n[:1].upper() + n[1:])(name_of[int(y)]) for i, y in enumerate(D["classes"])}
    clips = {c.clip_id: c for c in D["clips"]}
    c = clips.get(clip_id, D["clips"][0])
    o0, o2 = run(c, D["model"], D["rule"], budget=0), run(c, D["model"], D["rule"], budget=2)
    rows = frames[frames.clip_id == c.clip_id].sort_values("frame_idx")
    imgs = [Image.open(paths.DATA / "raw" / "iwildcam_v2.0" / p).convert("RGB") for p in rows.path]
    f1, k1 = o2.looks[0]
    W, H = imgs[f1].size

    fig = plt.figure(figsize=(10, 2.15))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.52, 0.5, 0.95, 0.62, 0.78], wspace=0.12,
                          left=0.03, right=0.995, bottom=0.12, top=0.88)
    sub = gs[0].subgridspec(min(len(imgs), 3), 1, hspace=0.05)          # a: the frames
    for i, im in enumerate(imgs[:3]):
        ax = fig.add_subplot(sub[i])
        ax.imshow(np.asarray(im.resize((256, 144))))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            letter(ax, "a", x=-0.04, y=1.02)
    ax = fig.add_subplot(gs[1])                                          # b: P(animal), 3 x 3 crop offsets
    pa = c.p_animal[f1].reshape(3, 3)
    ax.imshow(pa, cmap="Blues", vmin=0, vmax=1)
    for r in range(3):
        for q in range(3):
            ax.text(q, r, f"{pa[r, q]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if pa[r, q] > 0.6 else "black")
    r1, q1 = divmod(k1 - 1, 3)
    ax.add_patch(plt.Rectangle((q1 - 0.5, r1 - 0.5), 1, 1, fill=False, ec="black", lw=2))
    ax.set_xticks([])
    ax.set_yticks([])
    letter(ax, "b", x=-0.04, y=1.02)
    ax = fig.add_subplot(gs[2])                                          # c: the selected crop at 2x
    b = boxes()[k1]
    ax.imshow(np.asarray(imgs[f1].crop((b[0] * W, b[1] * H, b[2] * W, b[3] * H))))
    ax.set_xticks([])
    ax.set_yticks([])
    letter(ax, "c", x=-0.02, y=1.02)
    ax = fig.add_subplot(gs[4])                                          # d: posterior (names in gs[3])
    top = list(np.argsort(-o2.p)[:3])[::-1]
    y = np.arange(len(top))
    ax.barh(y + 0.19, [o0.p[k] for k in top], height=0.36, color=C_BASE, label="glance")
    ax.barh(y - 0.19, [o2.p[k] for k in top], height=0.36, color=C_FINAL, label="+ 2 looks")
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in top], fontsize=10.5, fontstyle="italic")
    for lab, k in zip(ax.get_yticklabels(), top):
        if k == c.label:
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", left=False, right=False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.legend(loc="lower right", fontsize=9.5)
    box = gs[3].get_position(fig)
    fig.text(box.x0 + 0.005, 0.88, "d", ha="left", va="bottom", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches=None)
    plt.close(fig)


# ------------------------------------------------------------------ page
def fmt(v):
    return "&ndash;" if v is None else f"{v:.1f}"


def table_html(data):
    head = ("<tr><th>Encoder</th><th>Glance<br><span>T</span></th><th>As proposed<br><span>T+2</span></th>"
            "<th>Final<br><span>T+2</span></th><th>Final + stopping<br><span>looks/clip</span></th>"
            "<th>Random<br><span>T+2</span></th><th>Motion (B4)<br><span>T+2</span></th>"
            "<th>Attention (B3)<br><span>T+2</span></th><th>High-res (B1)<br><span>4T</span></th>"
            "<th>All crops<br><span>10T</span></th></tr>")
    rows = []
    for enc, _ in ENCODERS:
        d, f = data[enc], data[enc]["final"]
        rows.append(f"<tr><td>{d['name']}</td><td>{fmt(f['glance'])}</td><td>{fmt(d['first']['eyes'][1])}</td>"
                    f"<td><b>{fmt(f['eyes'][1])}</b></td><td>{fmt(f['stop'])} ({f['stop_looks']:.1f})</td>"
                    f"<td>{fmt(f['random'][1])}</td><td>{fmt(f['motion'])}</td><td>{fmt(f['attention'])}</td>"
                    f"<td>{fmt(f['hires'])}</td><td>{fmt(f['all9'])}</td></tr>")
    return f"<table class=\"table results-table\"><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def compare_items(video_dir):
    """The side-by-side videos listed (in order) by make_videos.py in videos.txt: file, species, frames."""
    items = []
    lines = (video_dir / "videos.txt").read_text().splitlines() if (video_dir / "videos.txt").exists() else []
    for i, line in enumerate(filter(None, lines)):
        name, species, T = line.split("\t")
        rel = f"static/videos/{name}"
        items.append(f"""
    <div class="video-block">
      <video poster="{rel.replace('.mp4', '.jpg')}" controls autoplay muted loop playsinline preload="metadata" width="100%">
        <source src="{rel}" type="video/mp4">
      </video>
      <p class="caption"><b>Video {i + 2}.</b> <i>{species}</i>, {T} frames. Left: method as proposed;
      right: final method.</p>
    </div>""")
    return "".join(items)


PAGE_FILE = Path(__file__).with_name("project_page_template.html")

CSS = """body { font-family: 'Noto Sans', 'Helvetica Neue', Arial, sans-serif; }
.publication-title, .publication-authors, .teaser { font-family: 'Google Sans', 'Noto Sans', 'Helvetica Neue', Arial, sans-serif; }
.publication-authors { color: #4a4a4a; }
.author-block { display: inline-block; }
.link-block a { margin-top: 5px; margin-bottom: 5px; }
.teaser .hero-body { padding-top: 0; padding-bottom: 3rem; }
.figure { display: block; width: 100%; margin: 0.75rem 0 0.25rem; }
.caption { font-size: 0.92rem; color: #4a4a4a; margin-bottom: 1.5rem; text-align: justify; }
.caption b { color: #363636; }
.results-carousel { overflow: hidden; padding: 0 3rem; }
.results-carousel .item { margin: 5px; overflow: hidden; border: 1px solid #ddd; padding: 0; font-size: 0; }
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
            .replace("__HIT__", f"{hits['eig_final']:.0f}")
            .replace("__CHANCE__", f"{hits['chance']:.0f}")
            .replace("__FIRST_HIT__", f"{hits['eig_first']:.0f}"))
    (out / "static" / "css" / "index.css").write_text(CSS)
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
