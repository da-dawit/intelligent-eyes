"""Build the project page docs/project_page/index.html, its figures, and (with make_videos.py) its videos.

    python scripts/make_project_page.py --final-dump final.pkl

Layout and styles follow the Nerfies project page (CC BY-SA 4.0, credited in the footer). All files
are local, so the page also works offline. Numbers come from results/iwildcam/, for the method as
proposed (<encoder>) and the final method (<encoder>-boxes-preview-gated-tau). MegaDetector boxes are
used only to measure where looks land. The method never sees them.
Figures are drawn at the size they are shown (10 in = the 960 px page column) in a plain paper style,
and the HTML captions explain them. The benchmark tables also read the ablation runs, results/ddr/ and
results/synthetic/summary.md. The midpoint memo and derivation PDFs are copied next to the page, so
docs/project_page/ is a complete site (GitHub Pages serves docs/ from main).
"""
import argparse
import pickle
import re
import shutil
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
from summarize import THRESH, encoder_summary  # noqa: E402

ENCODERS = [("clip", "CLIP"), ("dinov2", "DINOv2"), ("dinov3", "DINOv3"), ("siglip2", "SigLIP 2")]
FINAL = "-boxes-preview-gated-tau"
C_FINAL, C_FIRST, C_RANDOM, C_REF, C_BASE = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f", "#b0b0b0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04})


def seeds(run_name, dataset="iwildcam"):
    return sorted((paths.RESULTS / dataset / run_name).glob("seed*/clips.csv.gz"))


def load(run_name, dataset="iwildcam"):
    return pd.concat([pd.read_csv(f).assign(seed=f.parent.name) for f in seeds(run_name, dataset)])


def acc(r, m):
    x = r[r.method == m]
    return None if x.empty else 100 * float(x.correct.mean())


def acc_seeds(r, m):
    """Accuracy (%) per seed. For random looks this also averages over the draws."""
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


def look_hits(frames, encoders=("dinov2",)):
    """Share of the 2 looks (3 seeds) that land on a crop holding the animal, per encoder.

    Only crops that clearly hold the animal or clearly don't are counted (crop_labels in eyes/boxes.py).
    The boxes are used only to measure this."""
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

    c = content[frames.index[frames.split == "test"].values]
    chance = 100 * (c == 1).sum() / (c >= 0).sum()
    out = {}
    for enc in encoders:
        first, final = load(enc), load(enc + FINAL)
        out[enc] = dict(chance=chance, random=share(final, "random_b2"), eig_first=share(first, "eyes_b2"),
                        motion=share(final, "motion_b2"), attention=share(final, "attention_b2"),
                        eig_final=share(final, "eyesmix_b2"))
    return out


# Each row adds one change to the row above (README, "Changes from the proposal").
# Entries are (label, run suffix, rule).
LADDER = [("As proposed", "", "eyes"),
          ("+ empty class for crops (change 2)", "-boxes", "eyes"),
          ("+ look predictor starts from a guess (6)", "-boxes-preview", "eyes"),
          ("+ gated fusion (3)", "-boxes-preview-gated", "eyes"),
          ("+ two-branch EIG (5)", "-boxes-preview-gated", "eyesmix"),
          ("+ shared crop error &tau; in <i>R</i> (4), final", FINAL, "eyesmix")]


def ladder():
    """Accuracy with 2 looks and with all crops after each change. None where a run has fewer than 3 seeds."""
    out = {}
    for enc, _ in ENCODERS:
        for i, (_, suffix, rule) in enumerate(LADDER):
            if len(seeds(enc + suffix)) < 3:
                out[enc, i] = None
                continue
            r = load(enc + suffix)
            out[enc, i] = (acc(r, f"{rule}_b2"), acc(r, "all9"))
    return out


def proposal_metrics():
    """The proposal's success metrics for the final method on iWildCam (scripts/summarize.py)."""
    return {enc: encoder_summary(paths.RESULTS / "iwildcam" / (enc + FINAL))[0] for enc, _ in ENCODERS}


DDR_METHODS = [("glance", "Glance", "1"), ("random_b2", "Random looks", "3"), ("eyes_b2", "Eyes as proposed", "3"),
               ("hires", "High-res (B1)", "4"), ("all9", "All crops", "10"), ("best_pair", "Best pair (oracle)", "3")]


def ddr():
    """Accuracy and quadratic kappa per method on DDR (one photo per clip, grades 0-4), method as proposed."""
    out = {}
    for enc, _ in ENCODERS:
        if len(seeds(enc, "ddr")) < 3:
            continue
        r = load(enc, "ddr")
        official = encoder_summary(paths.RESULTS / "ddr" / enc, ordinal=True)[0]["official"]
        out[enc] = {m: (acc(r, m), official[m][1]) for m, _, _ in DDR_METHODS if m in official}
    return out


def md_table(text, first_header):
    """Rows (lists of cell strings) of the markdown table whose header starts with `first_header`."""
    lines = text.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith("| " + first_header))
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def synthetic():
    """Looks budget, shrinkage sweep and All-9 from results/synthetic/summary.md (scripts/synthetic_study.py)."""
    text = (paths.RESULTS / "synthetic" / "summary.md").read_text()
    def pm(cell):                                                       # "74.1 +- 1.4" -> (74.1, 1.4)
        return tuple(float(x) for x in cell.split("\u00b1"))
    looks = [(int(r[0]), *pm(r[2]), *pm(r[3])) for r in md_table(text, "looks")]
    lam = [(float(r[0]), float(r[3]), float(r[5])) for r in md_table(text, "lam")]
    all9, all9_passes = map(float, re.search(r"All-9: ([\d.]+)% \(([\d.]+) passes\)", text).groups())
    chosen = [float(x) for x in re.search(r"per seed: \[([^\]]+)\]", text).group(1).split(",")]
    return dict(looks=looks, lam=lam, all9=all9, all9_passes=all9_passes, chosen=chosen)


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
    """Panel a is the crop classifier alone (scripts/diagnose_crops.py, DINOv2 seed 0), panel b where 2 looks land."""
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


def fig_synthetic(syn, out):
    """Synthetic clips with a known R. Panel a is accuracy against looks, panel b against the shrinkage of A."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.55), layout="constrained")
    L = np.array(syn["looks"])                                          # looks, eyes mean, sd, random mean, sd
    a.errorbar(L[:, 0], L[:, 3], yerr=L[:, 4], color=C_RANDOM, marker="s", capsize=2, ms=4.5)
    a.errorbar(L[:, 0], L[:, 1], yerr=L[:, 2], color=C_FINAL, marker="o", capsize=2, ms=4.5)
    a.axhline(syn["all9"], color=C_REF, ls="--", lw=1)
    a.text(L[-1, 0], L[-1, 1] - 4, "eyes (EIG)", color=C_FINAL, ha="right", va="top")           # direct labels
    a.text(L[-1, 0], L[-1, 3] - 4, "random looks", color=C_RANDOM, ha="right", va="top")
    a.text(0, syn["all9"] + 1.5, f"all crops ({syn['all9_passes']:.0f} passes)", color=C_REF, ha="left", va="bottom")
    a.set_xticks(L[:, 0])
    a.set_xlabel("looks per clip")
    a.set_ylabel("accuracy (%)")
    a.set_ylim(30, 90)
    letter(a, "a", x=-0.09)
    lam = np.array(syn["lam"])                                          # lambda, eyes (2 looks), all crops
    b.axvspan(min(syn["chosen"]) - 0.02, max(syn["chosen"]) + 0.02, color="#ececec", zorder=0,
              label="chosen by held-out likelihood")
    b.plot(lam[:, 0], lam[:, 2], color=C_REF, marker="^", ls="--", ms=5, label="all crops")
    b.plot(lam[:, 0], lam[:, 1], color=C_FINAL, marker="o", ms=4.5, label="eyes, 2 looks")
    b.set_xlabel(r"shrinkage $\lambda$ in $A_\lambda = (1-\lambda)A + \lambda I$")
    b.set_ylim(30, 90)
    b.set_xlim(-0.03, 1.03)
    b.legend(loc="lower right")
    letter(b, "b", x=-0.09)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_method(dump, frames, out, clip_id=None):
    """Method figure. Panels are (a) the glance frames, (b) P(animal) per crop, (c) the selected crop at 2x
    and (d) the posterior."""
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


def verdict(ok, note=""):
    icon, word = ("check", "pass") if ok else ("times", "fail")
    return f'<span class="verdict {word}"><i class="fas fa-{icon}"></i> {word}</span>' + (f"<span>{note}</span>" if note else "")


def metrics_html(m, hits):
    """The proposal's success metrics (metric 4 was dropped after the proposal feedback)."""
    E = [e for e, _ in ENCODERS]
    gain = [100 * m[e]["gain"][0] for e in E]
    vs_rand = [100 * m[e]["vs_random"][0] for e in E]
    rec_ok = [m[e]["recovered"] >= THRESH["recovered"] for e in E if m[e]["all9_gain"] >= THRESH["gain"]]
    rho = [m[e]["spearman"] for e in E]
    ece = [m[e]["ece_eyes"] for e in E]
    ci = [f"{g:+.1f}<span>[{100 * m[e]['gain'][1]:.1f}, {100 * m[e]['gain'][2]:.1f}]</span>" for g, e in zip(gain, E)]
    n1 = sum(g >= 100 * THRESH["gain"] for g in gain)
    rows = [("1. Gain over the glance (points)", "&ge; 2 on 3 of 4 encoders", ci,
             verdict(n1 >= 3, f"{n1} of 4")),
            ("2. Gain over random looks (points)", "&ge; 1, mean", [f"{v:+.1f}" for v in vs_rand],
             verdict(np.mean(vs_rand) >= 100 * THRESH["beats_random"], f"mean {np.mean(vs_rand):+.1f}")),
            ("3. Share of the all-crops gain", "&ge; 2/3", [f"{m[e]['recovered']:.2f}" for e in E],
             verdict(all(rec_ok))),
            ("5. EIG ranks the looks (Spearman)", "&ge; 0.3, mean", [f"{v:.2f}" for v in rho],
             verdict(np.mean(rho) >= THRESH["spearman"], f"mean {np.mean(rho):.2f}")),
            ("6. Calibration error (ECE)", "&le; 0.05", [f"{v:.3f}" for v in ece],
             verdict(max(ece) <= THRESH["ece"])),
            ("Looks that land on the animal (%)", f"chance is {hits[E[0]]['chance']:.0f}",
             [f"{hits[e]['eig_final']:.0f}" for e in E], "&ndash;")]
    head = ("<tr><th>Metric</th><th>Threshold</th>" + "".join(f"<th>{n}</th>" for _, n in ENCODERS)
            + "<th>Result</th></tr>")
    body = "".join(f"<tr><td>{a}</td><td>{t}</td>" + "".join(f"<td>{c}</td>" for c in cells) + f"<td>{r}</td></tr>"
                   for a, t, cells, r in rows)
    return f'<table class="table results-table metrics-table"><thead>{head}</thead><tbody>{body}</tbody></table>'


def ladder_html(lad):
    head = ("<tr><th rowspan=\"2\">Change</th>" + "".join(f"<th colspan=\"2\" class=\"group\">{n}</th>" for _, n in ENCODERS)
            + "</tr><tr>" + "<th>2 looks</th><th>all crops</th>" * len(ENCODERS) + "</tr>")
    rows = []
    for i, (label, _, _) in enumerate(LADDER):
        cells = []
        for enc, _ in ENCODERS:
            v = lad[enc, i]
            cells += ["&ndash;", "&ndash;"] if v is None else [fmt(v[0]), fmt(v[1])]
        if i == len(LADDER) - 1:
            cells = [f"<b>{c}</b>" for c in cells]
        rows.append(f"<tr><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<table class="table results-table ladder-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def ddr_html(d):
    head = ("<tr><th>Encoder</th>" + "".join(f"<th>{lab}<br><span>{n} pass{'es' if n != '1' else ''}</span></th>" for _, lab, n in DDR_METHODS)
            + "</tr>")
    rows = []
    for enc, name in ENCODERS:
        if enc not in d:
            continue
        cells = "".join("<td>&ndash;</td>" if m not in d[enc] else
                        f"<td>{d[enc][m][0]:.1f}<span>&kappa; {d[enc][m][1]:.2f}</span></td>" for m, _, _ in DDR_METHODS)
        rows.append(f"<tr><td>{name}</td>{cells}</tr>")
    return f'<table class="table results-table ddr-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def video_rows(video_dir, name):
    """Rows of a tab-separated list written by make_videos.py. The page shows the videos in this order."""
    path = video_dir / name
    return [line.split("\t") for line in path.read_text().splitlines() if line] if path.exists() else []


def video_tag(rel):
    """A looping, muted video that loads and plays only while on screen (script at the end of the page)."""
    return (f'<video poster="{rel[:-4]}.jpg" data-inview controls muted loop playsinline preload="none" width="100%">'
            f'<source src="{rel}" type="video/mp4"></video>')


def and_list(xs):
    """'a', 'a and b', 'a, b and c'."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def who(names, right, noun):
    """One sentence saying which panels get the species right, from the panel names and a '1,0,...' string."""
    ok = [n for n, r in zip(names, right.split(",")) if r == "1"]
    bad = [n for n, r in zip(names, right.split(",")) if r != "1"]
    if not bad:
        return f"All four {noun} get it right."
    if not ok:
        return f"All four {noun} get it wrong."
    verb = lambda xs: "gets" if len(xs) == 1 else "get"
    s = f"{and_list(ok)} {verb(ok)} it right, and {and_list(bad)} {verb(bad)} it wrong."
    return s[0].upper() + s[1:]


def tabbed(items):
    """One video at a time, with a button per clip (script at the end of the page)."""
    if not items:
        return ""
    buttons = "".join(f'<button class="tag is-medium{" is-active" if i == 0 else ""}">{label}</button>'
                      for i, (label, _) in enumerate(items))
    blocks = "".join(block.replace('<div class="video-block', '<div hidden class="video-block', 1) if i else block
                     for i, (_, block) in enumerate(items))
    return f'<div class="clip-tabs">{buttons}</div><div class="clip-panels">{blocks}</div>'


def grid_items(video_dir, name, num):
    """Clip tabs for the baseline or encoder grids. Rows are file, species, frames, correct per panel, panel names."""
    items = []
    for file, species, T, right, panels in video_rows(video_dir, name):
        items.append((f"<i>{species}</i>", f"""
    <div class="video-block">
      {video_tag(f"static/videos/{file}")}
      <p class="caption"><b>Video {num[0]}.</b> <i>{species}</i>, {T} frames. {who(panels.split(","), right, "encoders" if "encoders" in name else "methods")}</p>
    </div>"""))
        num[0] += 1
    return tabbed(items)


def gallery_items(video_dir, num):
    """Clip tabs for the stopping-rule cards. Rows are file, species, frames, looks, correct, predicted."""
    items = []
    for file, species, T, looks, ok, pred in video_rows(video_dir, "gallery.txt"):
        n = int(looks)
        took = ("It stops right after the glance" if n == 0 else "It uses all four looks" if n == 4 else
                f"It takes {n} look{'' if n == 1 else 's'}")
        result = "and gets the species right" if ok == "1" else f"but ends on <i>{pred}</i>, which is wrong"
        label = f"<i>{species}</i>" + ("" if ok == "1" else ' <span class="tab-note">wrong</span>')
        items.append((label, f"""
    <div class="video-block card-block">
      {video_tag(f"static/videos/{file}")}
      <p class="caption"><b>Video {num[0]}.</b> <i>{species}</i>, {T} frames. {took} {result}.</p>
    </div>"""))
        num[0] += 1
    return tabbed(items)


def compare_items(video_dir, num):
    """Clip tabs for the side-by-side videos (as proposed | final). Rows are file, species, frames."""
    items = []
    for file, species, T in video_rows(video_dir, "videos.txt"):
        items.append((f"<i>{species}</i>", f"""
    <div class="video-block">
      {video_tag(f"static/videos/{file}")}
      <p class="caption"><b>Video {num[0]}.</b> <i>{species}</i>, {T} frames.</p>
    </div>"""))
        num[0] += 1
    return tabbed(items)


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
.results-table td span { display: block; font-size: 0.78rem; color: #7a7a7a; }
.results-table th.group { text-align: center; border-bottom-width: 1px; }
.metrics-table td:nth-child(2) { color: #4a4a4a; white-space: nowrap; }
.metrics-table td:first-child, .ladder-table td:first-child { white-space: nowrap; }
.verdict { font-weight: 600; white-space: nowrap; }
.results-table td span.verdict { display: inline; font-size: inherit; }
.verdict.pass { color: #198754; }
.verdict.fail { color: #c82333; }
.bench-nav { margin: 0.25rem 0 1.25rem; }
.bench-nav .tag { margin: 0 0.35rem 0.4rem 0; }
.bench-nav a.tag:hover { background: #363636; color: #fff; }
.results h3.title { margin-top: 2.5rem; scroll-margin-top: 1rem; }
video { display: block; border-radius: 4px; }
.video-block { margin-top: 1rem; }
.viewer { margin-top: 1rem; border: 1px solid #e4e4e4; border-radius: 8px; padding: 1rem 1.25rem 0.5rem; }
.viewer-cats { display: flex; flex-wrap: wrap; gap: 0.4rem; border-bottom: 1px solid #e4e4e4; padding-bottom: 0.9rem; }
.viewer-cats button { font: inherit; font-size: 0.95rem; padding: 0.45rem 0.9rem; border: 1px solid #dbdbdb;
  border-radius: 999px; background: #fff; color: #363636; cursor: pointer; }
.viewer-cats button span { color: #7a7a7a; font-size: 0.8rem; margin-left: 0.2rem; }
.viewer-cats button.is-active { background: #363636; border-color: #363636; color: #fff; }
.viewer-cats button.is-active span { color: #d0d0d0; }
.viewer-desc { margin: 0.9rem 0 0.25rem; font-size: 0.95rem; }
.card-block { max-width: 560px; margin-left: auto; margin-right: auto; }
.tab-note { font-style: normal; font-size: 0.75rem; color: #c82333; margin-left: 0.2rem; }
.clip-tabs .tag.is-active .tab-note { color: #ffb3b3; }
.videos h3.title { margin-top: 2.5rem; scroll-margin-top: 1rem; }
.clip-tabs { margin: 0.75rem 0 0.25rem; }
.clip-tabs .tag { margin: 0 0.35rem 0.4rem 0; border: 0; cursor: pointer; }
.clip-tabs .tag.is-active { background: #363636; color: #fff; }
@media (max-width: 600px) {
  .metrics-table td:first-child, .ladder-table td:first-child { white-space: normal; min-width: 9rem; }
}
pre { white-space: pre; overflow-x: auto; }
.method h4 { font-size: 1.1rem; margin: 1.6rem 0 0.4rem; }
.method ul { margin-top: 0.25rem; }
.method li { margin-bottom: 0.3rem; }
.katex { font-size: 1.08em; }
.katex-display { margin: 0.7rem 0 0.9rem; padding: 0.2rem 0; overflow-x: auto; overflow-y: hidden; }
@media (max-width: 600px) { .katex-display { font-size: 0.72em; } }
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
    (out / "static" / "pdf").mkdir(parents=True, exist_ok=True)
    for name in ("midpoint_memo.pdf", "derivations_step_by_step.pdf"):             # linked from the page
        shutil.copy2(paths.REPO / "docs" / name, out / "static" / "pdf" / name)
    frames = pd.read_csv(paths.processed("iwildcam") / "frames.csv")
    data = numbers()
    hits_all = look_hits(frames, [e for e, _ in ENCODERS])
    hits = hits_all["dinov2"]
    syn = synthetic()
    fig_curves(data, out / "static" / "images" / "curves.png")
    fig_diagnosis(hits, out / "static" / "images" / "diagnosis.png")
    fig_synthetic(syn, out / "static" / "images" / "synthetic.png")
    if args.final_dump:
        fig_method(args.final_dump, frames, out / "static" / "images" / "method.png", args.method_clip)
    gains = [data[e]["final"]["eyes"][1] - data[e]["final"]["glance"] for e, _ in ENCODERS]
    on_animal = [hits_all[e]["eig_final"] for e, _ in ENCODERS]
    looks = {n: (e, r) for n, e, _, r, _ in syn["looks"]}
    lam = {round(x, 2): a9 for x, _, a9 in syn["lam"]}
    vids, num = out / "static" / "videos", [2]                                    # Video 1 is the teaser
    page = (PAGE_FILE.read_text().replace("__TABLE__", table_html(data))
            .replace("__METRICS__", metrics_html(proposal_metrics(), hits_all))
            .replace("__LADDER__", ladder_html(ladder()))
            .replace("__DDR__", ddr_html(ddr()))
            .replace("__BASELINES__", grid_items(vids, "baselines.txt", num))
            .replace("__ENCODERS__", grid_items(vids, "encoders.txt", num))
            .replace("__GALLERY__", gallery_items(vids, num))
            .replace("__COMPARE__", compare_items(vids, num))
            .replace("__STOP_LOOKS__", f"{data['dinov2']['final']['stop_looks']:.1f}")
            .replace("__NB__", str(len(video_rows(vids, "baselines.txt"))))
            .replace("__NE__", str(len(video_rows(vids, "encoders.txt"))))
            .replace("__NG__", str(len(video_rows(vids, "gallery.txt"))))
            .replace("__NC__", str(len(video_rows(vids, "videos.txt"))))
            .replace("__GAINS__", f"{min(gains):.1f}&ndash;{max(gains):.1f}")
            .replace("__HITS__", f"{min(on_animal):.0f}&ndash;{max(on_animal):.0f}")
            .replace("__HIT__", f"{hits['eig_final']:.0f}")
            .replace("__CHANCE__", f"{hits['chance']:.0f}")
            .replace("__FIRST_HIT__", f"{hits['eig_first']:.0f}")
            .replace("__SYN_EYES2__", f"{looks[2][0]:.0f}")
            .replace("__SYN_RAND2__", f"{looks[2][1]:.0f}")
            .replace("__SYN_GLANCE__", f"{looks[0][0]:.0f}")
            .replace("__SYN_ALL9__", f"{syn['all9']:.0f}")
            .replace("__SYN_ALL9_PASSES__", f"{syn['all9_passes']:.0f}")
            .replace("__SYN_LAM0__", f"{lam[0.0]:.0f}")
            .replace("__SYN_LAMBEST__", f"{max(lam.values()):.0f}"))
    (out / "static" / "css" / "index.css").write_text(CSS)
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
