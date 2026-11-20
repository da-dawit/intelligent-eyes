"""Step 4: turn per-clip results into the proposal's Table I and the accuracy-vs-passes figure.

    python scripts/summarize.py --dataset iwildcam

Reads results/<dataset>/<run>/seed*/ and writes results/<dataset>/summary.md and
results/<dataset>/accuracy_vs_passes.png. Thresholds are the ones fixed in the proposal.
"The eyes" are the two-branch EIG (eyesmix_*) when a run has it, else plain EIG (eyes_*).
--only clip,dinov2-boxes-...  restricts the runs;  --out NAME  writes NAME.md / NAME.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eyes import paths
from eyes.metrics import camera_bootstrap, ece_from_confidence, macro_f1, mcnemar, quadratic_kappa

THRESH = dict(gain=0.02, beats_random=0.01, recovered=2 / 3, near_optimal=3 / 4, spearman=0.3, ece=0.05)


def per_clip(res: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-clip correctness/confidence averaged over seeds and random draws."""
    r = res[res["method"] == method]
    return r.groupby("clip_id").agg(correct=("correct", "mean"), conf=("conf", "mean"),
                                    passes=("passes", "mean"), camera=("camera", "first"), T=("T", "first"))


OFFICIAL = ("glance", "hires", "random_b2", "motion_b2", "attention_b2", "eyes_b2", "eyes_stop",
            "eyesmix_b2", "eyesmix_stop", "best_pair", "all9")


def official_metrics(res: pd.DataFrame, ordinal: bool) -> dict:
    """Macro-F1 (and quadratic kappa for graded labels) per method, mean over seeds and draws."""
    out = {}
    for m in OFFICIAL:
        r = res[res["method"] == m]
        if r.empty:
            continue
        groups = [g for _, g in r.groupby(["seed", "draw"])]
        f1 = np.mean([macro_f1(g["pred"].values, g["label"].values) for g in groups])
        kap = np.mean([quadratic_kappa(g["pred"].values, g["label"].values, int(res["label"].max()) + 1)
                       for g in groups]) if ordinal else float("nan")
        out[m] = (f1, kap)
    return out


def encoder_summary(enc_dir: Path, ordinal: bool = False):
    seeds = sorted(enc_dir.glob("seed*"))
    res = pd.concat([pd.read_csv(s / "clips.csv.gz").assign(seed=s.name) for s in seeds])
    ranking = pd.concat([pd.read_csv(s / "ranking.csv").assign(seed=s.name) for s in seeds])
    fits = [json.loads((s / "fit.json").read_text()) for s in seeds]
    pairs = pd.concat([pd.read_csv(s / "pairs.csv.gz").assign(seed=s.name) for s in seeds])
    res["correct"] = res["correct"].astype(bool).astype(float)

    acc_seed = res.groupby(["seed", "method"])["correct"].mean().unstack(0)
    passes = res.groupby("method")["passes"].mean()
    E = "eyesmix" if "eyesmix_b2" in set(res["method"]) else "eyes"
    eyes, glance, rand, all9 = (per_clip(res, m) for m in (f"{E}_b2", "glance", "random_b2", "all9"))
    stop = per_clip(res, f"{E}_stop")

    def boot(diff, cams):
        return camera_bootstrap(diff.values, cams.values)

    gain = boot(eyes["correct"] - glance["correct"], eyes["camera"])
    vs_rand = boot(eyes["correct"] - rand["correct"], eyes["camera"])
    baselines = {"B1 high-res": "hires", "B3 attention": "attention_b2", "B4 motion": "motion_b2"}
    vs_base = {}
    for label, m in baselines.items():
        if m in set(res["method"]):
            b = per_clip(res, m)
            vs_base[label] = (b["correct"].mean(), b["passes"].mean(),
                              boot(eyes["correct"] - b["correct"].loc[eyes.index], eyes["camera"]))
    total = all9["correct"].mean() - glance["correct"].mean()
    recovered = (eyes["correct"].mean() - glance["correct"].mean()) / total if total > 0 else float("nan")
    mcn = []
    for s in res["seed"].unique():
        a = res[(res.seed == s) & (res.method == f"{E}_b2")].set_index("clip_id")["correct"].astype(bool)
        b = res[(res.seed == s) & (res.method == "glance")].set_index("clip_id")["correct"].astype(bool)
        mcn.append(mcnemar(a.values, b.loc[a.index].values)["p"])

    def ece_of(method):
        r = res[res["method"] == method]
        return ece_from_confidence(r["conf"].values, r["correct"].values)

    # metric 4: information of the greedy pair / best pair (clips where some pair is informative)
    pz = pairs[pairs["f_best"] > 1e-6]
    near = dict(greedy=float((pz["f_greedy"] / pz["f_best"]).mean()),
                greedy_pooled=float(pz["f_greedy"].sum() / pz["f_best"].sum()),
                eyes=float((pz["f_eyes"] / pz["f_best"]).mean()),
                below=float((pz["f_greedy"] < THRESH["near_optimal"] * pz["f_best"]).mean()),
                exact=float(pz["best_is_greedy"].mean()))

    row = dict(
        official=official_metrics(res, ordinal), vs_base=vs_base, near=near, acc_best_pair=per_clip(res, "best_pair")["correct"].mean(),
        acc_glance=glance["correct"].mean(), acc_eyes=eyes["correct"].mean(), acc_random=rand["correct"].mean(),
        acc_all9=all9["correct"].mean(),
        gain=gain, vs_random=vs_rand, all9_gain=total, recovered=recovered, mcnemar_p_max=max(mcn),
        spearman=float(ranking["closed_vs_realised"].mean()),
        spearman_mc=float(ranking["mc_vs_realised"].mean()),
        spearman_closed_mc=float(ranking["closed_vs_mc"].mean()),
        ece_eyes=ece_of(f"{E}_b2"), ece_eyes_noR=ece_of("eyes_noR_b2"), eyes_rule=E,
        acc_stop=stop["correct"].mean(), looks_stop=stop["passes"].mean() - glance["passes"].mean(),
        ece_glance=ece_of("glance"), ece_glance_noR=ece_of("glance_noR"),
        ece_all9=ece_of("all9"), ece_all9_noR=ece_of("all9_noR"),
        rho=np.mean([f["rho"] for f in fits]), lam=np.mean([f["shrink_lambda"] for f in fits]),
        n_clips=len(eyes), seeds=len(seeds),
    )
    by_T = pd.DataFrame({"T": eyes["T"], "gain": eyes["correct"] - glance["correct"]})
    by_T["T bin"] = pd.cut(by_T["T"], [0, 1, 3, 6, 10], labels=["1", "2-3", "4-6", "7-10"])
    return row, acc_seed, passes, by_T.groupby("T bin", observed=True)["gain"].agg(["mean", "size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iwildcam")
    ap.add_argument("--only", default="", help="comma-separated run folders to include (default: all)")
    ap.add_argument("--out", default="summary")
    args = ap.parse_args()
    root = paths.RESULTS / args.dataset
    lines = [f"# Results: {args.dataset}\n"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("seed*/clips.csv.gz"))]
    if args.only:
        keep = args.only.split(",")
        runs = [root / r for r in keep if (root / r) in runs]
    for enc_dir in runs if args.only else sorted(runs):
        row, acc_seed, passes, by_T = encoder_summary(enc_dir, ordinal=args.dataset == "ddr")
        g, v = row["gain"], row["vs_random"]
        ok = lambda b: "PASS" if b else "fail"
        lines += [
            f"## {enc_dir.name}  ({row['n_clips']} test clips, {row['seeds']} seeds, rho={row['rho']:.2f}, lam={row['lam']:.2f})\n",
            "| Metric | Value | Threshold | |", "|---|---|---|---|",
            f"| 1. Gain over glance | {100 * g[0]:+.2f} pts [{100 * g[1]:+.2f}, {100 * g[2]:+.2f}], McNemar p<={row['mcnemar_p_max']:.3g} | >= +2 pts | {ok(g[0] >= THRESH['gain'])} |",
            f"| 2. Beats random | {100 * v[0]:+.2f} pts [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}] | >= +1 pt | {ok(v[0] >= THRESH['beats_random'])} |",
            f"| 3. Gain recovered | {row['recovered']:.2f} (All-9 gains {100 * row['all9_gain']:+.2f} pts) | >= 2/3 | {ok(row['recovered'] >= THRESH['recovered'])} |",
            f"| 4. Near-optimal | greedy/best {row['near']['greedy']:.3f} (pooled {row['near']['greedy_pooled']:.3f}); adaptive eyes {row['near']['eyes']:.3f}; greedy finds the best pair in {100 * row['near']['exact']:.0f}% of clips, falls below 3/4 in {100 * row['near']['below']:.1f}% | >= 3/4 | {ok(row['near']['greedy'] >= THRESH['near_optimal'])} |",
            f"| 5. Ranking (Spearman) | closed {row['spearman']:.2f}, MC {row['spearman_mc']:.2f}, closed~MC {row['spearman_closed_mc']:.2f} | >= 0.3 | {ok(row['spearman'] >= THRESH['spearman'])} |",
            f"| 6. Calibration (ECE) | eyes {row['ece_eyes']:.3f} vs no-R {row['ece_eyes_noR']:.3f}; glance {row['ece_glance']:.3f} vs {row['ece_glance_noR']:.3f}; All-9 {row['ece_all9']:.3f} vs {row['ece_all9_noR']:.3f} | <= 0.05 | {ok(row['ece_eyes'] <= THRESH['ece'])} |",
            "",
            f"Accuracy: glance {100 * row['acc_glance']:.2f}, eyes(2) {100 * row['acc_eyes']:.2f} ({row['eyes_rule']}), "
            f"eyes with stopping {100 * row['acc_stop']:.2f} ({row['looks_stop']:.2f} looks/clip), random(2) {100 * row['acc_random']:.2f}, "
            f"best pair (oracle) {100 * row['acc_best_pair']:.2f}, All-9 {100 * row['acc_all9']:.2f}\n",
            "| Eyes(2) vs. baseline | baseline accuracy | passes/clip | Eyes(2) minus baseline (pts) |", "|---|---|---|---|",
            *[f"| {k} | {100 * a:.2f} | {n:.1f} | {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}] |"
              for k, (a, n, d) in row["vs_base"].items()],
            "",
            "| method | passes/clip | accuracy per seed (%) |", "|---|---|---|",
        ]
        off_rows = ["", "Official metrics (mean over seeds):", "",
                    "| method | macro-F1 |" + (" quadratic kappa |" if args.dataset == "ddr" else ""),
                    "|---|---|" + ("---|" if args.dataset == "ddr" else "")]
        off_rows += [f"| {m} | {f1:.3f} |" + (f" {k:.3f} |" if args.dataset == "ddr" else "")
                     for m, (f1, k) in row["official"].items()]
        for m in acc_seed.index:
            lines.append(f"| {m} | {passes[m]:.2f} | " + ", ".join(f"{100 * a:.2f}" for a in acc_seed.loc[m]) + " |")
        lines += off_rows
        lines += ["", "Gain of eyes(2) over glance by clip length:", "", "| frames T | gain (pts) | clips |", "|---|---|---|"]
        lines += [f"| {t} | {100 * r['mean']:+.2f} | {int(r['size'])} |" for t, r in by_T.iterrows()]
        lines.append("")

        base = acc_seed.loc["glance"].mean()
        for prefix, style in ((f"{row['eyes_rule']}_b", "-o"), ("random_b", "--s")):
            ms = ["glance"] + [f"{prefix}{b}" for b in (1, 2, 3, 4)]
            ax.plot([passes[m] for m in ms], [100 * (acc_seed.loc[m].mean() - base) for m in ms], style,
                    label=f"{enc_dir.name} {'random' if prefix == 'random_b' else 'eyes'}")
        ax.plot(passes["all9"], 100 * (acc_seed.loc["all9"].mean() - base), "k*", ms=10)
        ax.plot(passes["best_pair"], 100 * (acc_seed.loc["best_pair"].mean() - base), "kx", ms=8)
    ax.plot([], [], "k*", ms=10, label="All-9 (10T passes)")
    ax.plot([], [], "kx", ms=8, label="best pair (oracle, T+2)")
    ax.axhline(2, color="green", ls=":", label="metric 1 threshold")
    ax.set_xscale("log")
    ax.set_xticks([5, 7, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("encoder passes per clip")
    ax.set_ylabel("accuracy gain over glance (pts)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / ("accuracy_vs_passes.png" if args.out == "summary" else f"{args.out}.png"), dpi=160)
    (root / f"{args.out}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
