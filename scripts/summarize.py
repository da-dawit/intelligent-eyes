"""Step 4: turn per-clip results into the proposal's Table I and the accuracy-vs-passes figure.

    python scripts/summarize.py --dataset iwildcam

Reads results/<dataset>/<encoder>/seed*/ and writes results/<dataset>/summary.md and
results/<dataset>/accuracy_vs_passes.png. Thresholds are the ones fixed in the proposal.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eyes import paths
from eyes.metrics import camera_bootstrap, ece_from_confidence, mcnemar

THRESH = dict(gain=0.02, beats_random=0.01, recovered=2 / 3, spearman=0.3, ece=0.05)


def per_clip(res: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-clip correctness/confidence averaged over seeds and random draws."""
    r = res[res["method"] == method]
    return r.groupby("clip_id").agg(correct=("correct", "mean"), conf=("conf", "mean"),
                                    passes=("passes", "mean"), camera=("camera", "first"), T=("T", "first"))


def encoder_summary(enc_dir: Path):
    seeds = sorted(enc_dir.glob("seed*"))
    res = pd.concat([pd.read_csv(s / "clips.csv.gz").assign(seed=s.name) for s in seeds])
    ranking = pd.concat([pd.read_csv(s / "ranking.csv").assign(seed=s.name) for s in seeds])
    fits = [json.loads((s / "fit.json").read_text()) for s in seeds]
    res["correct"] = res["correct"].astype(bool).astype(float)

    acc_seed = res.groupby(["seed", "method"])["correct"].mean().unstack(0)
    passes = res.groupby("method")["passes"].mean()
    eyes, glance, rand, all9 = (per_clip(res, m) for m in ("eyes_b2", "glance", "random_b2", "all9"))

    def boot(diff, cams):
        return camera_bootstrap(diff.values, cams.values)

    gain = boot(eyes["correct"] - glance["correct"], eyes["camera"])
    vs_rand = boot(eyes["correct"] - rand["correct"], eyes["camera"])
    total = all9["correct"].mean() - glance["correct"].mean()
    recovered = (eyes["correct"].mean() - glance["correct"].mean()) / total if total > 0 else float("nan")
    mcn = []
    for s in res["seed"].unique():
        a = res[(res.seed == s) & (res.method == "eyes_b2")].set_index("clip_id")["correct"].astype(bool)
        b = res[(res.seed == s) & (res.method == "glance")].set_index("clip_id")["correct"].astype(bool)
        mcn.append(mcnemar(a.values, b.loc[a.index].values)["p"])

    def ece_of(method):
        r = res[res["method"] == method]
        return ece_from_confidence(r["conf"].values, r["correct"].values)

    row = dict(
        acc_glance=glance["correct"].mean(), acc_eyes=eyes["correct"].mean(), acc_random=rand["correct"].mean(),
        acc_all9=all9["correct"].mean(),
        gain=gain, vs_random=vs_rand, all9_gain=total, recovered=recovered, mcnemar_p_max=max(mcn),
        spearman=float(ranking["closed_vs_realised"].mean()),
        spearman_mc=float(ranking["mc_vs_realised"].mean()),
        spearman_closed_mc=float(ranking["closed_vs_mc"].mean()),
        ece_eyes=ece_of("eyes_b2"), ece_eyes_noR=ece_of("eyes_noR_b2"),
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
    args = ap.parse_args()
    root = paths.RESULTS / args.dataset
    lines = [f"# Results: {args.dataset}\n"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for enc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        row, acc_seed, passes, by_T = encoder_summary(enc_dir)
        g, v = row["gain"], row["vs_random"]
        ok = lambda b: "PASS" if b else "fail"
        lines += [
            f"## {enc_dir.name}  ({row['n_clips']} test clips, {row['seeds']} seeds, rho={row['rho']:.2f}, lam={row['lam']:.2f})\n",
            "| Metric | Value | Threshold | |", "|---|---|---|---|",
            f"| 1. Gain over glance | {100 * g[0]:+.2f} pts [{100 * g[1]:+.2f}, {100 * g[2]:+.2f}], McNemar p<={row['mcnemar_p_max']:.3g} | >= +2 pts | {ok(g[0] >= THRESH['gain'])} |",
            f"| 2. Beats random | {100 * v[0]:+.2f} pts [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}] | >= +1 pt | {ok(v[0] >= THRESH['beats_random'])} |",
            f"| 3. Gain recovered | {row['recovered']:.2f} (All-9 gains {100 * row['all9_gain']:+.2f} pts) | >= 2/3 | {ok(row['recovered'] >= THRESH['recovered'])} |",
            f"| 5. Ranking (Spearman) | closed {row['spearman']:.2f}, MC {row['spearman_mc']:.2f}, closed~MC {row['spearman_closed_mc']:.2f} | >= 0.3 | {ok(row['spearman'] >= THRESH['spearman'])} |",
            f"| 6. Calibration (ECE) | eyes {row['ece_eyes']:.3f} vs no-R {row['ece_eyes_noR']:.3f}; glance {row['ece_glance']:.3f} vs {row['ece_glance_noR']:.3f}; All-9 {row['ece_all9']:.3f} vs {row['ece_all9_noR']:.3f} | <= 0.05 | {ok(row['ece_eyes'] <= THRESH['ece'])} |",
            "",
            f"Accuracy: glance {100 * row['acc_glance']:.2f}, eyes(2) {100 * row['acc_eyes']:.2f}, random(2) {100 * row['acc_random']:.2f}, All-9 {100 * row['acc_all9']:.2f}\n",
            "| method | passes/clip | accuracy per seed (%) |", "|---|---|---|",
        ]
        for m in acc_seed.index:
            lines.append(f"| {m} | {passes[m]:.2f} | " + ", ".join(f"{100 * a:.2f}" for a in acc_seed.loc[m]) + " |")
        lines += ["", "Gain of eyes(2) over glance by clip length:", "", "| frames T | gain (pts) | clips |", "|---|---|---|"]
        lines += [f"| {t} | {100 * r['mean']:+.2f} | {int(r['size'])} |" for t, r in by_T.iterrows()]
        lines.append("")

        base = acc_seed.loc["glance"].mean()
        for prefix, style in (("eyes_b", "-o"), ("random_b", "--s")):
            ms = ["glance"] + [f"{prefix}{b}" for b in (1, 2, 3, 4)]
            ax.plot([passes[m] for m in ms], [100 * (acc_seed.loc[m].mean() - base) for m in ms], style,
                    label=f"{enc_dir.name} {'eyes' if prefix == 'eyes_b' else 'random'}")
        ax.plot(passes["all9"], 100 * (acc_seed.loc["all9"].mean() - base), "k*", ms=10)
    ax.axhline(2, color="green", ls=":", label="metric 1 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("encoder passes per clip")
    ax.set_ylabel("accuracy gain over glance (pts)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "accuracy_vs_passes.png", dpi=160)
    (root / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
