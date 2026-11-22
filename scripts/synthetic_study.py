"""Synthetic study: the controller on clips with a known R (3 seeds), and failure mode
'wrong R' (the singular within-frame table A and how much shrinkage fixes it).

    python scripts/synthetic_study.py

Writes results/synthetic/{summary.md, figure.png}.
"""
import numpy as np

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eyes import paths
from eyes.controller import Model, run
from eyes.correlation import evidence_weights, shrink
from eyes.fit import choose_shrinkage
from eyes.geometry import overlap_correlation
from eyes.metrics import accuracy, ece
from eyes.synthetic import make_clips, predictor_sigma2

SEEDS = (0, 1, 2)
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
RHO = 0.8


def evaluate(clips, model, rule, budget, seed):
    rng = np.random.default_rng(seed)
    out = [run(c, model, rule, budget=budget, rng=rng) for c in clips]
    labels = np.array([c.label for c in clips])
    P = np.stack([o.p for o in out])
    return accuracy(P.argmax(1), labels), ece(P, labels), float(np.mean([o.passes for o in out]))


def main():
    A = overlap_correlation()
    out_dir = paths.RESULTS / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) accuracy/ECE against the shrinkage lam, per seed
    sweep = {lam: [] for lam in LAMBDAS}
    curves = {"eyes": [], "random": []}
    chosen, refs = [], []
    for seed in SEEDS:
        held, log_prior = make_clips(300, rho=RHO, A=A, seed=100 + seed)
        test, _ = make_clips(1000, rho=RHO, A=A, seed=200 + seed)
        s2 = predictor_sigma2(held)
        for lam in LAMBDAS:
            m = Model(shrink(A, lam), RHO, log_prior, s2)
            sweep[lam].append([evaluate(test, m, r, b, seed)[:2] for r, b in (("eig", 0), ("eig", 2), ("all", 0))])
        lam, _ = choose_shrinkage(held, A, RHO, log_prior, s2, lambdas=LAMBDAS, seed=seed)
        chosen.append(lam)
        m = Model(shrink(A, lam), RHO, log_prior, s2)
        naive = Model.independent(log_prior, s2)
        curves["eyes"].append([evaluate(test, m, "eig", b, seed) for b in range(5)])
        curves["random"].append([evaluate(test, m, "random", b, seed) for b in range(5)])
        refs.append(dict(all9=evaluate(test, m, "all", 0, seed), eyes_noR=evaluate(test, naive, "eig", 2, seed),
                         all9_noR=evaluate(test, naive, "all", 0, seed)))

    # summary.md
    S = {lam: np.array(v) for lam, v in sweep.items()}             # (seeds, 3 methods, acc/ece)
    lines = ["# Synthetic study (known R, 3 seeds x 1000 test clips)\n",
             f"Shrinkage lam chosen by held-out NLL per seed: {chosen}\n",
             "## Accuracy / ECE against shrinkage of the within-frame table A\n",
             "| lam | glance acc | glance ECE | eyes(2) acc | eyes(2) ECE | All-9 acc | All-9 ECE |", "|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        mu = S[lam].mean(0)
        lines.append(f"| {lam} | " + " | ".join(f"{100 * mu[i, 0]:.1f} | {mu[i, 1]:.3f}" for i in range(3)) + " |")
    E, Rn = np.array(curves["eyes"]), np.array(curves["random"])  # (seeds, budget, acc/ece/passes)
    lines += ["", "## Looks budget (lam chosen by held-out NLL)\n",
              "| looks | passes | eyes acc | random acc |", "|---|---|---|---|"]
    for b in range(5):
        lines.append(f"| {b} | {E[:, b, 2].mean():.1f} | {100 * E[:, b, 0].mean():.1f} ± {100 * E[:, b, 0].std():.1f} "
                     f"| {100 * Rn[:, b, 0].mean():.1f} ± {100 * Rn[:, b, 0].std():.1f} |")
    a9 = np.mean([r["all9"][0] for r in refs])
    g0 = E[:, 0, 0].mean()
    lines += ["", f"All-9: {100 * a9:.1f}% ({np.mean([r['all9'][2] for r in refs]):.1f} passes); "
              f"eyes(2) recovers {(E[:, 2, 0].mean() - g0) / (a9 - g0):.2f} of All-9's gain over the glance.",
              f"Eyes without R: acc {100 * np.mean([r['eyes_noR'][0] for r in refs]):.1f}, "
              f"ECE {np.mean([r['eyes_noR'][1] for r in refs]):.3f} "
              f"(with R: ECE {E[:, 2, 1].mean():.3f}). All-9 without R: ECE {np.mean([r['all9_noR'][1] for r in refs]):.3f}.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # figure.png
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    lams = np.linspace(0, 1, 41)
    W = np.array([evidence_weights(shrink(A, l), 1e-9) for l in lams])
    ax[0].plot(lams, W[:, 0], label="glance")
    ax[0].plot(lams, W[:, 5], label="centre crop")
    ax[0].plot(lams, W[:, 1], label="corner crop")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylim(-3, 2)
    ax[0].set_xlabel("shrinkage λ")
    ax[0].set_ylabel("weight in w = A_λ⁻¹ 1")
    ax[0].set_title("(a) weights of one frame's observations")
    ax[0].legend(fontsize=8)
    for i, name in enumerate(("glance", "eyes (2 looks)", "All-9")):
        ax[1].plot(LAMBDAS, [S[l][:, i, 1].mean() for l in LAMBDAS], "-o", ms=3, label=name)
    ax[1].set_xlabel("shrinkage λ")
    ax[1].set_ylabel("ECE")
    ax[1].set_title("(b) calibration")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[2].errorbar(E[0, :, 2], 100 * E[:, :, 0].mean(0), 100 * E[:, :, 0].std(0), fmt="-o", ms=3, label="eyes (EIG)")
    ax[2].errorbar(Rn[0, :, 2], 100 * Rn[:, :, 0].mean(0), 100 * Rn[:, :, 0].std(0), fmt="--s", ms=3, label="random looks")
    ax[2].plot(np.mean([r["all9"][2] for r in refs]), 100 * a9, "k*", ms=10, label="All-9")
    ax[2].set_xscale("log")
    ax[2].set_xticks([5, 7, 10, 20, 50])
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_xlabel("encoder passes per clip")
    ax[2].set_ylabel("accuracy (%)")
    ax[2].set_title("(c) accuracy vs passes")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure.png", dpi=160)


if __name__ == "__main__":
    main()
