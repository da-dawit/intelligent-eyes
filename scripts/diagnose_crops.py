"""Failure-mode diagnosis: do crop votes hurt because most crops show only background?

    python scripts/diagnose_crops.py --encoder dinov2 --seed 0

Uses MegaDetector v4 animal boxes (iWildCam 2022 metadata, LILA) ONLY as a measuring stick:
a crop "contains the animal" if at least half of some animal box (conf >= 0.5) lies inside it.
Reports, on the test clips:
  1. how often each method's looks land on an animal (from results/<ds>/<enc>/seed<k>/clips.csv.gz)
  2. how good a single crop's vote is on animal crops vs background crops
  3. All-9 accuracy when background crops are silenced (vote set to 0) -- an oracle gate,
     i.e. how much a perfect "is the animal in this crop?" test could recover
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import build_clips, heldout_cameras, votes_for  # noqa: E402

from eyes import paths  # noqa: E402
from eyes.cache import load_features  # noqa: E402
from eyes.classifier import fit_temperature, train_probe  # noqa: E402
from eyes.controller import Model, run  # noqa: E402
from eyes.correlation import class_conditional_residuals, fit_A, fit_rho, shrink  # noqa: E402
from eyes.fit import choose_shrinkage  # noqa: E402
from eyes.geometry import boxes  # noqa: E402


def animal_masks(frames: pd.DataFrame, md_file: Path, conf: float = 0.5, cover: float = 0.5):
    """(N, 9) bool: crop k holds >= `cover` of some animal box; (N,) bool: frame has a box; (N,) covered by MD."""
    md = json.load(open(md_file))
    by_name = {im["file"].split("/")[-1]: im for im in md["images"]}
    crops = boxes()[1:]                                                    # (9, 4) x0 y0 x1 y1
    N = len(frames)
    in_crop, has_box, covered = np.zeros((N, 9), bool), np.zeros(N, bool), np.zeros(N, bool)
    for n, name in enumerate(frames["path"].str.split("/").str[-1].values):
        im = by_name.get(name)
        if im is None:
            continue
        covered[n] = True
        for d in im.get("detections", []):
            if d["category"] != "1" or d["conf"] < conf:
                continue
            x, y, w, h = d["bbox"]
            has_box[n] = True
            ix = np.clip(np.minimum(x + w, crops[:, 2]) - np.maximum(x, crops[:, 0]), 0, None)
            iy = np.clip(np.minimum(y + h, crops[:, 3]) - np.maximum(y, crops[:, 1]), 0, None)
            in_crop[n] |= ix * iy >= cover * max(w * h, 1e-9)
    return in_crop, has_box, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iwildcam")
    ap.add_argument("--encoder", default="dinov2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=str(paths.DATA / "raw" / "iwildcam_mdv4" / "iwildcam2022_mdv4_detections.json"))
    args = ap.parse_args()

    frames = pd.read_csv(paths.processed(args.dataset) / "frames.csv")
    feats = load_features(paths.features(args.dataset, args.encoder), len(frames))
    in_crop, has_box, covered = animal_masks(frames, Path(args.md))
    print(f"MD covers {covered.mean():.1%} of frames; {has_box.mean():.1%} have an animal box; "
          f"crops holding an animal: {in_crop[has_box].mean():.1%} of crops in those frames")

    # ---- 1. where do the looks land? (saved results)
    res = pd.read_csv(paths.RESULTS / args.dataset / args.encoder / f"seed{args.seed}" / "clips.csv.gz")
    rows_of = {cid: idx for cid, idx in frames.groupby("clip_id").indices.items()}
    order = {cid: np.sort(idx) for cid, idx in rows_of.items()}
    print("\n1. share of looks that land on an animal (frames with a box only)")
    for m in ["random_b2", "eyes_b2", "eyes_mc_b2", "motion_b2", "attention_b2", "best_pair"]:
        r = res[(res["method"] == m) & (res["draw"] == 0)]
        if r.empty:
            continue
        hit, tot = 0, 0
        for cid, looks in zip(r["clip_id"], r["looks"].fillna("")):
            idx = order[cid]
            for lk in filter(None, looks.split(";")):
                f, k = map(int, lk.split("-"))
                n = idx[f]
                if has_box[n]:
                    tot += 1
                    hit += in_crop[n, k - 1]
        print(f"   {m:14s} {hit / max(tot, 1):6.1%}  of {tot} looks")
    base = in_crop[has_box].mean()
    print(f"   {'any crop':14s} {base:6.1%}  (chance)")

    # ---- refit the classifiers exactly as run_eval does for this seed
    rng = np.random.default_rng(args.seed)
    frames["ok"] = feats["ok"].astype(bool)
    keep = frames.groupby("clip_id")["ok"].transform("all").values
    frames, in_crop, has_box = frames[keep].reset_index(drop=True), in_crop[keep], has_box[keep]
    train = frames[frames["split"] == "train"]
    held_cams = heldout_cameras(train, 0.10, rng)
    fit_df = train[~train["camera"].isin(held_cams)]
    classes = np.sort(fit_df["y_clip"].unique())
    to_idx = {y: i for i, y in enumerate(classes)}
    K = len(classes)
    frames = frames.assign(label=frames["y_clip"].map(to_idx).fillna(-1).astype(int))
    fit_df = frames.loc[fit_df.index]
    held_df = frames[(frames["split"] == "train") & frames["camera"].isin(held_cams) & (frames["label"] >= 0)]
    test_df = frames[(frames["split"] == "test") & (frames["label"] >= 0)]
    counts = np.bincount(fit_df["label"], minlength=K)
    log_prior = np.log(counts / counts.sum())
    D = feats["g"].shape[1]
    probe_g = train_probe(feats["g"][fit_df["row"].values].astype(np.float32), fit_df["label"].values, K, seed=args.seed)
    Xc = feats["crops"][fit_df["row"].values].reshape(-1, D)
    yc = np.repeat(fit_df["label"].values, 9)
    if len(Xc) > 300_000:
        pick = rng.choice(len(Xc), 300_000, replace=False)
        Xc, yc = Xc[pick], yc[pick]
    probe_c = train_probe(Xc.astype(np.float32), yc, K, seed=args.seed)
    hr, hy = held_df["row"].values, held_df["label"].values
    probe_g.temperature = fit_temperature(probe_g.logits(feats["g"][hr]), hy)
    probe_c.temperature = fit_temperature(probe_c.logits(feats["crops"][hr].reshape(-1, D)), np.repeat(hy, 9))

    # ---- 2. single-crop votes: animal crops vs background crops (test frames with a box)
    tr = test_df["row"].values
    v = votes_for(tr, feats, probe_g, probe_c, log_prior)                          # (N, 10, K)
    y = test_df["label"].values
    m_in, m_box = in_crop[test_df.index.values], has_box[test_df.index.values]
    top = v[:, 1:].argmax(-1) == y[:, None]                                         # (N, 9)
    gl = v[:, 0].argmax(-1) == y
    adv = v[:, 1:][np.arange(len(y)), :, y] - v[:, 1:].mean(-1)                     # true-class vote minus mean vote
    print("\n2. one crop alone (test frames with an animal box)")
    print(f"   glance                top-1 {gl[m_box].mean():6.1%}")
    print(f"   crop with the animal  top-1 {top[m_box][m_in[m_box]].mean():6.1%}   true-class vote advantage {adv[m_box][m_in[m_box]].mean():+.2f}")
    print(f"   background crop       top-1 {top[m_box][~m_in[m_box]].mean():6.1%}   true-class vote advantage {adv[m_box][~m_in[m_box]].mean():+.2f}")

    # ---- 3. All-9 with background crops silenced (oracle gate), same R as run_eval
    votes_h = votes_for(hr, feats, probe_g, probe_c, log_prior)
    resid = class_conditional_residuals(votes_h, hy, K)
    A_fit = fit_A(resid)
    rho, _ = fit_rho(resid, held_df["clip_id"].values, held_df["frame_idx"].values)
    held_df = held_df.reset_index(drop=True)
    zero_mu = np.zeros_like(votes_h)
    lam, _ = choose_shrinkage(build_clips(held_df, votes_h, zero_mu, feats["thumb"][hr]), A_fit, rho, log_prior, 1.0, seed=args.seed)
    model = Model(shrink(A_fit, lam), rho, log_prior, 1.0)
    test_df = test_df.reset_index(drop=True)
    gated = v.copy()
    gated[:, 1:][~m_in] = 0.0                                                       # silence background crops
    print("\n3. clip accuracy on the test clips")
    for name, vv, rule in [("glance", v, "eig_b0"), ("All-9", v, "all"), ("All-9, oracle gate", gated, "all")]:
        clips = build_clips(test_df, vv, np.zeros_like(vv), feats["thumb"][tr])
        acc = np.mean([run(c, model, rule="all" if rule == "all" else "eig", budget=0).pred == c.label for c in clips])
        print(f"   {name:20s} {acc:6.1%}")

    # ---- 4. the fix: crop classifier with an 'empty' class, labels from boxes (training data only)
    #      vote  l(y) = log( q(y|c) / pi(y) + q(empty|c) )
    fit_idx = fit_df.index.values
    lab = np.full((len(fit_idx), 9), -2)                                  # -2: not used
    box = has_box[fit_idx]
    lab[box] = np.where(in_crop[fit_idx][box], fit_df["label"].values[box, None], -2)
    lab[box & ~in_crop[fit_idx].any(1)] = -2
    # background: crops of boxed frames that hold no part of a box, and every crop of 'empty' frames
    free = crop_free_of_boxes(frames.iloc[fit_idx], Path(args.md))
    lab[box[:, None] & free] = K
    lab[fit_df["y_frame"].values == 0] = K
    Xc, yc = feats["crops"][fit_df["row"].values].reshape(-1, D), lab.reshape(-1)
    use = np.flatnonzero(yc >= 0)
    if len(use) > 300_000:
        use = rng.choice(use, 300_000, replace=False)
    print(f"\n4. crop classifier with an empty class: {np.mean(yc[use] < K):.1%} animal crops, "
          f"{np.mean(yc[use] == K):.1%} empty, {len(use)} rows")
    probe_e = train_probe(Xc[use].astype(np.float32), yc[use], K + 1, seed=args.seed)
    # temperature on held-out cameras, same labelling
    h_idx = frames.index[frames["row"].isin(hr)].values
    hl = np.full((len(hr), 9), -2)
    hb = has_box[h_idx]
    hl[hb] = np.where(in_crop[h_idx][hb], hy[hb, None], -2)
    hl[hb[:, None] & crop_free_of_boxes(frames.iloc[h_idx], Path(args.md))] = K
    hl[frames["y_frame"].values[h_idx] == 0] = K
    hx, hly = feats["crops"][hr].reshape(-1, D), hl.reshape(-1)
    probe_e.temperature = fit_temperature(probe_e.logits(hx[hly >= 0]), hly[hly >= 0])
    crop_prior = np.bincount(yc[use][yc[use] < K], minlength=K) + 1.0
    log_pc = np.log(crop_prior / crop_prior.sum())

    def mixture_votes(rows):
        N = len(rows)
        lq = probe_e.log_probs(feats["crops"][rows].reshape(-1, D)).reshape(N, 9, K + 1)
        lc = np.logaddexp(lq[..., :K] - log_pc, lq[..., K:])                  # log(q(y)/pi(y) + q(empty))
        lg = probe_g.log_probs(feats["g"][rows])
        return np.concatenate([(lg - log_prior)[:, None], lc], axis=1).astype(np.float32)

    v2 = mixture_votes(tr)
    top2 = v2[:, 1:].argmax(-1) == y[:, None]
    print(f"   crop with the animal  top-1 {top2[m_box][m_in[m_box]].mean():6.1%}   (was {top[m_box][m_in[m_box]].mean():.1%})")
    pe = np.exp(probe_e.log_probs(feats["crops"][tr].reshape(-1, D)).reshape(len(tr), 9, K + 1)[..., K])
    print(f"   q(empty): animal crops {pe[m_box][m_in[m_box]].mean():.2f}, background crops {pe[m_box][~m_in[m_box]].mean():.2f}")
    vh2 = mixture_votes(hr)
    resid2 = class_conditional_residuals(vh2, hy, K)
    A2 = fit_A(resid2)
    rho2, _ = fit_rho(resid2, held_df["clip_id"].values, held_df["frame_idx"].values)
    lam2, _ = choose_shrinkage(build_clips(held_df, vh2, np.zeros_like(vh2), feats["thumb"][hr]), A2, rho2, log_prior, 1.0, seed=args.seed)
    model2 = Model(shrink(A2, lam2), rho2, log_prior, 1.0)
    clips2 = build_clips(test_df, v2, np.zeros_like(v2), feats["thumb"][tr],
                         feats["attn"][tr] if "attn" in feats else None)
    acc_all = np.mean([run(c, model2, rule="all").pred == c.label for c in clips2])
    print(f"   All-9 with the new votes   {acc_all:6.1%}   (rho={rho2:.2f}, lam={lam2})")
    for rule in ["score:motion", "score:attention", "random"]:
        acc = np.mean([run(c, model2, rule=rule, budget=2, rng=np.random.default_rng(1)).pred == c.label for c in clips2])
        print(f"   glance + 2 looks by {rule.split(':')[-1]:9s} {acc:6.1%}")


def crop_free_of_boxes(frames: pd.DataFrame, md_file: Path, conf: float = 0.2) -> np.ndarray:
    """(N, 9) bool: crop k overlaps no animal box at all (boxes down to conf 0.2, to be safe)."""
    md = json.load(open(md_file))
    by_name = {im["file"].split("/")[-1]: im for im in md["images"]}
    crops = boxes()[1:]
    out = np.ones((len(frames), 9), bool)
    for n, name in enumerate(frames["path"].str.split("/").str[-1].values):
        for d in (by_name.get(name) or {}).get("detections", []):
            if d["category"] != "1" or d["conf"] < conf:
                continue
            x, y, w, h = d["bbox"]
            ix = np.minimum(x + w, crops[:, 2]) - np.maximum(x, crops[:, 0])
            iy = np.minimum(y + h, crops[:, 3]) - np.maximum(y, crops[:, 1])
            out[n] &= ~((ix > 0) & (iy > 0))
    return out


if __name__ == "__main__":
    main()
