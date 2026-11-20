"""Step 3: fit on held-out training cameras, then run every method on the test clips.

    python scripts/run_eval.py --dataset iwildcam --encoder dinov2 --seeds 0 1 2

Per seed (the seed sets the camera split, classifier training, MC samples, random looks):
  1. split training cameras 90 / 10: classifiers learn on the 90, everything else is
     fitted on the 10 held out (temperatures, look predictor, sigma2, A, rho, lam, eps)
  2. turn every cached view into a vote  l = log q - log pi
  3. run each method on every test clip and record the outcome per clip
  4. score every pair of looks of every test clip: metric 4 and the best-pair reference
B1 (every frame at 2x resolution) runs when a high-res cache exists (<encoder>_hr): its own
probe, temperature and rho, frames combined by eq. (1) with T_rho; counted as 4 passes a frame.
Outputs results/<dataset>/<encoder>/seed<k>/{clips.csv.gz, ranking.csv, pairs.csv.gz, fit.json}.
"""
import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd

from eyes import paths
from eyes.belief import scores, softmax
from eyes.boxes import crop_labels, load_detections, mixture_votes
from eyes.cache import load_features
from eyes.classifier import fit_temperature, train_probe
from eyes.controller import Clip, Model, run
from eyes.correlation import class_conditional_residuals, fit_A, fit_rho, frame_correlation, shrink
from eyes.fit import choose_eps, choose_shrinkage
from eyes.geometry import overlap_correlation
from eyes.metrics import spearman
from eyes.pairs import pair_report
from eyes.predictor import fit_ridge, out_of_fold
from eyes.views import motion_scores

BUDGETS = (1, 2, 3, 4)
RANDOM_DRAWS = 10


def heldout_cameras(train: pd.DataFrame, frac: float, rng) -> set:
    """Random cameras until they hold `frac` of the training clips."""
    per_cam = train.groupby("camera")["clip_id"].nunique()
    target, held, n = frac * train["clip_id"].nunique(), [], 0
    for cam in rng.permutation(per_cam.index.values):
        if n >= target:
            break
        held.append(cam)
        n += per_cam[cam]
    return set(held)


def votes_for(rows, feats, probe_g, probe_c, log_prior, log_pi_crop=None, shape=(0.0, 1.0)):
    """(N, 10, K) votes: glance then the 9 crops, l = log q - log pi.

    If the crop classifier has an extra 'empty' class (log_pi_crop given), crop votes are
    the mixture votes log(q(y|c)/pi(y) + q(empty|c)) of eyes/boxes.py, shaped by
    (eps, gain): pi is softened to (1 - eps) pi + eps / K, so that a little probability
    leaking onto a rare species is not multiplied by 1 / pi(y), and the votes are scaled by
    `gain`. Both are fitted on held-out cameras (choose_vote_shape).
    """
    N = len(rows)
    D = feats["g"].shape[1]
    lg = probe_g.log_probs(feats["g"][rows]) - log_prior
    lc = probe_c.log_probs(feats["crops"][rows].reshape(-1, D)).reshape(N, 9, -1)
    if log_pi_crop is None:
        lc = lc - log_prior
    else:
        eps, gain = shape
        K = lc.shape[-1] - 1
        lc = gain * mixture_votes(lc, np.log((1 - eps) * np.exp(log_pi_crop) + eps / K))
    return np.concatenate([lg[:, None], lc], axis=1).astype(np.float32)


def crop_gate(rows, feats, probe_c, K):
    """(N, 9) bool, gated fusion: once a look is taken, its own crop classifier decides whether
    it shows the animal (q(empty | crop) < 1/2). Looks that show only background stay out of eq. (1)."""
    D = feats["g"].shape[1]
    lq = probe_c.log_probs(feats["crops"][rows].reshape(-1, D)).reshape(len(rows), 9, K + 1)
    return np.exp(lq[..., K]) < 0.5


def gate_mask(gate):
    """(N, 10) observations that enter eq. (1): glances always, looks when gated in."""
    return None if gate is None else np.concatenate([np.ones((len(gate), 1), bool), gate], axis=1)


def choose_vote_shape(held_df, feats, probe_g, probe_c, log_prior, log_pi_crop, seed,
                      eps_grid=(0.0, 0.1, 0.3), gain_grid=(1.0, 0.7, 0.5, 0.3), gate=None):
    """(eps, gain) with the lowest held-out NLL of eq. (1) (R refitted for each, lam chosen)."""
    hr, hy, K = held_df["row"].values, held_df["label"].values, len(log_prior)
    df = held_df.reset_index(drop=True)
    mask = gate_mask(gate)
    table = {}
    for eps in eps_grid:
        for gain in gain_grid:
            v = votes_for(hr, feats, probe_g, probe_c, log_prior, log_pi_crop, (eps, gain))
            r = class_conditional_residuals(v, hy, K, mask)
            rho, _ = fit_rho(r, df["clip_id"].values, df["frame_idx"].values, mask=mask)
            lam, nll = choose_shrinkage(build_clips(df, v, np.zeros_like(v), feats["thumb"][hr], gate=gate),
                                        fit_A(r, mask), rho, log_prior, 1.0, seed=seed)
            table[f"{eps},{gain}"] = nll[lam]
    best = min(table, key=table.get)
    return tuple(map(float, best.split(","))), table


def where_features(frames, feats):
    """(N, 9, F) cheap 'where is something' cues per crop, from the glance and thumbnails only:
    log motion (B4's score) and, when cached, the share of the glance's CLS attention (B3's)."""
    motion = np.zeros((len(frames), 9), np.float32)
    for _, idx in frames.groupby("clip_id").indices.items():
        idx = np.sort(idx)
        motion[idx] = motion_scores(feats["thumb"][frames["row"].values[idx]])
    cues = [np.log(motion + 1e-3)]
    if "attn" in feats:
        cues.append(np.log(feats["attn"][frames["row"].values].astype(np.float32) + 1e-3))
    return np.stack(cues, axis=2)


def predictor_inputs(rows, feats, preview=None):
    """[u_bar_fk ; g_f] for the 9 crops of each frame -> (N * 9, 2D).

    preview: optional (probe_u, log_pi_crop, where). probe_u classifies a crop's preview
    u_bar_fk (the glance's own tokens over that crop, plus the `where` cues if given: a
    row-indexed (N_all, 9, F) array) as species or 'empty', trained on the same box labels as
    the crop classifier; its mixture vote -- the look's vote as guessed from the glance alone --
    is appended, so the ridge of eq. (2) only has to correct it.
    """
    u = feats["ubar"][rows].astype(np.float32)
    g = np.repeat(feats["g"][rows].astype(np.float32)[:, None], 9, axis=1)
    parts = [u, g]
    N, D = u.shape[0], u.shape[2]
    if preview is not None:
        probe_u, log_pi_crop, where = preview
        x = u if where is None else np.concatenate([u, where[rows]], axis=2)
        lq = probe_u.log_probs(x.reshape(N * 9, -1)).reshape(N, 9, -1)
        parts.append(mixture_votes(lq, log_pi_crop).astype(np.float32))
    return np.concatenate(parts, axis=2).reshape(N * 9, -1)


def animal_branch(rows, feats, preview, shape, probe_w=None):
    """From the glance preview: P(crop shows the animal) (N, 9) and the vote it would give if
    it does (N, 9, K), shaped like the real crop votes (eig_mix). probe_w: optional binary
    'is the animal in this crop?' classifier on u_bar, used for P(animal) when given."""
    probe_u, log_pi_crop, where = preview
    u = feats["ubar"][rows].astype(np.float32)
    x = u if where is None else np.concatenate([u, where[rows]], axis=2)
    N, K = len(rows), len(log_pi_crop)
    lq = probe_u.log_probs(x.reshape(N * 9, -1)).reshape(N, 9, K + 1)
    p_species = np.clip(1.0 - np.exp(lq[..., K]), 1e-6, 1.0)
    eps, gain = shape
    log_pi = np.log((1 - eps) * np.exp(log_pi_crop) + eps / K)
    mu_animal = gain * (lq[..., :K] - np.log(p_species)[..., None] - log_pi)
    p_animal = p_species
    if probe_w is not None:
        p_animal = np.exp(probe_w.log_probs(u.reshape(N * 9, -1))[:, 1]).reshape(N, 9)
    return p_animal.astype(np.float64), mu_animal.astype(np.float64)


def build_clips(df, votes, mu, thumbs, attn=None, branch=None, gate=None):
    """Group consecutive frame rows of `df` (positional, aligned with the arrays) into Clips.

    look_scores: 'motion' (B4) from thumbnails; 'attention' (B3) when the cache has it.
    branch: optional (p_animal, mu_animal) arrays for the eig_mix rule.
    gate: optional (N, 9) bool for gated fusion.
    """
    clips = []
    for cid, idx in df.groupby("clip_id", sort=False).indices.items():
        idx = np.sort(idx)
        scores = {"motion": motion_scores(thumbs[idx])}
        if attn is not None:
            scores["attention"] = attn[idx].astype(np.float64)
        clips.append(Clip(votes=votes[idx].astype(np.float64), mu=mu[idx].astype(np.float64),
                          label=int(df["label"].iloc[idx[0]]), clip_id=cid, camera=int(df["camera"].iloc[idx[0]]),
                          look_scores=scores,
                          p_animal=None if branch is None else branch[0][idx],
                          mu_animal=None if branch is None else branch[1][idx],
                          gate=None if gate is None else gate[idx]))
    return clips


def crop_training_labels(df, content, K):
    """(N*9,) crop labels: species index if the crop holds the animal, K if empty, -1 unusable.

    content is indexed by the frame's original row number (frames.csv `row`).
    """
    c = content[df["row"].values]
    lab = np.where(c == 1, df["label"].values[:, None], np.where(c == 0, K, -1))
    return lab.reshape(-1)


def run_seed(frames, feats, hires, seed, out_dir, args, content=None):
    t0 = time.time()
    where_by_row = None
    if args.predictor == "preview_where":
        where_by_row = np.zeros((int(frames["row"].max()) + 1, 9, 2 if "attn" in feats else 1), np.float32)
        where_by_row[frames["row"].values] = where_features(frames, feats)
    rng = np.random.default_rng(seed)
    ok_clip = frames.groupby("clip_id")["ok"].transform("all")
    frames = frames[ok_clip].reset_index(drop=True)

    train = frames[frames["split"] == "train"]
    held_cams = heldout_cameras(train, args.heldout_frac, rng)
    fit_df = train[~train["camera"].isin(held_cams)]
    classes = np.sort(fit_df["y_clip"].unique())
    to_idx = {y: i for i, y in enumerate(classes)}
    K = len(classes)
    frames = frames.assign(label=frames["y_clip"].map(to_idx).fillna(-1).astype(int))
    held_df = frames[(frames["split"] == "train") & frames["camera"].isin(held_cams) & (frames["label"] >= 0)]
    test_all = frames[frames["split"] == "test"]
    test_df = test_all[test_all["label"] >= 0]
    fit_df = frames.loc[fit_df.index]

    # ---- classifiers (glance, crop) on the 90%
    counts = np.bincount(fit_df["label"], minlength=K)
    log_prior = np.log(counts / counts.sum())
    D = feats["g"].shape[1]
    probe_g = train_probe(feats["g"][fit_df["row"].values].astype(np.float32), fit_df["label"].values, K, seed=seed)
    Xc = feats["crops"][fit_df["row"].values].reshape(-1, D)
    if content is None:                                     # every crop gets the clip's species
        yc, n_out, log_pi_crop = np.repeat(fit_df["label"].values, 9), K, None
        use = np.arange(len(yc))
    else:                                                   # species if the crop holds the animal, else 'empty'
        yc, n_out = crop_training_labels(fit_df, content, K), K + 1
        use = np.flatnonzero(yc >= 0)
        animal = np.bincount(yc[use][yc[use] < K], minlength=K) + 1.0
        log_pi_crop = np.log(animal / animal.sum())
    if len(use) > args.max_crop_rows:
        use = rng.choice(use, args.max_crop_rows, replace=False)
    probe_c = train_probe(Xc[use].astype(np.float32), yc[use], n_out, seed=seed)

    # ---- temperatures on held-out cameras (A3)
    hr = held_df["row"].values
    hy = held_df["label"].values
    probe_g.temperature = fit_temperature(probe_g.logits(feats["g"][hr]), hy)
    lc_h = probe_c.logits(feats["crops"][hr].reshape(-1, D))
    yh_c = np.repeat(hy, 9) if content is None else crop_training_labels(held_df, content, K)
    ok_c = yh_c >= 0
    probe_c.temperature = fit_temperature(lc_h[ok_c], yh_c[ok_c])
    acc_g = float(np.mean(probe_g.logits(feats["g"][hr]).argmax(1) == hy))
    acc_c = float(np.mean(lc_h[ok_c].argmax(1) == yh_c[ok_c]))

    # ---- gated fusion: a look enters eq. (1) only if its crop shows the animal
    gate_h = None
    if args.fusion == "gated":
        if content is None:
            raise SystemExit("--fusion gated needs --crop-labels boxes")
        gate_h = crop_gate(hr, feats, probe_c, K)

    # ---- votes, look predictor (eq. 2) on held-out
    shape, shape_table = (0.0, 1.0), {}
    if content is not None and args.vote_shape == "fit":
        shape, shape_table = choose_vote_shape(held_df, feats, probe_g, probe_c, log_prior, log_pi_crop, seed,
                                               gate=gate_h)
    votes_h = votes_for(hr, feats, probe_g, probe_c, log_prior, log_pi_crop, shape)
    preview = None
    if args.predictor in ("preview", "preview_where"):
        if content is None:
            raise SystemExit("--predictor preview needs --crop-labels boxes")
        where = where_by_row if args.predictor == "preview_where" else None

        def preview_x(rows):
            u = feats["ubar"][rows].astype(np.float32)
            return (u if where is None else np.concatenate([u, where[rows]], axis=2)).reshape(len(rows) * 9, -1)

        probe_u = train_probe(preview_x(fit_df["row"].values)[use], yc[use], K + 1, seed=seed)
        lu_h = probe_u.logits(preview_x(hr))
        probe_u.temperature = fit_temperature(lu_h[ok_c], yh_c[ok_c])
        preview = (probe_u, log_pi_crop, where)
    Xh = predictor_inputs(hr, feats, preview)
    Yh = votes_h[:, 1:].reshape(-1, K)
    groups = np.repeat(held_df["camera"].values, 9)
    pred = fit_ridge(Xh, Yh, groups, seed=seed)
    mu_h = np.zeros_like(votes_h)
    mu_h[:, 1:] = out_of_fold(Xh, Yh, groups, pred.lam, seed=seed).reshape(len(hr), 9, K)

    # ---- repetition R = T_rho (x) A_lam on held-out (eq. 1)
    mask_h = gate_mask(gate_h)
    resid = class_conditional_residuals(votes_h, hy, K, mask_h)
    A_fit = fit_A(resid, mask_h)
    rho, c_hat = fit_rho(resid, held_df["clip_id"].values, held_df["frame_idx"].values, mask=mask_h)

    probe_w = None                       # 'is the animal in this crop?' from the glance (gated runs)
    if preview is not None and args.fusion == "gated":
        lab = content[fit_df["row"].values].reshape(-1)
        okw = np.flatnonzero(lab >= 0)
        if len(okw) > args.max_crop_rows:
            okw = rng.choice(okw, args.max_crop_rows, replace=False)
        Xu = feats["ubar"][fit_df["row"].values].reshape(-1, D)
        probe_w = train_probe(Xu[okw].astype(np.float32), lab[okw].astype(int), 2, seed=seed)
        lab_h = content[hr].reshape(-1)
        okh = lab_h >= 0
        probe_w.temperature = fit_temperature(probe_w.logits(feats["ubar"][hr].reshape(-1, D))[okh], lab_h[okh].astype(int))
    held_df = held_df.reset_index(drop=True)
    branch_h = animal_branch(hr, feats, preview, shape, probe_w) if preview is not None else None
    held_clips = build_clips(held_df, votes_h, mu_h, feats["thumb"][hr], branch=branch_h, gate=gate_h)
    lam, nll_table = choose_shrinkage(held_clips, A_fit, rho, log_prior, pred.sigma2, seed=seed)
    lam_ov, nll_ov = choose_shrinkage(held_clips, overlap_correlation(), rho, log_prior, pred.sigma2, seed=seed)
    model = Model(shrink(A_fit, lam), rho, log_prior, pred.sigma2)
    models = {"R": model, "I": Model.independent(log_prior, pred.sigma2),
              "overlap": Model(shrink(overlap_correlation(), lam_ov), rho, log_prior, pred.sigma2)}
    eps, eps_table = choose_eps(held_clips, model, budget=max(BUDGETS), seed=seed)
    if preview is not None:
        eps_mix, _ = choose_eps(held_clips, model, budget=max(BUDGETS), seed=seed, rule="eig_mix")

    # ---- B1: the whole frame at 2x resolution, its own probe / temperature / rho
    if hires is not None:
        probe_h = train_probe(hires[fit_df["row"].values].astype(np.float32), fit_df["label"].values, K, seed=seed)
        probe_h.temperature = fit_temperature(probe_h.logits(hires[hr]), hy)
        lv = (probe_h.log_probs(hires[hr]) - log_prior)[:, None]                  # (N, 1, K)
        rho_h, _ = fit_rho(class_conditional_residuals(lv, hy, K), held_df["clip_id"].values,
                           held_df["frame_idx"].values)
        acc_h = float(np.mean(probe_h.logits(hires[hr]).argmax(1) == hy))
    print(f"  seed {seed}: fitted in {time.time() - t0:.0f}s  K={K} rho={rho:.3f} lam={lam} "
          f"sigma2={pred.sigma2:.3f} eps={eps} T_g={probe_g.temperature:.2f} T_c={probe_c.temperature:.2f}", flush=True)

    # ---- test clips
    tr = test_df["row"].values
    votes_t = votes_for(tr, feats, probe_g, probe_c, log_prior, log_pi_crop, shape)
    mu_t = np.zeros_like(votes_t)
    mu_t[:, 1:] = pred.predict(predictor_inputs(tr, feats, preview)).reshape(len(tr), 9, K)
    test_df = test_df.reset_index(drop=True)
    branch_t = animal_branch(tr, feats, preview, shape, probe_w) if preview is not None else None
    gate_t = crop_gate(tr, feats, probe_c, K) if gate_h is not None else None
    clips = build_clips(test_df, votes_t, mu_t, feats["thumb"][tr], feats["attn"][tr] if "attn" in feats else None,
                        branch=branch_t, gate=gate_t)

    methods = [("glance", "eig", 0, "R", 0.0, 1), ("glance_noR", "eig", 0, "I", 0.0, 1)]
    methods += [(f"eyes_b{b}", "eig", b, "R", 0.0, 1) for b in BUDGETS]
    methods += [("eyes_stop", "eig", max(BUDGETS), "R", eps, 1),
                ("eyes_mc_b2", "eig_mc", 2, "R", 0.0, 1),
                ("eyes_noR_b2", "eig", 2, "I", 0.0, 1),
                ("eyes_overlapR_b2", "eig", 2, "overlap", 0.0, 1),
                ("motion_b2", "score:motion", 2, "R", 0.0, 1),
                *([("attention_b2", "score:attention", 2, "R", 0.0, 1)] if "attn" in feats else []),
                ("all9", "all", 0, "R", 0.0, 1),
                ("all9_noR", "all", 0, "I", 0.0, 1)]
    methods += [(f"random_b{b}", "random", b, "R", 0.0, RANDOM_DRAWS) for b in BUDGETS]
    if preview is not None:
        methods += [(f"eyesmix_b{b}", "eig_mix", b, "R", 0.0, 1) for b in BUDGETS]
        methods += [("eyesmix_stop", "eig_mix", max(BUDGETS), "R", eps_mix, 1)]

    rows, eyes_looks = [], {}
    for name, rule, budget, mkey, e, draws in methods:
        t1 = time.time()
        for d in range(draws):
            mrng = np.random.default_rng([seed, d, zlib.crc32(name.encode())])
            for c in clips:
                o = run(c, models[mkey], rule=rule, budget=budget, eps=e, rng=mrng)
                if name == "eyes_b2":
                    eyes_looks[c.clip_id] = [f * 10 + k for f, k in o.looks]
                rows.append((name, d, c.clip_id, c.camera, c.T, c.label, o.pred, o.pred == c.label,
                             float(o.p.max()), float(o.p[c.label]), o.passes,
                             ";".join(f"{f}-{k}" for f, k in o.looks), o.stopped))
        print(f"    {name:18s} {time.time() - t1:6.1f}s", flush=True)

    # ---- metric 4 + best-pair reference: every pair of looks, from the glance state
    t1, pr = time.time(), []
    for c in clips:
        r = pair_report(c, model, eyes_looks[c.clip_id])
        pr.append(dict(clip_id=c.clip_id, T=c.T, **{k: v for k, v in r.items() if k != "oracle_looks"}))
        rows.append(("best_pair", 0, c.clip_id, c.camera, c.T, c.label, r["oracle_pred"], r["oracle_pred"] == c.label,
                     r["oracle_conf"], r["oracle_p_true"], c.T + 2,
                     ";".join(f"{i // 10}-{i % 10}" for i in r["oracle_looks"]), False))
    pairs = pd.DataFrame(pr)
    print(f"    {'pairs (metric 4)':18s} {time.time() - t1:6.1f}s", flush=True)

    if hires is not None:
        lv_t = probe_h.log_probs(hires[tr]) - log_prior                          # (N, K)
        for cid, idx in test_df.groupby("clip_id", sort=False).indices.items():
            idx = np.sort(idx)
            p = softmax(scores(lv_t[idx], frame_correlation(len(idx), rho_h), log_prior))
            y = int(test_df["label"].iloc[idx[0]])
            rows.append(("hires", 0, cid, int(test_df["camera"].iloc[idx[0]]), len(idx), y, int(p.argmax()),
                         int(p.argmax()) == y, float(p.max()), float(p[y]), 4 * len(idx), "", False))
    res = pd.DataFrame(rows, columns=["method", "draw", "clip_id", "camera", "T", "label", "pred", "correct",
                                      "conf", "p_true", "passes", "looks", "stopped"])

    # ---- metric 5: does predicted EIG rank looks like their realised effect?
    rk = []
    for c in clips[: args.ranking_clips]:
        o = run(c, model, "eig", budget=1, record_first=True, rng=np.random.default_rng(seed))
        f = o.first_step
        rk.append((c.clip_id, c.T, spearman(f["closed"], f["realised_kl"]), spearman(f["mc"], f["realised_kl"]),
                   spearman(f["closed"], f["mc"]), spearman(f["closed"], f["realised_true_gain"])))
    ranking = pd.DataFrame(rk, columns=["clip_id", "T", "closed_vs_realised", "mc_vs_realised",
                                        "closed_vs_mc", "closed_vs_true_gain"])

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "clips.csv.gz", index=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    pairs.to_csv(out_dir / "pairs.csv.gz", index=False)
    fit = dict(seed=seed, K=K, n_heldout_cameras=len(held_cams), n_heldout_clips=len(held_clips),
               n_test_clips=len(clips), n_test_clips_unseen_label=int(test_all.loc[test_all["label"] < 0, "clip_id"].nunique()),
               glance_probe_heldout_acc=acc_g, crop_probe_heldout_acc=acc_c,
               T_glance=probe_g.temperature, T_crop=probe_c.temperature,
               predictor_lambda=pred.lam, sigma2=pred.sigma2, predictor_cv=pred.cv_mse,
               rho=rho, rho_lag_corr=list(map(float, c_hat)), A_fit=A_fit.round(4).tolist(),
               shrink_lambda=lam, shrink_nll=nll_table, overlap_shrink_lambda=lam_ov, overlap_shrink_nll=nll_ov,
               eps=eps, eps_table={str(k): v for k, v in eps_table.items()}, seconds=time.time() - t0)
    fit.update(vote_shape=dict(eps=shape[0], gain=shape[1]), vote_shape_nll=shape_table)
    if hires is not None:
        fit.update(hires_probe_heldout_acc=acc_h, T_hires=probe_h.temperature, rho_hires=rho_h)
    (out_dir / "fit.json").write_text(json.dumps(fit, indent=2, default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iwildcam")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--heldout-frac", type=float, default=0.10)
    ap.add_argument("--max-crop-rows", type=int, default=300_000)
    ap.add_argument("--ranking-clips", type=int, default=1000)
    ap.add_argument("--crop-labels", choices=["clip", "boxes"], default="clip")
    ap.add_argument("--fusion", choices=["all", "gated"], default="all",
                    help="gated: a look enters eq. (1) only if its own crop classifier says it shows the animal")
    ap.add_argument("--vote-shape", choices=["fit", "none"], default="fit",
                    help="with --crop-labels boxes: fit the prior softening and gain of crop votes on held-out")
    ap.add_argument("--predictor", choices=["ridge", "preview", "preview_where"], default="ridge",
                    help="preview: add the glance-preview classifier's guess of the vote to eq. (2); "
                         "preview_where: the preview classifier also sees motion and attention cues")
    ap.add_argument("--md", default=str(paths.DATA / "raw" / "iwildcam_mdv4" / "iwildcam2022_mdv4_detections.json"))
    args = ap.parse_args()

    frames = pd.read_csv(paths.processed(args.dataset) / "frames.csv")
    feats = load_features(paths.features(args.dataset, args.encoder), len(frames))
    frames["ok"] = feats["ok"].astype(bool)
    hr_dir = paths.features(args.dataset, args.encoder + "_hr")
    hires = None
    if hr_dir.exists() and any(hr_dir.glob("chunk_*.npz")):
        h = load_features(hr_dir, len(frames))
        hires = h["g"]
        frames["ok"] &= h["ok"].astype(bool)
    print(f"{args.dataset}/{args.encoder}: {len(frames)} frames, {frames['clip_id'].nunique()} clips, "
          f"{(~frames['ok']).sum()} unreadable frames, high-res B1: {'yes' if hires is not None else 'no'}")
    content, tag = None, args.encoder
    if args.crop_labels == "boxes":
        content = crop_labels(frames, load_detections(args.md))
        tag = (f"{args.encoder}-boxes" + {"ridge": "", "preview": "-preview", "preview_where": "-where"}[args.predictor]
               + ("-shaped" if args.vote_shape == "fit" else "") + ("-gated" if args.fusion == "gated" else ""))
        print(f"crop labels from boxes: {np.mean(content == 1):.1%} animal, {np.mean(content == 0):.1%} empty, "
              f"{np.mean(content == -1):.1%} unusable")
    for seed in args.seeds:
        run_seed(frames, feats, hires, seed, paths.RESULTS / args.dataset / tag / f"seed{seed}", args, content)


if __name__ == "__main__":
    main()
