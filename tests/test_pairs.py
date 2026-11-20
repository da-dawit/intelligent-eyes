"""Checks of the pair extension of Prop. 4.1 (eyes/pairs.py) against recomputation from scratch."""
import numpy as np
import pytest

from eyes.belief import scores
from eyes.controller import Model, candidate_terms, glance_indices, look_indices, run
from eyes.correlation import evidence_weights, shrink
from eyes.eig import eig_closed_form
from eyes.geometry import overlap_correlation
from eyes.pairs import all_pairs, pair_report
from eyes.synthetic import make_clips, predictor_sigma2


@pytest.fixture(scope="module")
def world():
    clips, log_prior = make_clips(40, T_max=4, rho=0.8, A=overlap_correlation(), seed=3)
    model = Model(shrink(overlap_correlation(), 0.5), 0.8, log_prior, predictor_sigma2(clips))
    return clips, model


def test_pair_shift_matches_recomputing_eq1(world):
    """Delta_B = g_B^T (l_B - a_B^T l_S) is exactly the change of the eq. (1) scores."""
    clips, model = world
    for clip in clips[:10]:
        P = all_pairs(clip, model)
        S = glance_indices(clip.T)
        votes = clip.votes.reshape(-1, clip.votes.shape[-1])
        R = model.R(clip.T)
        for n in np.random.default_rng(0).choice(len(P["f"]), 5, replace=False):
            B = S + [P["i"][n], P["j"][n]]
            direct = scores(votes[B], R[np.ix_(B, B)], model.log_prior) - P["s"]
            np.testing.assert_allclose(P["delta"][n], direct, atol=1e-6)


def test_pair_variance_uses_the_eq1_weights(world):
    """v_B = sigma2 g^T R_BB g, and g are the weights eq. (1) gives the two looks."""
    clips, model = world
    clip = clips[0]
    S = glance_indices(clip.T)
    P = all_pairs(clip, model)
    R = model.R(clip.T)
    n = 7
    B = [P["i"][n], P["j"][n]]
    w = evidence_weights(R[np.ix_(S + B, S + B)])[-2:]          # the looks' weights in eq. (1)
    K = clip.votes.shape[-1]
    mu = clip.mu.reshape(-1, K)
    votes = clip.votes.reshape(-1, K)
    # Delta is linear in l_B with coefficients w, so its mean and variance follow directly
    a_B = np.linalg.solve(R[np.ix_(S, S)], R[np.ix_(S, B)])
    m = w @ (mu[B] - a_B.T @ votes[S])
    v = model.sigma2 * w @ R[np.ix_(B, B)] @ w
    p = np.exp(P["s"] - np.logaddexp.reduce(P["s"]))
    np.testing.assert_allclose(P["f"][n], eig_closed_form(p, m[None], np.array([v]))[0], rtol=1e-6)


def test_independent_pair_adds_single_look_terms(world):
    """With R = I nothing repeats, so a pair shifts the scores by the sum of its two looks' shifts."""
    clips, model = world
    naive = Model.independent(model.log_prior, model.sigma2)
    clip = clips[1]
    S, C = glance_indices(clip.T), look_indices(clip.T)
    s, m1, v1, shift1 = candidate_terms(clip, naive, S, [], C)
    P = all_pairs(clip, naive)
    pos = {c: n for n, c in enumerate(C)}
    for n in range(0, len(P["f"]), 11):
        a, b = pos[P["i"][n]], pos[P["j"][n]]
        np.testing.assert_allclose(P["delta"][n], shift1[a] + shift1[b], rtol=1e-5)   # ridge 1e-6


def test_eyes_pair_posterior_matches_the_controller(world):
    """The adaptive controller's final belief equals the pair formula at the pair it took."""
    clips, model = world
    for clip in clips[:10]:
        o = run(clip, model, "eig", budget=2)
        P = all_pairs(clip, model)
        idx = sorted(f * 10 + k for f, k in o.looks)
        n = np.flatnonzero((P["i"] == idx[0]) & (P["j"] == idx[1]))[0]
        s = P["s"] + P["delta"][n]
        np.testing.assert_allclose(o.p, np.exp(s - np.logaddexp.reduce(s)), atol=1e-8)
        r = pair_report(clip, model, idx)
        assert r["f_greedy"] <= r["f_best"] + 1e-12 and r["f_eyes"] <= r["f_best"] + 1e-12
        assert r["oracle_p_true"] >= o.p[clip.label] - 1e-9        # the oracle is an upper bound
