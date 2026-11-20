"""Each test checks one numbered result of derivations.pdf against the code."""
import numpy as np
import pytest

from eyes.belief import add_one_look_terms, posterior, scores, softmax
from eyes.correlation import (ar1_effective_frames, class_conditional_residuals, clip_correlation,
                              effective_count, evidence_weights, fit_A, fit_rho, frame_correlation, shrink)
from eyes.eig import eig_closed_form, eig_monte_carlo, fisher, kl, trace_fisher, var_under_p
from eyes.geometry import N_POS, overlap_correlation, token_masks
from eyes.predictor import condition_on_looks, fit_ridge

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- Sec. 3: eq. (1)

def test_two_votes_weight_one_over_one_plus_r():
    """Example after Prop. 3.1: w = 1/(1+r) each, n_eff = 2/(1+r)."""
    for r in (0.0, 0.3, 0.9):
        R = np.array([[1, r], [r, 1]])
        np.testing.assert_allclose(evidence_weights(R, ridge=0), [1 / (1 + r)] * 2)
        assert effective_count(R, ridge=0) == pytest.approx(2 / (1 + r))


@pytest.mark.parametrize("T,rho", [(1, 0.5), (2, 0.3), (5, 0.5), (10, 0.9), (7, 0.0)])
def test_ar1_effective_frames(T, rho):
    """Prop. 3.3: T glances count as (T - (T-2) rho) / (1 + rho) frames."""
    assert effective_count(frame_correlation(T, rho), ridge=0) == pytest.approx(ar1_effective_frames(T, rho))


def test_ten_still_frames_count_as_one_and_a_half():
    assert ar1_effective_frames(10, 0.9) == pytest.approx(1.47, abs=0.01)


def test_repeat_cannot_change_the_answer():
    """Prop. 3.2: feeding an observation twice leaves the belief unchanged (isolation test)."""
    T, K = 3, 6
    R = clip_correlation(T, overlap_correlation(), 0.7)
    votes = rng.normal(size=(T * N_POS, K))
    log_prior = np.log(np.full(K, 1 / K))
    S = [0, 3, 10, 14, 20]
    p = posterior(votes[S], R[np.ix_(S, S)], log_prior)
    i = 3                                              # feed observation 3 again
    S2 = S + [i]
    R2 = R[np.ix_(S2, S2)]
    p2 = posterior(votes[S2], R2, log_prior, ridge=1e-9)
    np.testing.assert_allclose(p, p2, atol=1e-5)


def test_overlap_table_is_a_correlation_matrix():
    A = overlap_correlation()
    np.testing.assert_allclose(np.diag(A), 1)
    assert np.linalg.eigvalsh(A).min() > -1e-12
    assert A[0, 1] == pytest.approx(0.5)               # glance vs crop
    assert A[1, 2] == pytest.approx(0.5)               # side neighbours
    assert A[1, 5] == pytest.approx(0.25)              # diagonal neighbours
    assert A[1, 3] == pytest.approx(0.0)               # two apart


def test_overlap_table_is_singular_because_corners_tile_the_frame():
    """Found before real data: the 4 corner crops tile the frame, so the glance is a linear
    combination of them and A has a zero eigenvalue; w = A^{-1} 1 then ignores the centre crop.
    Shrinking A toward I (correlation.shrink) makes it well conditioned and gives the centre
    crop its weight back."""
    A = overlap_correlation()
    assert np.linalg.eigvalsh(A).min() < 1e-10
    w = evidence_weights(A[1:, 1:], ridge=0).reshape(3, 3)
    np.testing.assert_allclose(w, [[1, 0, 1], [0, 0, 0], [1, 0, 1]], atol=1e-8)
    assert np.linalg.cond(shrink(A, 0.3)) < 10
    assert evidence_weights(shrink(A, 0.4), ridge=0)[5] > 0          # centre crop counts again
    assert np.all(evidence_weights(shrink(A, 0.7), ridge=0) > 0)


def test_token_masks_cover_half_the_side():
    for grid in (14, 16):
        m = token_masks(grid)
        assert m.shape == (9, grid, grid)
        assert all(abs(mk.sum() - grid * grid / 4) <= 2 * grid for mk in m)


def test_fit_recovers_known_A_and_rho():
    """Sec. 3.1 fitting: residuals generated with R = T_rho (x) A give back A and rho."""
    A_true, rho_true, K, T = overlap_correlation(), 0.7, 5, 6
    R = clip_correlation(T, A_true, rho_true)
    Lc = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
    resid, clip_ids, frame_idx = [], [], []
    for c in range(1500):
        e = (Lc @ rng.standard_normal((T * N_POS, K))).reshape(T, N_POS, K)
        resid.append(e)
        clip_ids += [c] * T
        frame_idx += list(range(T))
    resid = np.concatenate(resid)
    resid = resid - resid.mean(axis=2, keepdims=True)
    A_hat = fit_A(resid)
    rho_hat, _ = fit_rho(resid, np.array(clip_ids), np.array(frame_idx))
    np.testing.assert_allclose(A_hat, A_true, atol=0.05)
    assert rho_hat == pytest.approx(rho_true, abs=0.05)


def test_class_conditional_residuals_remove_species_means():
    votes = rng.normal(size=(200, N_POS, 4)) + np.arange(4)
    labels = rng.integers(0, 3, size=200)
    r = class_conditional_residuals(votes, labels, 3)
    for c in range(3):
        np.testing.assert_allclose(r[labels == c].mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(r.sum(axis=2), 0, atol=1e-10)


# ---------------------------------------------------------------- Sec. 4: adding one look

def test_block_inverse_update_matches_recomputation():
    """Prop. 4.1: E_{S+j} - E_S = c_j (l_j - R_jS R_SS^{-1} l_S)."""
    T, K = 3, 4
    R = clip_correlation(T, overlap_correlation(), 0.6)
    votes = rng.normal(size=(T * N_POS, K))
    S, C = [0, 10, 20, 4], [1, 5, 13, 27]
    a, s, c = add_one_look_terms(R[np.ix_(S, S)], R[np.ix_(S, C)], ridge=0)
    zero = np.zeros(K)
    E_S = scores(votes[S], R[np.ix_(S, S)], zero, ridge=0)
    for n, j in enumerate(C):
        Sj = S + [j]
        E_Sj = scores(votes[Sj], R[np.ix_(Sj, Sj)], zero, ridge=0)
        np.testing.assert_allclose(E_Sj - E_S, c[n] * (votes[j] - a[:, n] @ votes[S]), atol=1e-10)


def test_one_earlier_look_gives_c_one_over_one_plus_r():
    """Example after Prop. 4.1: one earlier look with correlation r -> c = 1/(1+r)."""
    r = 0.9
    _, s, c = add_one_look_terms(np.array([[1.0]]), np.array([[r]]), ridge=0)
    assert c[0] == pytest.approx(1 / (1 + r))
    assert s[0] == pytest.approx(1 - r ** 2)


# ---------------------------------------------------------------- Sec. 5: eq. (2)

def test_conditioning_matches_empirical_regression():
    """Prop. 5.1, checked by sampling: E[l_j | l_L] and Var[l_j | l_L]."""
    R = clip_correlation(2, overlap_correlation(), 0.8)
    L, j, sigma2 = [1, 2, 11], 5, 0.7
    idx = L + [j]
    Sig = sigma2 * R[np.ix_(idx, idx)]
    draws = rng.multivariate_normal(np.zeros(len(idx)), Sig, size=200_000)
    X, y = draws[:, :-1], draws[:, -1]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_var = np.var(y - X @ beta)
    l_L = np.array([[0.3], [-1.0], [0.5]])
    mu_c, var_c = condition_on_looks(np.zeros((1, 1)), sigma2, R[np.ix_(L, [j])], R[np.ix_(L, L)],
                                     l_L, np.zeros((3, 1)), ridge=0)
    assert mu_c[0, 0] == pytest.approx(beta @ l_L[:, 0], abs=0.02)
    assert var_c[0] == pytest.approx(resid_var, rel=0.02)


def test_ridge_recovers_a_linear_map():
    X = rng.normal(size=(3000, 6))
    W = rng.normal(size=(6, 4))
    Y = X @ W + 0.1 * rng.normal(size=(3000, 4))
    pred = fit_ridge(X, Y, groups=np.arange(3000) % 30, lambdas=(1e-6, 1e-3, 1.0))
    np.testing.assert_allclose(pred.predict(X), X @ W, atol=0.05)
    assert pred.sigma2 == pytest.approx(0.01, rel=0.2)


# ---------------------------------------------------------------- Sec. 6: eq. (3)

def test_softmax_curvature_and_variance_identity():
    """Lemmas 6.2 and 6.4."""
    p = softmax(rng.normal(size=7))
    F = fisher(p)
    D = rng.normal(size=(5, 7))
    np.testing.assert_allclose(np.einsum("ck,kl,cl->c", D, F, D), var_under_p(p, D), atol=1e-12)
    np.testing.assert_allclose(F @ np.ones(7), 0, atol=1e-12)
    assert np.trace(F) == pytest.approx(trace_fisher(p))
    assert np.linalg.eigvalsh(F).min() > -1e-12


def test_kl_is_quadratic_for_small_shifts():
    """Lemma 6.3: KL(softmax(s + t D) || softmax(s)) / t^2 -> 1/2 D^T F D."""
    s, D = rng.normal(size=6), rng.normal(size=6)
    p = softmax(s)
    target = 0.5 * var_under_p(p, D[None])[0]
    for t in (1e-2, 1e-3):
        assert kl(softmax(s + t * D), p) / t ** 2 == pytest.approx(target, rel=5 * t)


def test_two_species_worked_example():
    """Proposal Step 3 example: p = 0.5, Delta mu = 2, sigma = 0.5 -> 1/2 * 0.25 * (4 + 0.5) = 0.5625 nats.

    Two species: m^T F m = p(1-p)(m_1 - m_2)^2 and tr F = 2 p (1-p) (derivations, Example 6).
    """
    p = np.array([0.5, 0.5])
    m = np.array([[1.0, -1.0]])                 # m_1 - m_2 = 2
    v = np.array([0.5 ** 2])
    assert eig_closed_form(p, m, v)[0] == pytest.approx(0.5625)
    assert eig_closed_form(p, m, v)[0] < np.log(2)


def test_closed_form_converges_to_monte_carlo_as_shifts_shrink():
    """Theorem 6.5 is second order: its relative error should fall roughly in proportion to the shift."""
    s = rng.normal(size=8)
    m0 = rng.normal(size=(6, 8))
    errors = []
    for scale in (0.4, 0.1, 0.025):
        m, v = scale * m0, np.full(6, 0.5 * scale ** 2)
        closed = eig_closed_form(softmax(s), m, v, cap=False)
        mc, _ = eig_monte_carlo(s, m, v, n_samples=40_000, rng=np.random.default_rng(1))
        errors.append(np.max(np.abs(closed - mc) / mc))
    assert errors[2] < 0.03
    assert errors[2] < errors[1] < errors[0]


def test_repeated_look_keeps_five_percent_of_surprise():
    """Example in Step 3: r = 0.9 -> c = 0.53, 19% variance left, ~5% of the surprise term."""
    r = 0.9
    c = 1 / (1 + r)
    assert c ** 2 * (1 - r ** 2) == pytest.approx(0.05, abs=0.003)


def test_eig_vanishes_as_belief_becomes_sure():
    """Cor. 7.1: EIG <= (1 - p_max)(|m|^2 + v)."""
    m, v = rng.normal(size=(4, 5)), np.ones(4)
    for sharp in (1, 5, 20):
        p = softmax(np.array([sharp, 0, 0, 0, 0.0]))
        eig = eig_closed_form(p, m, v, cap=False)
        bound = (1 - p.max()) * (np.sum(m ** 2, axis=1) + v)
        assert np.all(eig <= bound + 1e-12)


def test_masked_fit_matches_unmasked_when_nothing_is_masked():
    from eyes.correlation import fit_A, fit_rho
    rng = np.random.default_rng(0)
    resid = rng.normal(size=(400, 10, 6))
    clips, fidx = np.repeat(np.arange(100), 4), np.tile(np.arange(4), 100)
    full = np.ones((400, 10), bool)
    np.testing.assert_allclose(fit_A(resid, full), fit_A(resid), atol=1e-10)
    assert abs(fit_rho(resid, clips, fidx, mask=full)[0] - fit_rho(resid, clips, fidx)[0]) < 1e-9


def test_masked_fit_ignores_what_is_left_out():
    """Garbage in masked-out entries must not change the fitted A."""
    from eyes.correlation import fit_A
    rng = np.random.default_rng(1)
    resid = rng.normal(size=(500, 10, 5))
    mask = rng.random((500, 10)) > 0.3
    mask[:, 0] = True
    dirty = resid.copy()
    dirty[~mask] = 1e3 * rng.normal(size=((~mask).sum(), 5))
    np.testing.assert_allclose(fit_A(dirty * mask[..., None], mask), fit_A(resid * mask[..., None], mask), atol=1e-10)
