"""The best pair of looks: proposal metric 4 and the greedy guarantee (derivations Sec. 8).

Prop. 4.1 adds one look with a block inverse. Adding a *block* B of looks works the same
way, with matrices where Prop. 4.1 has numbers:

    a_B     = R_SS^{-1} R_SB              (|S|, |B|)  what S already says about B
    Sig_B   = R_BB - R_BS a_B             (|B|, |B|)  Schur complement: what is new
    g_B     = Sig_B^{-1} (1 - a_B^T 1)    (|B|,)      turns B's innovations into evidence
    Delta_B = g_B^T (l_B - a_B^T l_S)     (K,)        change in the species scores

For |B| = 1 this is g = c_j of Prop. 4.1. Before the looks are taken, eq. (2) predicts
l_B ~ N(mu_B, sigma2 R_BB) (prediction errors repeat like the votes, as in Prop. 5.1), so

    Delta_B ~ N(m_B, v_B I),   m_B = g_B^T (mu_B - a_B^T l_S),   v_B = sigma2 g_B^T R_BB g_B

and eq. (3) gives f(B), the information the pair is expected to give, before seeing it.
For pairs, Sig_B is 2 x 2 and is inverted by hand, so all (9T choose 2) pairs of a clip
are scored at once.

Two things are compared for each test clip, both from the glance state:
  * information: f(greedy pair) / max_B f(B)  -- metric 4, the guarantee says >= 3/4 under A4
  * accuracy:    the oracle pair that puts the most belief on the true species, an upper
                 bound for any policy that takes two looks ("the best pair" reference).
"""
from __future__ import annotations

import numpy as np

from .belief import scores, softmax
from .controller import Clip, Model, candidate_terms, glance_indices, look_indices
from .eig import eig_closed_form


def all_pairs(clip: Clip, model: Model) -> dict:
    """Score every pair of looks of `clip` from the glance state.

    Returns (P = number of pairs):
        i, j     (P,)    observation indices of the two looks (frame * 10 + position)
        f        (P,)    expected information f(B), eq. (3) for the block
        delta    (P, K)  the actual score change if the pair were taken (exact, given R)
        s        (K,)    the glance scores
    """
    T = clip.T
    K = clip.votes.shape[-1]
    S, C = glance_indices(T), look_indices(T)
    votes = clip.votes.reshape(-1, K)
    mu = clip.mu.reshape(-1, K)
    R = model.R(T)
    R_SS, R_SC, R_CC = R[np.ix_(S, S)], R[np.ix_(S, C)], R[np.ix_(C, C)]

    a = np.linalg.solve(R_SS + model.ridge * np.eye(len(S)), R_SC)   # (|S|, |C|)
    M = R_CC - R_SC.T @ a                     # Schur complement for every pair of candidates
    u = 1.0 - a.sum(axis=0)                   # (|C|,)
    e = votes[C] - a.T @ votes[S]             # innovations, actual   (|C|, K)
    e_bar = mu[C] - a.T @ votes[S]            # innovations, predicted

    i, j = np.triu_indices(len(C), k=1)
    d_i, d_j, off = M[i, i] + model.ridge, M[j, j] + model.ridge, M[i, j]
    det = np.clip(d_i * d_j - off ** 2, 1e-12, None)
    g_i = (d_j * u[i] - off * u[j]) / det     # g_B = Sig_B^{-1} (1 - a_B^T 1), 2 x 2 by hand
    g_j = (d_i * u[j] - off * u[i]) / det

    s = scores(votes[S], R_SS, model.log_prior, model.ridge)
    p = softmax(s)
    m = g_i[:, None] * e_bar[i] + g_j[:, None] * e_bar[j]
    v = model.sigma2 * (g_i ** 2 + g_j ** 2 + 2 * g_i * g_j * R_CC[i, j])
    delta = g_i[:, None] * e[i] + g_j[:, None] * e[j]
    C = np.asarray(C)
    return dict(i=C[i], j=C[j], f=eig_closed_form(p, m, v), delta=delta, s=s)


def pair_report(clip: Clip, model: Model, eyes_looks: list[int]) -> dict:
    """Metric 4 and the best-pair reference for one clip.

    eyes_looks: the two observation indices the (adaptive) controller actually took.
    """
    P = all_pairs(clip, model)
    S, C = glance_indices(clip.T), look_indices(clip.T)
    _, m1, v1, _ = candidate_terms(clip, model, S, [], C)
    first = C[int(np.argmax(eig_closed_form(softmax(P["s"]), m1, v1)))]

    def where(x, y):
        x, y = min(x, y), max(x, y)
        return int(np.flatnonzero((P["i"] == x) & (P["j"] == y))[0])

    with_first = np.flatnonzero((P["i"] == first) | (P["j"] == first))
    greedy = int(with_first[np.argmax(P["f"][with_first])])          # best partner of look 1
    best = int(np.argmax(P["f"]))
    eyes = where(*eyes_looks)

    log_p = P["s"][None, :] + P["delta"]
    log_p = log_p - np.logaddexp.reduce(log_p, axis=1, keepdims=True)
    oracle = int(np.argmax(log_p[:, clip.label])) if clip.label >= 0 else best
    p_oracle = np.exp(log_p[oracle])
    return dict(
        f_best=float(P["f"][best]), f_greedy=float(P["f"][greedy]), f_eyes=float(P["f"][eyes]),
        best_is_greedy=best == greedy,
        oracle_pred=int(np.argmax(p_oracle)), oracle_conf=float(p_oracle.max()),
        oracle_p_true=float(p_oracle[clip.label]) if clip.label >= 0 else float("nan"),
        oracle_looks=(int(P["i"][oracle]), int(P["j"][oracle])),
        n_pairs=len(P["f"]),
    )
