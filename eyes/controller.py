"""The eyes: glance at every frame, then take looks where and when they are worth most.

Loop (proposal Fig. 3):
    1. glance at all T frames, combine the votes with eq. (1)
    2. for every candidate look (f, k), predict what it would add (eq. 2 + Prop. 4.1)
    3. score candidates (EIG by eq. 3, or a baseline rule), take the best one
    4. stop when the budget is spent or the best EIG falls below eps

Every encoder output is cached, so "taking" a look reads its vote from the cache and
counts one encoder pass. The same loop runs the baselines: only the scoring rule changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .belief import add_one_look_terms, softmax, scores
from .correlation import clip_correlation
from .eig import eig_closed_form, eig_monte_carlo, kl
from .geometry import N_POS
from .predictor import condition_on_looks


@dataclass
class Clip:
    """Everything the controller may read about one clip (all from the cache)."""
    votes: np.ndarray                 # (T, 10, K) vote of every observation
    mu: np.ndarray                    # (T, 10, K) predicted votes (eq. 2); position 0 unused
    label: int = -1
    clip_id: str = ""
    camera: int = -1
    look_scores: dict = field(default_factory=dict)   # name -> (T, 9) scores for B3/B4

    @property
    def T(self) -> int:
        return self.votes.shape[0]


@dataclass
class Model:
    """Fitted pieces shared by all clips."""
    A: np.ndarray                     # (10, 10) position repetition table
    rho: float                        # frame-to-frame repetition
    log_prior: np.ndarray             # (K,) log pi(y)
    sigma2: float                     # prediction error variance of eq. (2)
    ridge: float = 1e-6
    _R_cache: dict = field(default_factory=dict, repr=False)

    def R(self, T: int) -> np.ndarray:
        if T not in self._R_cache:
            self._R_cache[T] = clip_correlation(T, self.A, self.rho)
        return self._R_cache[T]

    @staticmethod
    def independent(log_prior, sigma2, ridge=1e-6) -> "Model":
        """'Eyes without R': every observation treated as independent (R = I)."""
        return Model(np.eye(N_POS), 0.0, log_prior, sigma2, ridge)


@dataclass
class Outcome:
    pred: int
    p: np.ndarray                     # final belief (K,)
    looks: list                       # [(f, k), ...] in the order taken
    passes: int                       # encoder passes used = T glances + looks
    best_eig: list                    # best EIG at each decision (nan for non-EIG rules)
    stopped: bool                     # stopped by eps before the budget ran out
    first_step: dict | None = None    # predicted vs realised gains at the first decision


def glance_indices(T: int) -> list[int]:
    return [f * N_POS for f in range(T)]


def look_indices(T: int) -> list[int]:
    return [f * N_POS + k for f in range(T) for k in range(1, N_POS)]


def candidate_terms(clip: Clip, model: Model, S: list[int], L: list[int], C: list[int]):
    """For candidates C given observations S (looks L in S), return

        s      (K,)    current scores, p = softmax(s)
        m, v   (C,K),(C,)  mean and variance of each candidate's score shift Delta
        shift  (C, K)  the *actual* shift each candidate would cause (for evaluation only)
    """
    R = model.R(clip.T)
    K = clip.votes.shape[-1]
    votes = clip.votes.reshape(-1, K)
    mu = clip.mu.reshape(-1, K)
    R_SS = R[np.ix_(S, S)]
    s = scores(votes[S], R_SS, model.log_prior, model.ridge)

    a, _, c = add_one_look_terms(R_SS, R[np.ix_(S, C)], model.ridge)       # Prop. 4.1
    implied = a.T @ votes[S]                                               # (C, K)
    mu_c, var_c = condition_on_looks(mu[C], model.sigma2, R[np.ix_(L, C)], R[np.ix_(L, L)],
                                     votes[L], mu[L], model.ridge)         # Prop. 5.1
    m = c[:, None] * (mu_c - implied)                                      # Cor. 5.2
    v = c ** 2 * var_c
    shift = c[:, None] * (votes[C] - implied)                              # exact, given R
    return s, m, v, shift


def run(clip: Clip, model: Model, rule: str = "eig", budget: int = 2, eps: float = 0.0,
        n_mc: int = 64, rng: np.random.Generator | None = None, record_first: bool = False) -> Outcome:
    """Glance at every frame, then take up to `budget` looks chosen by `rule`.

    rule: "eig"        closed-form EIG (eq. 3)              -- the eyes
          "eig_mc"     Monte Carlo EIG with exact KL
          "random"     uniformly random looks               -- B2
          "score:NAME" highest clip.look_scores[NAME]       -- B3 attention, B4 motion
          "all"        every look in every frame            -- All-9 reference
    """
    rng = rng or np.random.default_rng(0)
    T = clip.T
    S = glance_indices(T)
    L: list[int] = []
    best_eig: list[float] = []
    stopped = False
    first = None

    if rule == "all":
        L = look_indices(T)
        S = S + L
        budget = 0

    for step in range(budget):
        C = [i for i in look_indices(T) if i not in L]
        if not C:
            break
        if rule in ("eig", "eig_mc") or (record_first and step == 0):
            s, m, v, shift = candidate_terms(clip, model, S, L, C)
            p = softmax(s)
        if rule == "eig":
            value = eig_closed_form(p, m, v)
        elif rule == "eig_mc":
            value, _ = eig_monte_carlo(s, m, v, n_mc, rng)
        elif rule == "random":
            value = rng.random(len(C))
        elif rule.startswith("score:"):
            table = clip.look_scores[rule.split(":", 1)[1]]                  # (T, 9)
            value = np.array([table[i // N_POS, i % N_POS - 1] for i in C], dtype=float)
        else:
            raise ValueError(f"unknown rule {rule!r}")

        if record_first and step == 0:
            p_new = softmax(s[None, :] + shift)
            first = dict(
                closed=eig_closed_form(p, m, v),
                mc=eig_monte_carlo(s, m, v, n_mc, rng)[0],
                realised_kl=kl(p_new, p[None, :]),
                realised_true_gain=(np.log(p_new[:, clip.label] + 1e-12) - np.log(p[clip.label] + 1e-12))
                if clip.label >= 0 else None,
            )

        best = int(np.argmax(value))
        if rule in ("eig", "eig_mc"):
            best_eig.append(float(value[best]))
            if value[best] < eps:
                stopped = True
                break
        else:
            best_eig.append(float("nan"))
        L.append(C[best])
        S.append(C[best])

    K = clip.votes.shape[-1]
    votes = clip.votes.reshape(-1, K)
    R = model.R(T)
    p = softmax(scores(votes[S], R[np.ix_(S, S)], model.log_prior, model.ridge))
    looks = [(i // N_POS, i % N_POS) for i in L]
    return Outcome(int(np.argmax(p)), p, looks, T + len(L), best_eig, stopped, first)
