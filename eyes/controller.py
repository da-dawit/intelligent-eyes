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
    p_animal: np.ndarray | None = None  # (T, 9) P(crop shows the animal), from the glance (eig_mix)
    mu_animal: np.ndarray | None = None # (T, 9, K) predicted vote if it does (eig_mix)
    gate: np.ndarray | None = None      # (T, 9) bool: once taken, does the crop show the animal?
                                        # (its own classifier says so). Gated fusion leaves the
                                        # others out of eq. (1); None = every look enters.

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


def enters(clip: Clip, i: int) -> bool:
    """Does observation i enter eq. (1)? Glances always; looks unless gated out as background."""
    k = i % N_POS
    return clip.gate is None or k == 0 or bool(clip.gate[i // N_POS, k - 1])


def glance_indices(T: int) -> list[int]:
    return [f * N_POS for f in range(T)]


def look_indices(T: int) -> list[int]:
    return [f * N_POS + k for f in range(T) for k in range(1, N_POS)]


def candidate_terms(clip: Clip, model: Model, S: list[int], L: list[int], C: list[int], parts: bool = False):
    """For candidates C given observations S (looks L in S), return

        s      (K,)    current scores, p = softmax(s)
        m, v   (C,K),(C,)  mean and variance of each candidate's score shift Delta
        shift  (C, K)  the *actual* shift each candidate would cause (for evaluation only)

    parts=True also returns (c, implied, correction): c_j of Prop. 4.1, what S implies about
    each candidate (a_j^T l_S), and how conditioning on the looks taken moved each
    prediction (Prop. 5.1) -- what eig_mix needs to score its two branches.
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
    if parts:
        return s, m, v, shift, (c, implied, mu_c - mu[C])
    return s, m, v, shift


def eig_mixture(p: np.ndarray, c: np.ndarray, implied: np.ndarray, correction: np.ndarray, v: np.ndarray,
                p_animal: np.ndarray, mu_animal: np.ndarray, gated: bool = False) -> np.ndarray:
    """EIG of a look whose vote has two branches: the crop shows the animal (probability
    p_animal, predicted vote mu_animal) or only background (vote 0, eyes/boxes.py).

        EIG = p_animal EIG(m_A, v) + (1 - p_animal) EIG(m_B, v)
        m_A = c (mu_animal + correction - implied),   m_B = c (correction - implied)

    The split is exact for the expectation E[KL]; each branch uses the closed form of eq. (3).
    With gated fusion a background look is left out of eq. (1), so its branch changes nothing:
    EIG = p_animal EIG(m_A, v).
    """
    m_a = c[:, None] * (mu_animal + correction - implied)
    eig_a = p_animal * eig_closed_form(p, m_a, v)
    if gated:
        return eig_a
    m_b = c[:, None] * (correction - implied)
    return eig_a + (1 - p_animal) * eig_closed_form(p, m_b, v)


def run(clip: Clip, model: Model, rule: str = "eig", budget: int = 2, eps: float = 0.0,
        n_mc: int = 64, rng: np.random.Generator | None = None, record_first: bool = False) -> Outcome:
    """Glance at every frame, then take up to `budget` looks chosen by `rule`.

    rule: "eig"        closed-form EIG (eq. 3)              -- the eyes
          "eig_mix"    EIG with an animal / background branch per look (needs clip.p_animal)
          "eig_mc"     Monte Carlo EIG with exact KL
          "random"     uniformly random looks               -- B2
          "score:NAME" highest clip.look_scores[NAME]       -- B3 attention, B4 motion
          "all"        every look in every frame            -- All-9 reference
    """
    rng = rng or np.random.default_rng(0)
    T = clip.T
    S = glance_indices(T)              # observations in eq. (1)
    L: list[int] = []                  # every look taken (each costs a pass)
    L_in: list[int] = []               # looks taken that enter eq. (1)
    best_eig: list[float] = []
    stopped = False
    first = None

    if rule == "all":
        L = look_indices(T)
        L_in = [i for i in L if enters(clip, i)]
        S = S + L_in
        budget = 0

    for step in range(budget):
        C = [i for i in look_indices(T) if i not in L]
        if not C:
            break
        if rule in ("eig", "eig_mc", "eig_mix") or (record_first and step == 0):
            s, m, v, shift, (c, implied, corr) = candidate_terms(clip, model, S, L_in, C, parts=True)
            p = softmax(s)
        if rule == "eig":
            value = eig_closed_form(p, m, v)
        elif rule == "eig_mix":
            fk = [(i // N_POS, i % N_POS - 1) for i in C]
            pa = np.array([clip.p_animal[f, k] for f, k in fk])
            mu_a = np.stack([clip.mu_animal[f, k] for f, k in fk])
            value = eig_mixture(p, c, implied, corr, v, pa, mu_a, gated=clip.gate is not None)
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
        if rule in ("eig", "eig_mc", "eig_mix"):
            best_eig.append(float(value[best]))
            if value[best] < eps:
                stopped = True
                break
        else:
            best_eig.append(float("nan"))
        L.append(C[best])
        if enters(clip, C[best]):
            L_in.append(C[best])
            S.append(C[best])

    K = clip.votes.shape[-1]
    votes = clip.votes.reshape(-1, K)
    R = model.R(T)
    p = softmax(scores(votes[S], R[np.ix_(S, S)], model.log_prior, model.ridge))
    looks = [(i // N_POS, i % N_POS) for i in L]
    return Outcome(int(np.argmax(p)), p, looks, T + len(L), best_eig, stopped, first)
