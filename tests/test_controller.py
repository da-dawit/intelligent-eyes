"""End-to-end checks of the controller on synthetic clips with a known R."""
import numpy as np
import pytest

from eyes.controller import Model, run
from eyes.correlation import shrink
from eyes.fit import choose_eps, choose_shrinkage
from eyes.geometry import overlap_correlation
from eyes.metrics import accuracy, ece
from eyes.synthetic import make_clips, predictor_sigma2


@pytest.fixture(scope="module")
def world():
    A, rho = overlap_correlation(), 0.8
    heldout, log_prior = make_clips(300, rho=rho, A=A, seed=1)
    test, _ = make_clips(1200, rho=rho, A=A, seed=2)
    sigma2 = predictor_sigma2(heldout)
    lam, _ = choose_shrinkage(heldout, A, rho, log_prior, sigma2)
    model = Model(shrink(A, lam), rho, log_prior, sigma2)
    return dict(test=test, heldout=heldout, model=model, lam=lam,
                naive=Model.independent(log_prior, sigma2))


def _evaluate(clips, model, rule, budget=2, seed=0, **kw):
    rng = np.random.default_rng(seed)
    out = [run(c, model, rule=rule, budget=budget, rng=rng, **kw) for c in clips]
    labels = np.array([c.label for c in clips])
    probs = np.stack([o.p for o in out])
    return accuracy(probs.argmax(1), labels), ece(probs, labels), out


def test_shrinkage_is_needed_and_chosen_inside_the_range(world):
    assert 0.0 < world["lam"] < 1.0


def test_passes_are_counted(world):
    for c in world["test"][:20]:
        assert run(c, world["model"], "eig", budget=2).passes == c.T + 2
        assert run(c, world["model"], "all").passes == 10 * c.T
        assert run(c, world["model"], "eig", budget=0).passes == c.T


def test_eyes_beat_random_looks_and_recover_most_of_all9(world):
    clips, model = world["test"], world["model"]
    acc_glance, _, _ = _evaluate(clips, model, "eig", budget=0)
    acc_random, _, _ = _evaluate(clips, model, "random")
    acc_eyes, _, _ = _evaluate(clips, model, "eig")
    acc_all, _, _ = _evaluate(clips, model, "all")
    assert acc_eyes > acc_random + 0.05
    assert acc_all > acc_eyes
    assert (acc_eyes - acc_glance) / (acc_all - acc_glance) > 2 / 3      # side question (a)


def test_ignoring_R_is_overconfident(world):
    """Failure mode 2 in miniature: treating repeated evidence as new inflates confidence."""
    _, ece_R, _ = _evaluate(world["test"], world["model"], "all")
    _, ece_naive, _ = _evaluate(world["test"], world["naive"], "all")
    assert ece_naive > ece_R + 0.05


def test_stopping_rule_saves_passes(world):
    clips, model = world["test"][:400], world["model"]
    _, _, full = _evaluate(clips, model, "eig", budget=4)
    _, _, stop = _evaluate(clips, model, "eig", budget=4, eps=0.05)
    assert np.mean([o.passes for o in stop]) < np.mean([o.passes for o in full])
    assert any(o.stopped for o in stop)
    eps, table = choose_eps(world["heldout"], model, budget=4)
    assert eps in table


def test_first_step_record(world):
    c = world["test"][0]
    o = run(c, world["model"], "eig", record_first=True)
    assert o.first_step["closed"].shape == (9 * c.T,)
    assert o.first_step["realised_kl"].shape == (9 * c.T,)
