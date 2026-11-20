"""Checks of the crop labels from boxes and the mixture vote (eyes/boxes.py)."""
import numpy as np
import pandas as pd

from eyes.boxes import crop_labels, mixture_votes


def _logq(q):
    return np.log(np.asarray(q, dtype=float))


def test_background_crop_votes_zero():
    log_pi = np.log([0.5, 0.3, 0.2])
    v = mixture_votes(_logq([0.0 + 1e-300, 1e-300, 1e-300, 1.0]), log_pi)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_animal_crop_votes_the_usual_log_ratio():
    log_pi = np.log([0.5, 0.3, 0.2])
    q = np.array([0.6, 0.3, 0.1])
    v = mixture_votes(np.log(np.r_[q, 1e-300]), log_pi)
    np.testing.assert_allclose(v, np.log(q) - log_pi, atol=1e-12)


def test_mixture_vote_matches_bayes_rule_on_a_toy_model():
    """p(c|y) = a p(c|A=1,y) + (1-a) p(c|A=0); the vote must equal log p(c|y)/p(c) up to a constant."""
    rng = np.random.default_rng(0)
    K, a = 4, 0.3
    pi = rng.dirichlet(np.ones(K))
    lik_animal = rng.random(K)            # p(c | A=1, y)
    lik_bg = rng.random()                 # p(c | A=0)
    p_c_given_y = a * lik_animal + (1 - a) * lik_bg
    # the classifier's posterior over s in {species, empty} for this crop
    joint = np.r_[a * pi * lik_animal, (1 - a) * lik_bg]
    q = joint / joint.sum()
    v = mixture_votes(np.log(q), np.log(pi))
    truth = np.log(p_c_given_y)
    np.testing.assert_allclose(v - v.mean(), truth - truth.mean(), atol=1e-12)


def test_crop_labels_from_a_box():
    """A small box inside the top-left quarter: crop 1 holds it; crops it does not touch are empty."""
    frames = pd.DataFrame({"path": ["train/a.jpg", "train/b.jpg", "train/c.jpg"], "y_frame": [3, 0, 3]})
    det = {"a.jpg": [(0.9, 0.05, 0.05, 0.1, 0.1)],         # conf, x, y, w, h
           "c.jpg": [(0.1, 0.05, 0.05, 0.1, 0.1)]}         # too unsure: frame skipped
    lab = crop_labels(frames, det)
    assert lab[0, 0] == 1                                   # top-left crop holds the box
    assert (lab[0, [2, 5, 6, 7, 8]] == 0).all()             # crops far away: empty
    assert (lab[1] == 0).all()                              # frame labelled empty: all crops empty
    assert (lab[2] == -1).all()                             # no confident box: unusable
