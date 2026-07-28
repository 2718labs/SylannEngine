import gate_controls as gc
import numpy as np
from residual_affect_probe import ResidualAffectProbeV1


def test_paired_bootstrap_ci_excludes_zero_on_real_effect():
    rng = np.random.default_rng(0)
    k, per = 40, 20
    ids = np.repeat(np.arange(k), per)
    diff = np.repeat(rng.normal(0.05, 0.01, k), per) + rng.normal(0, 0.005, k * per)  # mean ~+0.05
    mean, lo, hi = gc.paired_bootstrap_ci(diff, ids, n_boot=1000, seed=1)
    assert lo <= mean <= hi
    assert lo > 0  # a clear positive clustered effect -> CI excludes 0


def test_paired_bootstrap_ci_includes_zero_on_noise():
    rng = np.random.default_rng(2)
    k, per = 40, 20
    ids = np.repeat(np.arange(k), per)
    diff = rng.normal(0.0, 0.2, k * per)  # zero-mean noise
    mean, lo, hi = gc.paired_bootstrap_ci(diff, ids, n_boot=1000, seed=3)
    assert lo < 0 < hi  # CI straddles 0


def test_familywise_holm():
    assert gc.familywise_holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {
        "a": True,
        "b": True,
        "c": True,
    }
    out = gc.familywise_holm({"a": 0.001, "b": 0.03, "c": 0.5})
    assert out == {"a": True, "b": False, "c": False}  # step-down stops at the first non-reject


def test_residual_shuffle_placebo_kills_a_real_effect():
    rng = np.random.default_rng(4)
    n_tr, n_te, d = 600, 600, 6
    ap_tr = np.column_stack([rng.uniform(-1, 1, n_tr), rng.uniform(0, 1, n_tr)])
    ap_te = np.column_stack([rng.uniform(-1, 1, n_te), rng.uniform(0, 1, n_te)])
    w = rng.normal(size=(d, 2)) * 0.1
    Xtr, Xte = rng.normal(size=(n_tr, d)), rng.normal(size=(n_te, d))
    at_tr = np.clip(ap_tr + Xtr @ w, [-1, 0], [1, 1])
    at_te = np.clip(ap_te + Xte @ w, [-1, 0], [1, 1])

    # real fit improves over persistence
    real = ResidualAffectProbeV1(ridge_lambda=0.1).fit(Xtr, ap_tr, at_tr)
    real_imp = np.abs(ap_te - at_te).mean() - np.abs(real.predict(Xte, ap_te) - at_te).mean()
    assert real_imp > 0.005
    # placebo (shuffled residual): a SINGLE shuffle is high-variance, so assert on the distribution
    # over several shuffles — it must center at ~0 (no real X->residual mapping to exploit).
    placebo = np.array(
        [
            gc.residual_shuffle_placebo(
                lambda: ResidualAffectProbeV1(ridge_lambda=0.1),
                Xtr,
                ap_tr,
                at_tr,
                Xte,
                ap_te,
                at_te,
                seed=s,
            )
            for s in range(12)
        ]
    )
    assert abs(placebo.mean()) < 0.005  # placebo improvement centers at zero
    assert placebo.mean() < real_imp
