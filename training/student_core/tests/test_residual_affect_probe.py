from pathlib import Path

import numpy as np
import residual_affect_probe as rap
from residual_affect_probe import ResidualAffectProbeV1, clip_reads


def _valid_reads(rng, n):
    return np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n)])


def test_init_equals_persistence_bit_for_bit():
    """The honesty property: an unfit probe predicts a_prev exactly -> MAE == persistence exactly."""
    rng = np.random.default_rng(0)
    n = 200
    a_prev = _valid_reads(rng, n)  # already inside valid ranges -> clip is a no-op
    a_t = _valid_reads(rng, n)
    X = rng.normal(size=(n, 12))
    probe = ResidualAffectProbeV1()
    pred = probe.predict(X, a_prev)
    assert np.array_equal(pred, a_prev)  # bit-for-bit persistence
    mae_probe = np.abs(pred - a_t).mean(axis=0)
    mae_pers = np.abs(a_prev - a_t).mean(axis=0)
    assert np.array_equal(mae_probe, mae_pers)  # exactly zero improvement possible at init


def test_empty_feature_block_is_persistence():
    """A persistence rung (no feature blocks) can never beat persistence — fit is a no-op."""
    rng = np.random.default_rng(1)
    n = 150
    a_prev = _valid_reads(rng, n)
    a_t = _valid_reads(rng, n)
    X = np.zeros((n, 0))
    probe = ResidualAffectProbeV1().fit(X, a_prev, a_t)
    assert np.array_equal(probe.predict(X, a_prev), a_prev)


def test_clip_reads_no_free_edge_on_out_of_range_anchor():
    """clip_reads clips out-of-range reads; estimator and persistence baseline are clipped the SAME
    way, so an out-of-range a_prev cannot hand the estimator a non-learned edge (Lens1-d fix)."""
    a_prev = np.array([[1.5, -0.3], [-2.0, 1.4]])  # both axes out of range
    clipped = clip_reads(a_prev)
    assert np.array_equal(clipped, np.array([[1.0, 0.0], [-1.0, 1.0]]))
    # an unfit probe's prediction equals clip_reads(a_prev) exactly -> same clip as the baseline
    probe = ResidualAffectProbeV1()
    assert np.array_equal(probe.predict(np.zeros((2, 5)), a_prev), clipped)


def test_no_torch_dependency():
    """The registered estimator is numpy-only; no torch import anywhere in the module source."""
    src = Path(rap.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src


def test_shuffle_placebo_no_real_gain():
    """On features independent of the residual, the fitted probe cannot beat persistence by the floor."""
    rng = np.random.default_rng(2)
    n_tr, n_te, d = 600, 600, 20
    a_prev_tr, a_prev_te = _valid_reads(rng, n_tr), _valid_reads(rng, n_te)
    # residual is pure noise, independent of X -> unpredictable
    a_t_tr = np.clip(a_prev_tr + 0.2 * rng.normal(size=(n_tr, 2)), [-1, 0], [1, 1])
    a_t_te = np.clip(a_prev_te + 0.2 * rng.normal(size=(n_te, 2)), [-1, 0], [1, 1])
    Xtr = rng.normal(size=(n_tr, d))
    Xte = rng.normal(size=(n_te, d))
    probe = ResidualAffectProbeV1(ridge_lambda=1.0).fit(Xtr, a_prev_tr, a_t_tr)
    est_mae = probe.mae(Xte, a_prev_te, a_t_te).mean()
    pers_mae = np.abs(a_prev_te - a_t_te).mean()
    # no >= 0.02 absolute improvement can be manufactured from noise
    assert est_mae >= pers_mae - 0.02


def test_fit_learns_a_real_residual():
    """Sanity: when the residual IS a linear function of X, the probe recovers it and beats persistence."""
    rng = np.random.default_rng(3)
    n_tr, n_te, d = 800, 800, 6
    a_prev_tr, a_prev_te = _valid_reads(rng, n_tr), _valid_reads(rng, n_te)
    w = rng.normal(size=(d, 2)) * 0.1
    Xtr, Xte = rng.normal(size=(n_tr, d)), rng.normal(size=(n_te, d))
    a_t_tr = np.clip(a_prev_tr + Xtr @ w, [-1, 0], [1, 1])
    a_t_te = np.clip(a_prev_te + Xte @ w, [-1, 0], [1, 1])
    probe = ResidualAffectProbeV1(ridge_lambda=0.1).fit(Xtr, a_prev_tr, a_t_tr)
    est_mae = probe.mae(Xte, a_prev_te, a_t_te).mean()
    pers_mae = np.abs(a_prev_te - a_t_te).mean()
    assert est_mae < pers_mae  # a genuinely learnable residual is learned
