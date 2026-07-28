import leakage_guards as lg
import numpy as np
import pytest


def test_overlapping_split_raises():
    lg.assert_session_disjoint([1, 2, 3], [4, 5])  # ok
    with pytest.raises(lg.LeakageError):
        lg.assert_session_disjoint([1, 2, 3], [3, 4])


def test_constant_column_rejected():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(100, 3))
    lg.variance_floor(good)  # ok
    bad = good.copy()
    bad[:, 1] = 7.0  # constant column
    with pytest.raises(lg.LeakageError):
        lg.variance_floor(bad)


def test_leaked_a_t_column_aborts():
    rng = np.random.default_rng(1)
    labels = rng.uniform(-1, 1, size=(80, 2))
    inputs = rng.normal(size=(80, 4))
    lg.assert_no_label_column(inputs, labels)  # ok
    inputs[:, 2] = labels[:, 0]  # smuggle the label in as a feature
    with pytest.raises(lg.LeakageError):
        lg.assert_no_label_column(inputs, labels)


def test_decorrelation_flags_and_raises():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=(n, 2))
    pred = (0.72 * x[:, 0] + 0.7 * rng.normal(size=n))[:, None]  # ~0.7 corr with col 0
    flags = lg.decorrelation_check(pred, x, warn=0.6, hard=0.99)
    assert any(ci == 0 for _, ci, _ in flags)
    leak_pred = x[:, [0]] + 1e-9  # ~1.0 corr
    with pytest.raises(lg.LeakageError):
        lg.decorrelation_check(leak_pred, x, warn=0.6, hard=0.95)


def test_field_ablation_control_reconstructs_linear():
    rng = np.random.default_rng(3)
    Xtr = rng.normal(size=(300, 3))
    w = np.array([0.5, -0.3, 0.2])
    ytr = (Xtr @ w)[:, None]
    Xte = rng.normal(size=(120, 3))
    yte = (Xte @ w)[:, None]
    mae = lg.field_ablation_control(Xtr, ytr, Xte, yte)
    assert mae.shape == (1,)
    assert mae[0] < 0.05  # a linear target is trivially reconstructable -> "win" is not learning


def test_derive_is_iid_separates_ar1_from_noise():
    rng = np.random.default_rng(4)
    # autocorrelated session
    m = 0.0
    ar = []
    for _ in range(60):
        m = 0.9 * m + 0.2 * rng.normal()
        ar.append(m)
    noise = rng.normal(size=60)
    flags = lg.derive_is_iid([np.array(ar), noise])
    assert flags[0] == False  # AR(1) -> autocorrelated (is_iid False)  # noqa: E712
    assert flags[1] == True  # white noise -> near-iid  # noqa: E712
