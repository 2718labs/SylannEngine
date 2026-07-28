"""leakage_guards.py — offline-gate leakage guards (C4 of the v3core instrument landing plan).

Pure numpy/sklearn. No torch, no engine import. These guards make the offline value gate honest:
they refuse a split that leaks, a constant "feature" that games the variance, a prediction that is
just a rename of a current-tick label, and a "win" that a trivial ridge on the same inputs could
reconstruct. Every guard has a fixture that trips it red (see tests/test_leakage_guards.py).

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, C4).
"""

from __future__ import annotations

import numpy as np


class LeakageError(AssertionError):
    """Raised when a leakage/decorrelation/variance guard trips."""


def assert_session_disjoint(train_ids, test_ids) -> None:
    """Raise if any session id appears in both splits (row-level leakage)."""
    inter = sorted(set(train_ids) & set(test_ids))
    if inter:
        raise LeakageError(f"train/test session overlap ({len(inter)}): {inter[:5]}")


def variance_floor(cols: np.ndarray, thr: float = 1e-6) -> None:
    """Raise if any feature column is constant / near-constant (var < thr)."""
    cols = np.asarray(cols, dtype=np.float64)
    if cols.ndim != 2:
        raise ValueError("variance_floor expects a 2-D (n, d) array")
    var = cols.var(axis=0)
    dead = np.where(var < thr)[0]
    if dead.size:
        raise LeakageError(f"constant/near-constant feature columns {dead.tolist()} (var < {thr})")


def assert_no_label_column(input_cols: np.ndarray, labels: np.ndarray, atol: float = 1e-6) -> None:
    """Raise if any input column is (numerically) identical to a current-tick label a_t.

    The most basic leak: feeding a column that *is* the thing you are predicting.
    """
    input_cols = np.asarray(input_cols, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    for li in range(labels.shape[1]):
        y = labels[:, li]
        for ci in range(input_cols.shape[1]):
            if np.allclose(input_cols[:, ci], y, atol=atol):
                raise LeakageError(f"input column {ci} == current-tick label {li} (direct leak)")


def decorrelation_check(
    pred: np.ndarray,
    input_cols: np.ndarray,
    names=None,
    warn: float = 0.7,
    hard: float = 0.95,
) -> list[tuple[int, int, float]]:
    """Flag |corr(pred, input)| > warn; RAISE on > hard.

    The §3.3 backstop: a prediction that is ~perfectly correlated with a single input column is
    reconstructing a planted signal, not learning. Returns the list of (pred_idx, input_idx, corr)
    that exceeded `warn` (below the hard ceiling).
    """
    P = np.asarray(pred, dtype=np.float64).reshape(np.asarray(pred).shape[0], -1)
    X = np.asarray(input_cols, dtype=np.float64)
    flags: list[tuple[int, int, float]] = []
    for pi in range(P.shape[1]):
        p = P[:, pi]
        if p.std() < 1e-12:
            continue
        for ci in range(X.shape[1]):
            c = X[:, ci]
            if c.std() < 1e-12:
                continue
            r = abs(float(np.corrcoef(p, c)[0, 1]))
            if r > hard:
                nm = names[ci] if names is not None else ci
                raise LeakageError(f"pred[{pi}] vs input {nm}: |corr|={r:.3f} > {hard} (leak)")
            if r > warn:
                flags.append((pi, ci, r))
    return flags


def field_ablation_control(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Ridge on the raw inputs -> per-axis test MAE.

    If a candidate estimator merely matches this trivial linear reconstruction, its "win" is
    reconstructable from the inputs and is not a learned increment. Callers compare the candidate's
    MAE against this control. Returns per-axis MAE (shape (k,)).
    """
    from sklearn.linear_model import Ridge

    Xtr = np.asarray(Xtr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    ytr = np.asarray(ytr, np.float64)
    yte = np.asarray(yte, np.float64)
    k = ytr.shape[1]
    mae = np.empty(k)
    for i in range(k):
        m = Ridge(alpha=alpha).fit(Xtr, ytr[:, i])
        mae[i] = np.abs(m.predict(Xte) - yte[:, i]).mean()
    return mae


def lag1_autocorr(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series; nan for < 3 finite points or zero variance."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 3 or s.std() < 1e-12:
        return float("nan")
    a, b = s[:-1], s[1:]
    denom = a.std() * b.std()
    if denom < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / denom)


def derive_is_iid(session_series: list[np.ndarray], acf_threshold: float = 0.3) -> np.ndarray:
    """Empirical autocorrelated/near-iid split, PINNED before any performance is seen.

    For real data there is no generator `is_iid` label, so the autocorrelated slice (the one the
    ADR PASS bar is measured on) must be derived from an observable rule fixed in advance. Rule:
    a session is autocorrelated iff |lag-1 autocorr of its valence read| >= acf_threshold.
    Returns a bool array (True == near-iid control, matching the corpus `is_iid` convention).
    """
    out = np.empty(len(session_series), dtype=bool)
    for i, s in enumerate(session_series):
        acf = lag1_autocorr(s)
        out[i] = (not np.isfinite(acf)) or (abs(acf) < acf_threshold)
    return out
