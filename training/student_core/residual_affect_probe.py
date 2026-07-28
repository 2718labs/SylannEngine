"""residual_affect_probe.py — the persistence-anchored residual affect predictor (C2).

The registered PRIMARY estimator of the offline value gate. Pure numpy (NO torch). Its defining
honesty property:

    a_hat_t = clip(a_prev + w . x_t,  valid_range)     with  w initialized to 0

so at init `a_hat_t == clip(a_prev)`: for an IN-RANGE `a_prev` (every valid assessor read is
in-range) that is `a_prev` exactly (persistence), and the only thing it ever learns is the residual
`r_t = a_t - a_prev` — which is exactly the quantity the ADR-0001 metric scores. A model that learns
nothing produces paired-MAE improvement of *exactly zero* over persistence; a win cannot be
manufactured by architecture. Ridge closed-form fit, no iteration, deterministic. (Callers must score
the persistence baseline with the SAME clip — see clip_reads — so an out-of-range read cannot hand
the estimator a free non-learned edge.)

Grounded ranges (training/student_core/simulate_corpus.py: synth_assessor):
    valence in [-1, 1], arousal in [0, 1].
[MUST-VERIFY-FIRST #2] confirm the real assessor's declared ranges (assessor.py DEFAULT_DIMENSIONS)
before pinning clip() bounds on real data; the synthetic ranges are a stand-in.

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, C2).
"""

from __future__ import annotations

import numpy as np

AXES = ("valence", "arousal")
VALID_RANGE = {"valence": (-1.0, 1.0), "arousal": (0.0, 1.0)}


def clip_reads(
    arr: np.ndarray, axes: tuple[str, ...] = AXES, valid_range: dict | None = None
) -> np.ndarray:
    """Clip an (n, k) array of affect reads to per-axis valid ranges (shared by estimator + baseline)."""
    vr = valid_range or VALID_RANGE
    out = np.asarray(arr, np.float64).copy()
    for i, ax in enumerate(axes):
        lo, hi = vr[ax]
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out


class ResidualAffectProbeV1:
    """Persistence-anchored, ridge-fit residual predictor over pre-a_t features.

    fit target: r_t = a_t - a_prev (per axis). predict: clip(a_prev + w . x_t).
    `w` starts as zeros; a never-fit probe predicts persistence bit-for-bit for in-range a_prev
    (clip is a no-op on valid reads).
    """

    def __init__(
        self,
        ridge_lambda: float = 1.0,
        valid_range: dict | None = None,
        axes: tuple[str, ...] = AXES,
    ):
        self.ridge_lambda = float(ridge_lambda)
        self.axes = tuple(axes)
        self.valid_range = dict(valid_range or VALID_RANGE)
        self.k = len(self.axes)
        self.d: int | None = None
        self.w: np.ndarray | None = None  # (k, d), zeros until fit

    # -- helpers ----------------------------------------------------------------
    def _clip(self, out: np.ndarray) -> np.ndarray:
        return clip_reads(out, self.axes, self.valid_range)

    def _ensure_w(self, d: int) -> None:
        if self.w is None:
            self.d = d
            self.w = np.zeros((self.k, d), dtype=np.float64)

    # -- API --------------------------------------------------------------------
    def fit(self, X: np.ndarray, a_prev: np.ndarray, a_t: np.ndarray) -> ResidualAffectProbeV1:
        """Ridge closed-form on the residual r = a_t - a_prev, per axis."""
        X = np.asarray(X, np.float64)
        a_prev = np.asarray(a_prev, np.float64)
        a_t = np.asarray(a_t, np.float64)
        n, d = X.shape
        self._ensure_w(d)
        if d == 0:
            return self  # persistence rung: no features, w stays (k, 0) zeros
        R = a_t[:, : self.k] - a_prev[:, : self.k]
        XtX = X.T @ X + self.ridge_lambda * np.eye(d)
        Xtr = X.T @ R  # (d, k)
        sol = np.linalg.solve(XtX, Xtr)  # (d, k)
        self.w = sol.T.copy()  # (k, d)
        return self

    def predict(self, X: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
        """clip(a_prev + w . x_t). With w == 0 (unfit) this is exactly a_prev clipped."""
        X = np.asarray(X, np.float64)
        a_prev = np.asarray(a_prev, np.float64)
        self._ensure_w(X.shape[1])
        out = a_prev[:, : self.k] + X @ self.w.T  # (n, k)
        return self._clip(out)

    def mae(self, X: np.ndarray, a_prev: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """Per-axis mean absolute error of the prediction vs a_t."""
        pred = self.predict(X, a_prev)
        return np.abs(pred - np.asarray(a_t, np.float64)[:, : self.k]).mean(axis=0)
