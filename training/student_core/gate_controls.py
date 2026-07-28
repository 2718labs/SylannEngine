"""gate_controls.py — statistical controls for the offline gate (F3).

Backs the PASS/NO-GO decision with cluster-aware inference instead of point estimates:
a paired bootstrap CI on the improvement (resampling WHOLE episodes, never rows), a family-wise
correction for the rung x axis multiplicity, and a residual-shuffle placebo. The 15%/0.02 effect
floor remains the substance test — a CI that merely excludes zero never suffices.

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, F3).
"""

from __future__ import annotations

import numpy as np


def _episode_groups(episode_ids: np.ndarray):
    ids = np.asarray(episode_ids)
    uniq = np.unique(ids)
    return uniq, [np.where(ids == u)[0] for u in uniq]


def paired_bootstrap_ci(
    diff: np.ndarray, episode_ids: np.ndarray, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """CI on E[diff] resampling whole episodes (clustered). diff: per-tick (n,), positive = est better.

    Returns (mean, lo, hi). Independence is at the episode level, so we resample episodes with
    replacement — the honest cluster bootstrap for autocorrelated within-session ticks.
    """
    diff = np.asarray(diff, np.float64)
    uniq, groups = _episode_groups(episode_ids)
    k = len(uniq)
    if k < 2:
        m = float(diff.mean()) if diff.size else float("nan")
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        rows = np.concatenate([groups[p] for p in pick])
        means[b] = diff[rows].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(diff.mean()), lo, hi


def familywise_holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down: return {key: reject_null} controlling FWER at alpha.

    Used across the rung x axis family so a single lucky comparison cannot pass on CI>0 alone.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, bool] = {}
    still_rejecting = True
    for i, (key, p) in enumerate(items):
        thresh = alpha / (m - i)
        if still_rejecting and p <= thresh:
            out[key] = True
        else:
            still_rejecting = False
            out[key] = False
    return out


def residual_shuffle_placebo(
    probe_ctor, Xtr, ap_tr, at_tr, Xte, ap_te, at_te, seed: int = 0
) -> float:
    """Refit the probe on a PERMUTED residual target; return est-improvement-vs-persistence (should ~0).

    A genuine placebo: permuting r_t across rows destroys any real X->residual mapping, so a probe
    that still 'improves' is exploiting a clip/saturation artifact, not signal.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(at_tr, np.float64) - np.asarray(ap_tr, np.float64)
    perm = rng.permutation(len(r))
    at_shuffled = np.asarray(ap_tr, np.float64) + r[perm]  # persistence + shuffled residual
    probe = probe_ctor().fit(Xtr, ap_tr, at_shuffled)
    est_mae = np.abs(probe.predict(Xte, ap_te) - np.asarray(at_te, np.float64)).mean()
    pers_mae = np.abs(np.asarray(ap_te, np.float64) - np.asarray(at_te, np.float64)).mean()
    return float(pers_mae - est_mae)  # improvement over persistence; ~0 under the placebo
