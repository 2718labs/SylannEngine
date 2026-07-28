"""pilot.py — the >=30-session pilot and the pre-registered kill criteria (F4).

Implements ADR-0001 sections 6.2-6.4: from a pilot corpus, measure the per-tick paired-difference std
`sigma_d`, the labeled-ticks-per-session `m` (at the real assessor-call rate), and the intra-session
`rho_icc` of the paired difference `d_t = |persistence_err| - |estimator_err|`; derive the required
session count `N` via the design-effect inflation using the UPPER-CI icc; and fire the escalation to
shelve-vs-redesign the moment the pilot autocorrelated effect is <= 0, OR N > 300, OR the projected
collection ETA > 8 weeks. A clean escalation is the gate working, not a failure.

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, F4).
"""

from __future__ import annotations

import math

import numpy as np

# z_{alpha/2} + z_{power} for alpha=0.05 two-sided, power=0.80 (ADR-0001 eq at line 310).
_Z_SUM = 1.959963985 + 0.841621234  # = 2.801585...


def icc_oneway(values: np.ndarray, episode_ids: np.ndarray) -> float:
    """One-way random-effects ICC(1) of `values` clustered by episode (unequal sizes)."""
    x = np.asarray(values, np.float64)
    ids = np.asarray(episode_ids)
    uniq = np.unique(ids)
    k = len(uniq)
    N = len(x)
    if k < 2 or k >= N:
        return 0.0
    grand = x.mean()
    ssb = 0.0
    ssw = 0.0
    sum_ni2 = 0.0
    for u in uniq:
        xi = x[ids == u]
        ni = len(xi)
        sum_ni2 += ni * ni
        mi = xi.mean()
        ssb += ni * (mi - grand) ** 2
        ssw += ((xi - mi) ** 2).sum()
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - sum_ni2 / N) / (k - 1)  # design-adjusted mean cluster size
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return 0.0
    return float(max(0.0, (msb - msw) / denom))


def derive_N(sigma_d: float, m: float, rho_icc: float, delta: float = 0.02) -> dict:
    """Required session count to detect a paired-MAE improvement `delta` (ADR 6.2-6.3)."""
    n_eff = (_Z_SUM**2) * (sigma_d**2) / (delta**2)
    deff = 1.0 + (m - 1.0) * rho_icc
    # ceil = the conservative sample-size floor (never round a power budget DOWN). This can exceed
    # the ADR 6.3 table (which rounds to nearest) by 1 in some cells; the direction is always safe.
    n_sessions = int(math.ceil(n_eff * deff / m)) if m > 0 else float("inf")
    return {
        "n_eff": n_eff,
        "design_effect": deff,
        "n_sessions": n_sessions,
        "delta": delta,
    }


def escalation(effect_ac: float, n_sessions, eta_weeks: float) -> dict:
    """ADR 6.4 pre-registered escalation: shelve-vs-redesign on ANY trigger."""
    reasons = []
    if effect_ac <= 0:
        reasons.append(f"autocorrelated pilot effect {effect_ac:+.4f} <= 0")
    if n_sessions is not None and n_sessions != float("inf") and n_sessions > 300:
        reasons.append(f"derived N={n_sessions} > 300 sessions")
    if eta_weeks > 8:
        reasons.append(f"collection ETA {eta_weeks:.1f} wk > 8 wk")
    return {"escalate": bool(reasons), "reasons": reasons}


def run_pilot(d: np.ndarray, episode_ids: np.ndarray, min_sessions: int = 30) -> dict:
    """Measure sigma_d, m, rho_icc, and the effect from a pilot's per-tick paired differences.

    `d` = per-tick (persistence_err - estimator_err) on the autocorrelated slice (positive = est
    better). `episode_ids` = the session id per tick. Raises if fewer than `min_sessions` clusters.
    """
    d = np.asarray(d, np.float64)
    ids = np.asarray(episode_ids)
    n_sessions = len(np.unique(ids))
    if n_sessions < min_sessions:
        raise ValueError(f"pilot needs >= {min_sessions} sessions, got {n_sessions}")
    m = len(d) / n_sessions  # labeled ticks per session
    return {
        "sigma_d": float(d.std(ddof=1)),
        "m": float(m),
        "rho_icc": icc_oneway(d, ids),
        "effect": float(d.mean()),
        "n_pilot_sessions": int(n_sessions),
    }
