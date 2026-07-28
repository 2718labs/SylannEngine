"""offline_gate.py — the ADR-0001 offline value gate (C3 + F2, numpy/sklearn, NO torch).

Runs the persistence-anchored residual probe (residual_affect_probe.ResidualAffectProbeV1) against
BOTH persistence AND a steelmanned field+nudge baseline (ridge + GBM), on the autocorrelated slice,
on a session-disjoint holdout, and applies the ADR PASS bar. PASS iff the estimator beats EACH
baseline by >= rel_floor relative AND >= abs_floor absolute paired-MAE (unambiguous conjunction over
baselines — stronger and clearer than a min/max reference). A clean NO-GO is a success. (v1 reports
point-estimate MAEs; a paired episode-bootstrap CI behind a "well-powered" claim is F3's job.)

Steelman inputs are LEAKAGE-FREE: base_pre_nudge (post-_evolve_base, PRE the assessment nudge, so it
never encodes a_t) + a_prev (a_{t-1}) + surprise. Never z_post (which is post-nudge and encodes a_t).

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, C3/F2).
"""

from __future__ import annotations

import argparse

import numpy as np
from gate_controls import paired_bootstrap_ci
from leakage_guards import (
    assert_no_label_column,
    assert_session_disjoint,
    derive_is_iid,
    variance_floor,
)
from residual_affect_probe import ResidualAffectProbeV1, clip_reads
from rung_registration import RungRegistrationV1

_FAST_DECAY = 0.5


def load_corpus(path: str) -> list[dict]:
    """Group the tick parquet into per-session arrays (sorted by tick, >= 3 ticks)."""
    import pandas as pd

    df = pd.read_parquet(path).sort_values(["session", "tick"]).reset_index(drop=True)
    sessions = []
    for sid, g in df.groupby("session"):
        if len(g) < 3:
            continue
        a = g[["a_valence", "a_arousal"]].to_numpy(np.float64)
        ts = g.ts.to_numpy(np.float64)
        dt = np.log1p(np.clip(np.diff(ts, prepend=ts[0]) / 60.0, 0.0, 60.0))
        sessions.append(
            {
                "sid": int(sid),
                "is_iid": bool(g.is_iid.iloc[0]),
                "a": a,  # (n, 2) valence, arousal
                "surprise": g.surprise.to_numpy(np.float64),  # (n,)
                "base_pre_nudge": np.array(g.base_pre_nudge.tolist(), np.float64),  # (n, 8)
                "hdc64": np.array(g.hdc64.tolist(), np.float64),  # (n, 64)
                "dt": dt,  # (n,)
            }
        )
    return sessions


def _fast_latent(a: np.ndarray) -> np.ndarray:
    """Within-session EMA of prior reads: fl[t] = EMA(a[0..t-1]); fl[0]=a[0]. Pre-a_t state."""
    e = np.empty_like(a)
    e[0] = a[0]
    for t in range(1, len(a)):
        e[t] = (1 - _FAST_DECAY) * e[t - 1] + _FAST_DECAY * a[t]
    # state visible at tick t (before a_t) is e[t-1]; shift up, seed row 0 with a[0]
    fl = np.empty_like(a)
    fl[0] = a[0]
    fl[1:] = e[:-1]
    return fl


_BLOCK_DIMS = {"hdc64": 64, "fast_latent": 2, "base_pre_nudge": 8, "dt": 1}


def _rung_X(s: dict, blocks: tuple[str, ...]) -> np.ndarray:
    """Assemble the estimator's residual-driving features for a session (all strictly pre-a_t)."""
    n = len(s["a"])
    parts = []
    for b in blocks:
        if b == "hdc64":
            parts.append(s["hdc64"])
        elif b == "fast_latent":
            parts.append(_fast_latent(s["a"]))
        elif b == "base_pre_nudge":
            parts.append(s["base_pre_nudge"])
        elif b == "dt":
            parts.append(s["dt"][:, None])
        else:
            raise ValueError(f"unknown feature block {b!r}")
    if not parts:
        return np.zeros((n, 0), np.float64)
    return np.concatenate(parts, axis=1)


def _stack(sessions: list[dict], blocks: tuple[str, ...]):
    """Flatten to per-tick rows for t>=1: (X, a_prev, a_t, steel_X, gen_iid_flag, session_id)."""
    X, a_prev, a_t, steel, iids, sids = [], [], [], [], [], []
    for s in sessions:
        n = len(s["a"])
        Xr = _rung_X(s, blocks)
        for t in range(1, n):
            X.append(Xr[t])
            a_prev.append(s["a"][t - 1])
            a_t.append(s["a"][t])
            # leakage-free steelman inputs: base_pre_nudge + a_prev + surprise
            steel.append(
                np.concatenate([s["base_pre_nudge"][t], s["a"][t - 1], [s["surprise"][t]]])
            )
            iids.append(s["is_iid"])
            sids.append(s["sid"])
    m = len(a_t)
    return (
        np.asarray(X, np.float64).reshape(m, -1),
        np.asarray(a_prev, np.float64),
        np.asarray(a_t, np.float64),
        np.asarray(steel, np.float64),
        np.asarray(iids, bool),
        np.asarray(sids),
    )


def _steelman(Xtr, ytr, Xte, kind: str, reg: RungRegistrationV1):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    preds = []
    for i in range(ytr.shape[1]):
        if kind == "ridge":
            m = Ridge(alpha=reg.ridge_lambda)
        else:
            m = HistGradientBoostingRegressor(
                max_iter=reg.gbm_max_iter, learning_rate=reg.gbm_learning_rate
            )
        m.fit(Xtr, ytr[:, i])
        preds.append(m.predict(Xte))
    return np.stack(preds, axis=1)


def _paired_mae(pred: np.ndarray, a_t: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error over valence+arousal on the masked slice (the paired metric)."""
    err = np.abs(pred[mask] - a_t[mask])
    return float(err.mean())


def _baseline_verdict(
    bmae: float, est_mae: float, ci_lo: float, rel_floor: float, abs_floor: float
) -> dict:
    """Decide whether the estimator beats one baseline — and beats it in a POWERED way.

    A GO must satisfy: point rel >= rel_floor, point abs >= abs_floor, AND the improvement's
    episode-bootstrap CI lower bound >= abs_floor. The CI term is what makes a GO statistically
    robust rather than a point estimate that merely lands over the line with a CI straddling it.
    """
    rel = (bmae - est_mae) / bmae if bmae > 0 else float("nan")
    absimp = bmae - est_mae
    beats = (rel >= rel_floor) and (absimp >= abs_floor) and (ci_lo >= abs_floor)
    return {"baseline_mae": bmae, "rel": rel, "abs": absimp, "beats": beats}


def run_gate(
    corpus_path: str,
    registration: RungRegistrationV1 | None = None,
    seed_base: int = 0,
    test_frac: float = 0.2,
) -> dict:
    """Run one rung of the offline gate; return a GateReport dict with GO/NO-GO."""
    reg = registration or RungRegistrationV1(
        rung_id="vfull",
        primary_estimator="ResidualAffectProbeV1",
        feature_blocks=("hdc64", "fast_latent", "base_pre_nudge", "dt"),
        baselines=("persistence", "steelman_ridge", "steelman_gbm"),
    )
    sessions = load_corpus(corpus_path)

    # Session-disjoint split, seeded by the registration digest (post-hoc change -> new split).
    rng = np.random.default_rng(reg.episode_seed(seed_base))
    idx = rng.permutation(len(sessions))
    n_test = max(1, int(len(sessions) * test_frac))
    train = [sessions[i] for i in idx[n_test:]]
    test = [sessions[i] for i in idx[:n_test]]
    assert_session_disjoint([s["sid"] for s in train], [s["sid"] for s in test])

    Xtr, ap_tr, at_tr, steel_tr, _, _ = _stack(train, reg.feature_blocks)
    Xte, ap_te, at_te, steel_te, gen_iid_te, sids_te = _stack(test, reg.feature_blocks)

    # Slice by the REGISTERED rule (derive_is_iid on the valence series), NOT the generator flag, so
    # the executed analysis matches the frozen slice_rule. Record agreement with the generator flag.
    derived = {s["sid"]: bool(derive_is_iid([s["a"][:, 0]])[0]) for s in test}
    derived_iid_te = np.array([derived[sid] for sid in sids_te], bool)
    slice_agreement = float((derived_iid_te == gen_iid_te).mean())
    ac = ~derived_iid_te  # autocorrelated slice (the ADR PASS bar is measured here)
    if ac.sum() < 10:
        raise RuntimeError(f"autocorrelated test slice too small ({int(ac.sum())})")

    # Defense-in-depth guards before scoring: no feature column equals the label; no dead columns.
    assert_no_label_column(steel_te, at_te)
    variance_floor(steel_te)
    if Xte.shape[1] > 0:
        assert_no_label_column(Xte, at_te)
        variance_floor(Xte)

    # --- estimator: persistence-anchored residual probe -----------------------
    probe = ResidualAffectProbeV1(ridge_lambda=reg.ridge_lambda).fit(Xtr, ap_tr, at_tr)
    est_pred = probe.predict(Xte, ap_te)

    # --- baselines ------------------------------------------------------------
    # Persistence clipped IDENTICALLY to the estimator (fair for any a_prev; no-op on in-range reads).
    persistence_pred = clip_reads(ap_te.copy())
    steel_ridge = _steelman(steel_tr, at_tr, steel_te, "ridge", reg)
    steel_gbm = _steelman(steel_tr, at_tr, steel_te, "gbm", reg)

    # Per-tick mean-over-axes abs error on the ac slice, for the paired episode-bootstrap CI.
    est_err = np.abs(est_pred[ac] - at_te[ac]).mean(axis=1)
    ep_ids = sids_te[ac]
    boot_seed = reg.episode_seed(seed_base)
    preds = {
        "persistence": persistence_pred,
        "steelman_ridge": steel_ridge,
        "steelman_gbm": steel_gbm,
    }
    est_mae = float(est_err.mean())

    # --- coded gate: beat EACH baseline by rel_floor AND abs_floor -------------
    baselines: dict[str, float] = {}
    per_baseline = {}
    passes = True
    for name, pred_b in preds.items():
        b_err = np.abs(pred_b[ac] - at_te[ac]).mean(axis=1)
        bmae = float(b_err.mean())
        baselines[name] = bmae
        _, ci_lo, ci_hi = paired_bootstrap_ci(b_err - est_err, ep_ids, seed=boot_seed)
        v = _baseline_verdict(bmae, est_mae, ci_lo, reg.rel_floor, reg.abs_floor)
        v["abs_ci95"] = [ci_lo, ci_hi]  # cluster (episode) bootstrap on the improvement
        per_baseline[name] = v
        passes = passes and v["beats"]

    return {
        "registration_digest": reg.digest(),
        "seed_base": int(seed_base),  # recorded: the split is (registration_digest XOR seed_base)
        "rung_id": reg.rung_id,
        "feature_blocks": list(reg.feature_blocks),
        "n_sessions": len(sessions),
        "n_test_ac_ticks": int(ac.sum()),
        "slice_source": "derive_is_iid: |lag1_acf(valence)| >= 0.30",
        "slice_vs_generator_agreement": slice_agreement,
        "estimator_mae_ac": est_mae,
        "baselines_ac": baselines,
        "per_baseline": per_baseline,
        "verdict": "GO" if passes else "NO-GO",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="training/student_core/spike_corpus.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = run_gate(args.corpus, seed_base=args.seed)
    print("=== v3core offline value gate (autocorrelated slice) ===")
    print(
        f"  registration digest: {r['registration_digest'][:16]}…  seed_base={r['seed_base']}  rung={r['rung_id']}"
    )
    print(
        f"  sessions={r['n_sessions']}  ac-test-ticks={r['n_test_ac_ticks']}  "
        f"slice={r['slice_source']}  (agree w/ generator flag: {r['slice_vs_generator_agreement'] * 100:.1f}%)"
    )
    print(f"  estimator (ResidualAffectProbeV1) MAE={r['estimator_mae_ac']:.4f}")
    for name, d in r["per_baseline"].items():
        lo, hi = d["abs_ci95"]
        print(
            f"  vs {name:16s} base={d['baseline_mae']:.4f}  "
            f"rel={d['rel'] * 100:+.1f}%  abs={d['abs']:+.4f}  "
            f"abs95%CI=[{lo:+.4f},{hi:+.4f}]  beats={d['beats']}"
        )
    print(
        f"\n  >>> {r['verdict']} <<<  (NO-GO is a success mode: an honest 'no' discharges the gate)"
    )
    if "spike_corpus" in args.corpus:
        print("  NOTE: this is the SYNTHETIC corpus (simulate_corpus.py). Per ADR-0001 a synthetic")
        print("  result is NOT bankable — the field is both generator and baseline. The real-data")
        print("  gate is blocked on Track A collection (salt/consent/deletion). This proves the")
        print("  machinery, not a real verdict.")


if __name__ == "__main__":
    main()
