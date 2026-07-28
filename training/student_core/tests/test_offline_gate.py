import numpy as np
import offline_gate as og
import pandas as pd
from rung_registration import RungRegistrationV1


def _synth_corpus(path, n_sessions=40, rho=0.9, seed=0):
    """Small AR(1) affect corpus whose residual is UNPREDICTABLE from the features -> NO-GO."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        n = int(rng.integers(18, 30))
        v = 0.0
        ar_a = 0.5
        ts = 1_000_000.0
        for t in range(n):
            v = float(np.clip(rho * v + 0.2 * rng.normal(), -1, 1))
            ar_a = float(np.clip(rho * ar_a + 0.1 * rng.normal(), 0, 1))
            ts += float(rng.uniform(30, 300))
            rows.append(
                {
                    "session": s,
                    "tick": t,
                    "ts": ts,
                    "is_iid": False,  # all autocorrelated
                    "a_valence": v,
                    "a_arousal": ar_a,
                    "surprise": float(rng.uniform(0, 1)),
                    "base_pre_nudge": [float(x) for x in rng.normal(size=8)],
                    "hdc64": [float(x) for x in rng.normal(size=64)],
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_persistence_rung_equals_persistence_exactly(tmp_path):
    """End-to-end anchor property: a no-feature rung's estimator MAE == persistence MAE, bit-for-bit."""
    corpus = _synth_corpus(tmp_path / "c.parquet", seed=1)
    reg = RungRegistrationV1(
        rung_id="persistence",
        primary_estimator="ResidualAffectProbeV1",
        feature_blocks=(),  # anchor only
        baselines=("persistence", "steelman_ridge", "steelman_gbm"),
    )
    r = og.run_gate(str(corpus), registration=reg)
    assert r["estimator_mae_ac"] == r["baselines_ac"]["persistence"]
    assert r["verdict"] == "NO-GO"  # cannot beat itself


def test_full_rung_is_nogo_on_unpredictable_residual(tmp_path):
    """The real design claim: on data where the residual is noise, the gate honestly reads NO-GO."""
    corpus = _synth_corpus(tmp_path / "c.parquet", seed=2)
    r = og.run_gate(str(corpus))  # default = full vfull rung
    assert r["verdict"] == "NO-GO"
    # and it does not manufacture a >=15% win over persistence
    assert r["per_baseline"]["persistence"]["rel"] < 0.15


def test_baseline_verdict_requires_powered_ci():
    """A GO must be POWERED: point over the floors is not enough — the CI lower bound must clear it."""
    # point rel below floor -> False regardless of CI
    assert og._baseline_verdict(0.20, 0.178, 0.05, 0.15, 0.02)["beats"] is False
    # point rel+abs clear the floors, but the CI straddles the floor -> underpowered -> False
    assert og._baseline_verdict(0.20, 0.16, 0.01, 0.15, 0.02)["beats"] is False
    # point AND CI lower bound clear the abs floor -> powered GO -> True
    assert og._baseline_verdict(0.20, 0.16, 0.03, 0.15, 0.02)["beats"] is True


def test_gate_report_shape(tmp_path):
    corpus = _synth_corpus(tmp_path / "c.parquet", seed=3)
    r = og.run_gate(str(corpus), seed_base=7)
    assert set(r["per_baseline"]) == {"persistence", "steelman_ridge", "steelman_gbm"}
    assert r["verdict"] in ("GO", "NO-GO")
    assert len(r["registration_digest"]) == 64
    # Lens2 fixes: seed_base is recorded; the executed slice is the registered rule and it agrees
    # with the generator flag (all AR sessions -> autocorrelated) on this corpus.
    assert r["seed_base"] == 7
    assert "derive_is_iid" in r["slice_source"]
    assert r["slice_vs_generator_agreement"] >= 0.95
