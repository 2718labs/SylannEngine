import numpy as np
import pilot
import pytest


def test_derive_N_matches_adr_table():
    """ADR-0001 6.2-6.3: sigma_d=0.20, delta=0.02 -> N_eff~=785; m=15, rho=0.10 -> DEFF=2.40, N~=126."""
    d = pilot.derive_N(sigma_d=0.20, m=15, rho_icc=0.10, delta=0.02)
    assert abs(d["n_eff"] - 785) < 5
    assert abs(d["design_effect"] - 2.40) < 0.01
    assert d["n_sessions"] == 126


def test_escalation_fires_on_each_condition():
    assert pilot.escalation(effect_ac=-0.001, n_sessions=100, eta_weeks=4)["escalate"] is True
    assert pilot.escalation(effect_ac=0.03, n_sessions=350, eta_weeks=4)["escalate"] is True
    assert pilot.escalation(effect_ac=0.03, n_sessions=100, eta_weeks=10)["escalate"] is True
    clear = pilot.escalation(effect_ac=0.03, n_sessions=100, eta_weeks=4)
    assert clear["escalate"] is False and clear["reasons"] == []


def test_icc_separates_clustered_from_iid():
    rng = np.random.default_rng(1)
    ids = np.repeat(np.arange(20), 15)
    offsets = rng.normal(0, 1, 20)
    clustered = np.repeat(offsets, 15) + rng.normal(0, 0.1, 300)  # strong between-cluster structure
    assert pilot.icc_oneway(clustered, ids) > 0.8
    iid = rng.normal(0, 1, 300)
    assert pilot.icc_oneway(iid, ids) < 0.2


def test_run_pilot_enforces_min_sessions():
    rng = np.random.default_rng(2)
    d = rng.normal(0.02, 0.2, 200)
    ids_small = np.repeat(np.arange(10), 20)  # only 10 sessions
    with pytest.raises(ValueError):
        pilot.run_pilot(d, ids_small, min_sessions=30)
    ids_ok = np.repeat(np.arange(40), 5)  # 40 sessions
    out = pilot.run_pilot(d, ids_ok, min_sessions=30)
    assert out["n_pilot_sessions"] == 40
    assert out["m"] == 5.0
    assert out["sigma_d"] > 0
