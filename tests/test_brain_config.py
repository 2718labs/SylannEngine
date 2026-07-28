"""Public configuration and backend contracts for opt-in brain compute."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import AsyncMock

import pytest

from sylanne_core import (
    BrainComputeConfig,
    FeedbackReceipt,
    SylanneConfig,
    SylanneEngine,
    get_brain_backend_factory,
    register_brain_backend,
)
from sylanne_core._config_store import CONFIG_FILENAME, load_config
from sylanne_core.compute import brain_backend as brain_backend_registry
from sylanne_core.compute.brain_backend import (
    BrainBackend,
    BrainBackendFactory,
    BrainFeedbackRequest,
    BrainFeedbackResult,
    BrainStepRequest,
    BrainStepResult,
)


@pytest.fixture
def isolated_brain_backend_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, BrainBackendFactory]:
    registry = {"builtin": brain_backend_registry._BACKEND_FACTORIES["builtin"]}
    monkeypatch.setattr(brain_backend_registry, "_BACKEND_FACTORIES", registry)
    return registry


def test_brain_config_defaults_off_and_frozen() -> None:
    brain = BrainComputeConfig()
    assert brain == BrainComputeConfig(
        enabled=False,
        c_enabled=False,
        sparse_routing=False,
        c_backend="builtin",
        c_authority=0.0,
        c_residual_cap=0.1,
        c_promoted=False,
        c_timeout_ms=30.0,
        feedback_horizon=8,
        feedback_ttl_seconds=7200.0,
        dedup_horizon=256,
        hot_session_limit=48,
    )
    assert brain.enabled is False
    with pytest.raises(FrozenInstanceError):
        brain.enabled = True  # type: ignore[misc]


def test_sylanne_config_contains_default_brain_config() -> None:
    cfg = SylanneConfig()
    assert cfg.brain_compute == BrainComputeConfig()


@pytest.mark.parametrize("field_name", ["enabled", "c_enabled", "sparse_routing", "c_promoted"])
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_brain_boolean_flags_require_actual_bool(field_name: str, value: object) -> None:
    kwargs: dict[str, object] = {"enabled": True, field_name: value}
    with pytest.raises(ValueError, match=rf"{field_name} must be bool"):
        BrainComputeConfig(**kwargs)  # type: ignore[arg-type]


def test_c_promoted_requires_enabled_and_positive_authority() -> None:
    # Promotion is the operator sign-off that clears the canary; it only means
    # anything when the C path is on and has authority to grant.
    ok = BrainComputeConfig(enabled=True, c_authority=0.05, c_promoted=True)
    assert ok.c_promoted is True
    with pytest.raises(ValueError, match="c_promoted requires enabled=True and c_authority > 0"):
        BrainComputeConfig(enabled=True, c_authority=0.0, c_promoted=True)
    with pytest.raises(ValueError, match="nonzero c_authority requires enabled=True"):
        BrainComputeConfig(enabled=False, c_authority=0.05, c_promoted=True)


def test_nested_brain_config_loads(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "brain_compute": {
                    "enabled": True,
                    "c_enabled": True,
                    "sparse_routing": True,
                    "c_authority": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg, block = load_config(tmp_path)

    assert cfg.brain_compute == BrainComputeConfig(
        enabled=True,
        c_enabled=True,
        sparse_routing=True,
        c_authority=0.05,
    )
    assert block is None


@pytest.mark.parametrize("field_name", ["enabled", "c_enabled", "sparse_routing"])
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_nested_non_bool_brain_flags_fall_back_to_defaults(
    tmp_path: Path, field_name: str, value: object
) -> None:
    payload = {"enabled": True, "hot_session_limit": 47, field_name: value}
    (tmp_path / CONFIG_FILENAME).write_text(
        json.dumps({"brain_compute": payload}), encoding="utf-8"
    )

    cfg, block = load_config(tmp_path)

    assert cfg == SylanneConfig()
    assert block is None


@pytest.mark.parametrize("value", [[], "enabled", 1])
def test_invalid_nested_brain_config_preserves_fallback_contract(
    tmp_path: Path, value: object
) -> None:
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"brain_compute": value}), encoding="utf-8")

    cfg, block = load_config(tmp_path)

    assert cfg == SylanneConfig()
    assert block is None


def test_brain_compute_rejects_non_lite() -> None:
    for mode in ("pro", "max"):
        with pytest.raises(ValueError, match="requires mode='lite'"):
            SylanneConfig(mode=mode, brain_compute=BrainComputeConfig(enabled=True))


@pytest.mark.parametrize(
    "writer_flag",
    [
        "pel_core_enabled",
        "affect_dynamics_enabled",
        "affect_takeover",
        "affect_slowchannel_enabled",
        "affect_plasticity_enabled",
        "affect_full_takeover",
    ],
)
def test_brain_compute_rejects_each_legacy_writer(writer_flag: str) -> None:
    with pytest.raises(ValueError, match=writer_flag):
        SylanneConfig(
            brain_compute=BrainComputeConfig(enabled=True),
            **{writer_flag: True},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c_enabled": True},
        {"sparse_routing": True},
        {"c_authority": 0.01},
    ],
)
def test_c_features_require_brain_compute_enabled(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="requires enabled=True"):
        BrainComputeConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("authority", [0.0, 0.1])
def test_c_authority_accepts_closed_boundaries(authority: float) -> None:
    assert BrainComputeConfig(enabled=True, c_authority=authority).c_authority == authority


@pytest.mark.parametrize("authority", [-0.000_001, 0.100_001, float("nan"), float("inf")])
def test_c_authority_rejects_values_outside_bounds(authority: float) -> None:
    with pytest.raises(ValueError, match="c_authority"):
        BrainComputeConfig(enabled=True, c_authority=authority)


@pytest.mark.parametrize("cap", [0.0, 0.1])
def test_c_residual_cap_accepts_closed_boundaries(cap: float) -> None:
    assert BrainComputeConfig(enabled=True, c_residual_cap=cap).c_residual_cap == cap


@pytest.mark.parametrize("cap", [-0.000_001, 0.100_001, float("nan"), float("inf")])
def test_c_residual_cap_rejects_values_outside_bounds(cap: float) -> None:
    with pytest.raises(ValueError, match="c_residual_cap"):
        BrainComputeConfig(enabled=True, c_residual_cap=cap)


def test_c_authority_cannot_exceed_residual_cap() -> None:
    with pytest.raises(ValueError, match="c_residual_cap"):
        BrainComputeConfig(enabled=True, c_authority=0.05, c_residual_cap=0.04)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_c_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="c_timeout_ms"):
        BrainComputeConfig(c_timeout_ms=timeout)


@pytest.mark.parametrize("horizon", [1, 32])
def test_feedback_horizon_accepts_closed_boundaries(horizon: int) -> None:
    assert BrainComputeConfig(feedback_horizon=horizon).feedback_horizon == horizon


@pytest.mark.parametrize("horizon", [0, 33])
def test_feedback_horizon_rejects_values_outside_bounds(horizon: int) -> None:
    with pytest.raises(ValueError, match="feedback_horizon"):
        BrainComputeConfig(feedback_horizon=horizon)


@pytest.mark.parametrize(
    "field_name", ["feedback_ttl_seconds", "dedup_horizon", "hot_session_limit"]
)
def test_positive_retention_and_resource_limits_accept_one(field_name: str) -> None:
    config = BrainComputeConfig(**{field_name: 1})  # type: ignore[arg-type]
    assert getattr(config, field_name) == 1


@pytest.mark.parametrize(
    "field_name", ["feedback_ttl_seconds", "dedup_horizon", "hot_session_limit"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_positive_retention_and_resource_limits_reject_nonpositive(
    field_name: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field_name):
        BrainComputeConfig(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_feedback_ttl_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="feedback_ttl_seconds"):
        BrainComputeConfig(feedback_ttl_seconds=value)


@pytest.mark.parametrize("field_name", ["c_timeout_ms", "feedback_ttl_seconds"])
def test_timeout_and_ttl_reject_arbitrarily_large_positive_ints(field_name: str) -> None:
    huge = 10**1000

    with pytest.raises(ValueError, match=field_name):
        BrainComputeConfig(**{field_name: huge})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["c_timeout_ms", "feedback_ttl_seconds"])
def test_nested_timeout_and_ttl_huge_ints_fall_back_to_defaults(
    tmp_path: Path, field_name: str
) -> None:
    huge = 10**1000
    (tmp_path / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "brain_compute": {
                    field_name: huge,
                    "hot_session_limit": 47,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg, block = load_config(tmp_path)

    assert cfg == SylanneConfig()
    assert block is None


def test_unknown_backend_fails_during_engine_construction(tmp_path: Path) -> None:
    config = SylanneConfig(brain_compute=BrainComputeConfig(c_backend="missing-backend"))
    with pytest.raises(ValueError, match="unknown brain backend.*missing-backend"):
        SylanneEngine(tmp_path, llm=AsyncMock(), config=config)


def test_builtin_backend_is_known_during_engine_construction(tmp_path: Path) -> None:
    engine = SylanneEngine(tmp_path, llm=AsyncMock())
    assert engine._config.brain_compute.c_backend == "builtin"
    assert get_brain_backend_factory("builtin") is not None


def test_backend_registry_is_explicit_idempotent_and_rejects_replacement(
    isolated_brain_backend_registry: dict[str, BrainBackendFactory],
) -> None:
    class StubBackend:
        def open(self, session_digest: str, checkpoint_token: bytes | None) -> None:
            pass

        def step(self, request: BrainStepRequest) -> BrainStepResult:
            raise NotImplementedError

        def apply_feedback(self, request: BrainFeedbackRequest) -> BrainFeedbackResult:
            raise NotImplementedError

        def checkpoint(self) -> bytes:
            return b""

        def close(self) -> None:
            pass

    def factory() -> BrainBackend:
        return StubBackend()

    def replacement() -> BrainBackend:
        return StubBackend()

    register_brain_backend("test-explicit", factory)
    register_brain_backend("test-explicit", factory)
    assert get_brain_backend_factory("test-explicit") is factory
    with pytest.raises(ValueError, match="already registered"):
        register_brain_backend("test-explicit", replacement)
    assert isolated_brain_backend_registry == {
        "builtin": brain_backend_registry._BACKEND_FACTORIES["builtin"],
        "test-explicit": factory,
    }


def test_concurrent_backend_registration_has_exactly_one_winner(
    isolated_brain_backend_registry: dict[str, BrainBackendFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowLookupRegistry(dict[str, BrainBackendFactory]):
        def get(
            self, key: str, default: BrainBackendFactory | None = None
        ) -> BrainBackendFactory | None:
            result = super().get(key, default)
            if key == "test-race":
                time.sleep(0.05)
            return result

    def first_factory() -> BrainBackend:
        raise NotImplementedError

    def second_factory() -> BrainBackend:
        raise NotImplementedError

    registry = SlowLookupRegistry(isolated_brain_backend_registry)
    monkeypatch.setattr(brain_backend_registry, "_BACKEND_FACTORIES", registry)
    start = threading.Barrier(3)
    result_lock = threading.Lock()
    outcomes: list[tuple[str, BrainBackendFactory]] = []

    def register(factory: BrainBackendFactory) -> None:
        start.wait()
        try:
            register_brain_backend("test-race", factory)
        except ValueError:
            outcome = "error"
        else:
            outcome = "success"
        with result_lock:
            outcomes.append((outcome, factory))

    threads = [
        threading.Thread(target=register, args=(first_factory,)),
        threading.Thread(target=register, args=(second_factory,)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    successes = [factory for outcome, factory in outcomes if outcome == "success"]
    errors = [factory for outcome, factory in outcomes if outcome == "error"]
    assert len(successes) == 1
    assert len(errors) == 1
    assert get_brain_backend_factory("test-race") is successes[0]


def test_backend_contract_records_are_frozen_and_typed() -> None:
    step = BrainStepRequest(
        event_id="event-1",
        tick_id=3,
        expected_state_version=2,
        event=(0.25, -0.5),
    )
    assert step.event == (0.25, -0.5)
    with pytest.raises(FrozenInstanceError):
        step.tick_id = 4  # type: ignore[misc]

    feedback = BrainFeedbackRequest(
        feedback_id="feedback-1",
        target_tick=3,
        expected_state_version=3,
        value=0.5,
        confidence=0.75,
    )
    assert feedback.target_tick == 3


def test_feedback_receipt_public_shape_status_and_frozen_contract() -> None:
    receipt = FeedbackReceipt(
        status="applied",
        session_id="s1",
        target_tick=7,
        feedback_id="f1",
        applied_dimensions=(1, 3),
        applied_synapses=4,
        mutation_seq=9,
    )
    assert [field.name for field in fields(FeedbackReceipt)] == [
        "status",
        "session_id",
        "target_tick",
        "feedback_id",
        "applied_dimensions",
        "applied_synapses",
        "mutation_seq",
    ]
    assert set(get_args(get_type_hints(FeedbackReceipt)["status"])) == {
        "applied",
        "duplicate",
        "missed",
        "no_effect",
        "disabled",
        "degraded",
    }
    assert receipt.applied_dimensions == (1, 3)
    with pytest.raises(FrozenInstanceError):
        receipt.status = "duplicate"  # type: ignore[misc]


def test_fresh_public_import_does_not_load_optional_accelerators() -> None:
    code = (
        "import sys; import sylanne_core; "
        "loaded = sorted(name for name in ('torch', 'cupy', 'numba') if name in sys.modules); "
        "assert not loaded, f'optional accelerators imported: {loaded}'; print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "OK" in proc.stdout
