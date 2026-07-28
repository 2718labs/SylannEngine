"""Deterministic sparse-routing contracts for both computation spines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import sylanne_core.compute.brain_backend_coordinator as coordinator_module
from sylanne_core.compute.autopoiesis import AutopoieticBoundary
from sylanne_core.compute.brain_c import BrainCEventCandidate, Route
from sylanne_core.compute.brain_store import BrainStateStore, SessionLoaded, session_digest
from sylanne_core.compute.computation_spine import ComputationSpine
from sylanne_core.compute.predictive_coding import PredictiveCodingGate
from sylanne_core.compute.resonance_integration import ResonanceSpine
from sylanne_core.compute.void_scar_engine import BrainSessionContext
from sylanne_core.config import BrainComputeConfig

SpineKind = Literal["computation", "resonance"]
Spine = ComputationSpine | ResonanceSpine


class _GateSpy:
    def __init__(self, route: Route) -> None:
        self.selected_route = route
        self.calls: list[str] = []
        self.precision = 0.5

    def surprise(self, _input: bytearray | list[int]) -> float:
        self.calls.append("surprise")
        return 0.25

    def route(self, _surprise: float) -> Route:
        self.calls.append("route")
        return self.selected_route

    def update(self, _input: bytearray | list[int], _surprise: float) -> None:
        self.calls.append("update")

    def update_stable(
        self,
        _input: bytearray | list[int],
        _surprise: float,
        event_id: str,
    ) -> None:
        self.calls.append(f"update_stable:{event_id}")

    def mean_surprise(self) -> float:
        return 0.25

    def set_route_thresholds(
        self,
        *,
        fast_threshold: float,
        full_threshold: float,
    ) -> None:
        del fast_threshold, full_threshold

    def to_dict(self) -> dict[str, Any]:
        return {"mean_surprise": self.mean_surprise()}


class _SheafSpy:
    def __init__(self) -> None:
        self.calls = 0

    def tick(
        self,
        _relationship_id: int,
        _local_state: list[float],
        *,
        timestamp: float,
    ) -> dict[str, Any]:
        del timestamp
        self.calls += 1
        return {"energy": 0.0}

    def derive_params(self, _personality: dict[str, float]) -> None:
        return None


class _HGTSpy:
    def __init__(self) -> None:
        self.build_calls = 0
        self.forward_calls = 0
        self._last_attention_weights: list[list[float]] = []
        self._last_active_experts: list[int] = []
        self._last_gate_values: list[float] = []
        self._plasticity = 0.5

    def derive_params(self, _personality: dict[str, float]) -> None:
        return None

    def build_tokens_from_spine(self, **_kwargs: Any) -> list[object]:
        self.build_calls += 1
        return []

    def forward(
        self,
        _tokens: list[object],
        _personality: dict[str, float],
    ) -> list[float]:
        self.forward_calls += 1
        return [0.0, 0.0, 0.0, 0.0]


def _new_spine(
    root: Path,
    kind: SpineKind,
    *,
    sparse_routing: bool,
) -> tuple[Spine, BrainStateStore, str]:
    store = BrainStateStore.start(root)
    session_id = f"routing-{kind}"
    context = BrainSessionContext(
        config=BrainComputeConfig(
            enabled=True,
            c_enabled=True,
            sparse_routing=sparse_routing,
        ),
        store=store,
        session_id=session_id,
    )
    if kind == "computation":
        spine: Spine = ComputationSpine(brain_context=context)
    else:
        spine = ResonanceSpine(brain_context=context)
    return spine, store, session_id


def _install_spies(
    spine: Spine,
    route: Route,
) -> tuple[_GateSpy, _SheafSpy, _HGTSpy]:
    gate = _GateSpy(route)
    sheaf = _SheafSpy()
    hgt = _HGTSpy()
    if isinstance(spine, ComputationSpine):
        spine.gate = cast(Any, gate)
        spine.sheaf = cast(Any, sheaf)
        spine.hgt = cast(Any, hgt)
    else:
        spine._gate = cast(Any, gate)
        spine._sheaf = cast(Any, sheaf)
        spine._hgt = cast(Any, hgt)
    return gate, sheaf, hgt


def _drive_gate(
    gate: PredictiveCodingGate,
    vector: bytearray,
    event_id: str,
) -> Route:
    surprise = gate.surprise(vector)
    route = cast(Route, gate.route(surprise))
    gate.update_stable(vector, surprise, event_id)
    return route


def test_stable_update_ignores_rng_state_and_matches_across_instances() -> None:
    left = PredictiveCodingGate(dim=64)
    right = PredictiveCodingGate(dim=64)
    left._rng.seed(1)
    right._rng.seed(999_999)

    for index in range(24):
        vector = bytearray(((index * 17 + offset * 29) & 0xFF) for offset in range(8))
        event_id = f"event-{index}"
        assert _drive_gate(left, vector, event_id) == _drive_gate(right, vector, event_id)

    assert left.to_dict() == right.to_dict()


def test_stable_update_has_a_cross_process_byte_golden() -> None:
    gate = PredictiveCodingGate(dim=64)

    gate.update_stable(
        bytearray.fromhex("001d3a577491aecb"),
        0.25,
        "golden-event-v1",
    )

    snapshot = gate.to_dict()
    assert snapshot["prediction"] == "AAgAAEAQgAg="
    assert snapshot["precision"] == 0.52
    assert snapshot["surprise_history"] == [0.25]


def test_stable_update_continues_identically_after_snapshot_restore() -> None:
    uninterrupted = PredictiveCodingGate(dim=64)
    uninterrupted.set_route_thresholds(0.05, 0.95)
    for index in range(12):
        vector = bytearray(((index * 11 + offset * 7) & 0xFF) for offset in range(8))
        _drive_gate(uninterrupted, vector, f"prefix-{index}")

    restored = PredictiveCodingGate(dim=64)
    restored.from_dict(uninterrupted.to_dict())
    restored._rng.seed(123_456)
    assert restored.route(0.5) == uninterrupted.route(0.5)

    for index in range(12, 32):
        vector = bytearray(((index * 11 + offset * 7) & 0xFF) for offset in range(8))
        event_id = f"suffix-{index}"
        assert _drive_gate(uninterrupted, vector, event_id) == _drive_gate(
            restored, vector, event_id
        )

    assert uninterrupted.to_dict() == restored.to_dict()


def test_legacy_update_keeps_the_existing_snapshot_key_set() -> None:
    gate = PredictiveCodingGate(dim=64)
    before_keys = set(gate.to_dict())
    gate.set_route_thresholds(0.05, 0.95)

    gate.update(bytearray(b"\xff" * 8), 0.5)

    assert set(gate.to_dict()) == before_keys


@pytest.mark.parametrize(
    ("fast_threshold", "full_threshold"),
    [
        (math.nan, 0.5),
        (0.1, math.inf),
        (-0.1, 0.5),
        (0.6, 0.5),
        (0.1, 1.1),
    ],
)
def test_stable_snapshot_rejects_invalid_route_thresholds(
    fast_threshold: float,
    full_threshold: float,
) -> None:
    gate = PredictiveCodingGate(dim=64)

    with pytest.raises(ValueError, match="route thresholds"):
        gate.from_dict(
            {
                "fast_threshold": fast_threshold,
                "full_threshold": full_threshold,
            }
        )


def test_stable_snapshot_rejects_an_incomplete_threshold_pair() -> None:
    gate = PredictiveCodingGate(dim=64)

    with pytest.raises(ValueError, match="route thresholds"):
        gate.from_dict({"fast_threshold": 0.1})


@pytest.mark.parametrize("kind", ["computation", "resonance"])
@pytest.mark.parametrize(
    ("route", "expected_relaxations", "expected_optional_calls"),
    [("fast", 0, 0), ("normal", 4, 0), ("full", 4, 1)],
)
def test_sparse_route_controls_c_and_optional_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: SpineKind,
    route: Route,
    expected_relaxations: int,
    expected_optional_calls: int,
) -> None:
    spine, store, session_id = _new_spine(tmp_path / f"{kind}-{route}", kind, sparse_routing=True)
    gate, sheaf, hgt = _install_spies(spine, route)
    relaxations: list[int] = []
    c_traces: list[tuple[float, ...]] = []
    committed_events = 0
    boundary_calls = 0
    original_evolve = coordinator_module.evolve_c_event
    original_commit = store.commit_event
    original_perturb = AutopoieticBoundary.perturb

    def record_evolve(*args: Any, **kwargs: Any) -> BrainCEventCandidate:
        candidate = original_evolve(*args, **kwargs)
        relaxations.append(candidate.relaxations)
        c_traces.append(tuple(candidate.c_trace))
        return candidate

    def record_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal committed_events
        committed_events += 1
        return original_commit(*args, **kwargs)

    def record_perturb(self: AutopoieticBoundary, *args: Any, **kwargs: Any) -> Any:
        nonlocal boundary_calls
        boundary_calls += 1
        return original_perturb(self, *args, **kwargs)

    monkeypatch.setattr(coordinator_module, "evolve_c_event", record_evolve)
    monkeypatch.setattr(store, "commit_event", record_commit)
    monkeypatch.setattr(AutopoieticBoundary, "perturb", record_perturb)

    try:
        event_id = f"{kind}-{route}-event"
        result = spine.process("route this", timestamp=100.0, event_id=event_id)

        assert gate.calls[:3] == ["surprise", "route", f"update_stable:{event_id}"]
        assert relaxations == [expected_relaxations]
        if route == "fast":
            assert c_traces and all(value == 0.0 for value in c_traces[0])
        assert committed_events == 1
        assert sheaf.calls == expected_optional_calls
        assert hgt.build_calls == expected_optional_calls
        assert hgt.forward_calls == expected_optional_calls
        if route == "normal":
            assert boundary_calls == 1

        loaded = store.load(session_digest(session_id))
        assert isinstance(loaded, SessionLoaded)
        assert loaded.bundle.b.tick_id == 1
        diagnostics = spine.diagnostics()
        assert diagnostics["last_route"] == route
        assert diagnostics["route_counts"][route] == 1
        if isinstance(spine, ComputationSpine):
            assert result["route"] == route
    finally:
        spine.engine.close() if isinstance(spine, ComputationSpine) else spine._engine.close()
        store.close()


@pytest.mark.parametrize("kind", ["computation", "resonance"])
def test_sparse_disabled_preserves_legacy_brain_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: SpineKind,
) -> None:
    spine, store, _session_id = _new_spine(tmp_path / f"legacy-{kind}", kind, sparse_routing=False)
    gate, sheaf, hgt = _install_spies(spine, "fast")
    routes: list[Route] = []
    original_evolve = coordinator_module.evolve_c_event

    def record_evolve(*args: Any, **kwargs: Any) -> BrainCEventCandidate:
        candidate = original_evolve(*args, **kwargs)
        routes.append(candidate.route)
        return candidate

    monkeypatch.setattr(coordinator_module, "evolve_c_event", record_evolve)

    try:
        spine.process("legacy route", timestamp=100.0, event_id=f"legacy-{kind}-event")

        assert gate.calls[:2] == ["surprise", "update"]
        assert "route" not in gate.calls
        assert not any(call.startswith("update_stable:") for call in gate.calls)
        assert routes == ["normal"]
        assert sheaf.calls == 1
        assert hgt.build_calls == 1
        assert hgt.forward_calls == 1
    finally:
        spine.engine.close() if isinstance(spine, ComputationSpine) else spine._engine.close()
        store.close()


def test_targeted_feedback_does_not_run_event_routing_layers(
    tmp_path: Path,
) -> None:
    spine, store, session_id = _new_spine(tmp_path / "feedback", "computation", sparse_routing=True)
    assert isinstance(spine, ComputationSpine)
    gate, sheaf, hgt = _install_spies(spine, "normal")

    try:
        spine.process("event", timestamp=100.0, event_id="event-1")
        gate.calls.clear()
        sheaf.calls = 0
        hgt.build_calls = 0
        hgt.forward_calls = 0

        spine.engine.apply_targeted_feedback(
            feedback_id="feedback-1",
            target_tick=1,
            value=0.75,
            confidence=1.0,
        )

        assert gate.calls == []
        assert sheaf.calls == 0
        assert hgt.build_calls == 0
        assert hgt.forward_calls == 0
        loaded = store.load(session_digest(session_id))
        assert isinstance(loaded, SessionLoaded)
        assert loaded.bundle.b.tick_id == 1
    finally:
        spine.engine.close()
        store.close()
