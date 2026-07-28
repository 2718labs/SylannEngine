"""Integration contracts for the authoritative B state across the legacy spines."""

from __future__ import annotations

import ast
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from sylanne_core.compute.brain_backend import register_brain_backend
from sylanne_core.compute.brain_backend_coordinator import BrainBackendCoordinator
from sylanne_core.compute.brain_compute import BrainComputeCore
from sylanne_core.compute.brain_errors import BrainDurabilityError, BrainOwnershipError
from sylanne_core.compute.brain_store import (
    BrainStateStore,
    EventDuplicate,
    SessionLoaded,
    event_id_digest,
    session_digest,
)
from sylanne_core.compute.computation_spine import ComputationSpine
from sylanne_core.compute.host import SylanneAlphaHost, SylanneAlphaHostEvent
from sylanne_core.compute.kernel import AlphaKernelEvent
from sylanne_core.compute.resonance_integration import ResonanceSpine
from sylanne_core.compute.scar_algebra import ScarredState
from sylanne_core.compute.void_calculus import VoidSpace
from sylanne_core.compute.void_scar_engine import (
    BrainSessionContext,
    VoidScarEngine,
    project_brain_assessment,
)
from sylanne_core.config import BrainComputeConfig


def _brain_config(*, c_enabled: bool = False) -> BrainComputeConfig:
    return BrainComputeConfig(enabled=True, c_enabled=c_enabled)


def _brain_host(
    tmp_path: Path,
    store: BrainStateStore,
    *,
    session_id: str = "session",
    c_enabled: bool = False,
) -> SylanneAlphaHost:
    return SylanneAlphaHost(
        root=tmp_path,
        session_key=session_id,
        brain_compute=_brain_config(c_enabled=c_enabled),
        brain_store=store,
    )


def _loaded(store: BrainStateStore, session_id: str = "session") -> SessionLoaded:
    result = store.load(session_digest(session_id))
    assert isinstance(result, SessionLoaded)
    return result


def _event(event_id: str, text: str = "same text") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "text": text,
        "confidence": 0.9,
        "flags": ["hurt", "boundary"],
        "now": 1000.0,
    }


def _assessment() -> dict[str, Any]:
    return {
        "confidence": 0.9,
        "flags": ["hurt", "negative"],
        "valence": -0.8,
        "arousal": 0.7,
        "wound_risk": 0.9,
        "intent": "boundary",
    }


def test_disabled_base_preserves_mutable_list_contract() -> None:
    state = ScarredState()

    assert isinstance(state.base, list)
    state.base[0] = 0.25

    assert state.base[0] == 0.25


def test_brain_base_is_immutable_and_requires_matching_capability() -> None:
    capability = object()
    state = ScarredState(
        brain_capability=capability,
        authoritative_base=(0.1,) * 8,
    )

    assert state.base == (0.1,) * 8
    assert isinstance(state.base, tuple)
    with pytest.raises(TypeError):
        state.base[0] = 0.5  # type: ignore[index]
    with pytest.raises(BrainOwnershipError):
        state._replace_base((0.2,) * 8, object())

    state._replace_base((0.3,) * 8, capability)
    assert state.base == (0.3,) * 8


def test_brain_session_context_does_not_expose_writer_capability(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id="private-capability",
        )

        assert not hasattr(context, "capability")
    finally:
        store.close()


def test_static_guard_finds_no_uncontrolled_private_base_write() -> None:
    allowed_writes = {
        ("sylanne_core/compute/scar_algebra.py", "__init__"),
        ("sylanne_core/compute/scar_algebra.py", "_replace_base"),
    }
    allowed_replacements = {
        ("sylanne_core/compute/scar_algebra.py", "__init__"),
        ("sylanne_core/compute/scar_algebra.py", "_replace_legacy_base"),
        ("sylanne_core/compute/void_scar_engine.py", "_replace_runtime"),
        ("sylanne_core/compute/void_scar_engine.py", "apply_targeted_feedback"),
        ("sylanne_core/compute/void_scar_engine.py", "process_event"),
    }
    writes: set[tuple[str, str]] = set()
    replacements: set[tuple[str, str]] = set()

    for path in Path("sylanne_core").rglob("*.py"):
        relative = path.as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            aliases: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    if isinstance(child.value, ast.Attribute) and child.value.attr == "_base":
                        aliases.update(
                            target.id for target in child.targets if isinstance(target, ast.Name)
                        )
                    targets = child.targets
                elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
                    targets = [child.target]
                else:
                    targets = []
                for target in targets:
                    direct = isinstance(target, ast.Attribute) and target.attr == "_base"
                    subscript = isinstance(target, ast.Subscript) and (
                        (isinstance(target.value, ast.Attribute) and target.value.attr == "_base")
                        or (isinstance(target.value, ast.Name) and target.value.id in aliases)
                    )
                    if direct or subscript:
                        writes.add((relative, node.name))
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute) and child.func.attr == "_replace_base":
                        replacements.add((relative, node.name))
                    is_setattr = isinstance(child.func, ast.Name) and child.func.id == "setattr"
                    is_object_setattr = (
                        isinstance(child.func, ast.Attribute) and child.func.attr == "__setattr__"
                    )
                    if (is_setattr or is_object_setattr) and any(
                        isinstance(arg, ast.Constant) and arg.value == "_base" for arg in child.args
                    ):
                        writes.add((relative, node.name))

    assert writes == allowed_writes
    assert replacements == allowed_replacements


def test_host_and_kernel_events_preserve_event_id() -> None:
    host_event = SylanneAlphaHostEvent(event_id="host-event", text="x")
    kernel_event = host_event.to_kernel_event()

    assert kernel_event.event_id == "host-event"
    assert AlphaKernelEvent(event_id="kernel-event").event_id == "kernel-event"


def test_distinct_ids_and_blank_text_each_advance_one_canonical_tick(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store)

        host.on_request(_event("e1"), assessment=_assessment())
        host.on_request(_event("e2"), assessment=_assessment())
        host.on_request(_event("e3", text=""), assessment=None)

        state = _loaded(store).bundle.b
        assert state.tick_id == 3
        assert state.history_epoch == 3
        assert state.mutation_seq == 3
    finally:
        store.close()


def test_one_external_event_commits_once_even_with_assessment_wound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    commits: list[str] = []
    original = store.commit_event

    def record_commit(session_key: bytes, event_key: bytes, commit: Any) -> Any:
        commits.append(commit.receipt.kind)
        return original(session_key, event_key, commit)

    monkeypatch.setattr(store, "commit_event", record_commit)
    try:
        host = _brain_host(tmp_path, store)
        host.on_request(_event("multi-wound"), assessment=_assessment())

        assert commits == ["event"]
        assert _loaded(store).bundle.b.tick_id == 1
    finally:
        store.close()


def test_c_disabled_freezes_complete_c_state_across_events(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, c_enabled=False)

        host.on_request(_event("e1"), assessment=_assessment())
        first = _loaded(store).bundle
        host.on_request(_event("e2", text="different"), assessment=_assessment())
        second = _loaded(store).bundle

        assert second.b.tick_id == first.b.tick_id + 1
        assert second.c == first.c
    finally:
        store.close()


def test_store_failure_escapes_kernel_without_swapping_base_or_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store)
        before_base = host.kernel.computation.engine.scar_state.base
        before_state = host.kernel.computation.engine.brain_state

        def fail_commit(*_args: object, **_kwargs: object) -> None:
            raise BrainDurabilityError("disk full")

        monkeypatch.setattr(store, "commit_event", fail_commit)
        with pytest.raises(BrainDurabilityError, match="disk full"):
            host.on_request(_event("failure"), assessment=_assessment())

        assert host.kernel.computation.engine.scar_state.base == before_base
        assert host.kernel.computation.engine.brain_state == before_state
    finally:
        store.close()


def test_event_recovery_read_failure_preserves_original_durability_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="event-recovery-failure")

        def fail_commit(*_args: object, **_kwargs: object) -> None:
            raise BrainDurabilityError("event commit failed")

        def fail_recovery_load(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("event recovery read failed")

        monkeypatch.setattr(store, "commit_event", fail_commit)
        monkeypatch.setattr(store, "load", fail_recovery_load)

        with pytest.raises(BrainDurabilityError, match="event commit failed"):
            host.on_request(_event("event-recovery-e1"), assessment=_assessment())
    finally:
        store.close()


def test_ownership_error_escapes_kernel_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store)

        def fail_process(*_args: object, **_kwargs: object) -> None:
            raise BrainOwnershipError("stale capability")

        monkeypatch.setattr(type(host.kernel.computation), "process", fail_process)
        with pytest.raises(BrainOwnershipError, match="stale capability"):
            host.on_request(_event("ownership"), assessment=_assessment())
    finally:
        store.close()


def test_ordinary_runtime_error_still_returns_kernel_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = SylanneAlphaHost(root=tmp_path, session_key="legacy")

    def fail_process(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("ordinary layer failure")

    monkeypatch.setattr(type(host.kernel.computation), "process", fail_process)
    surface = host.on_request(_event("legacy-error"))

    assert "decision" in surface
    assert "guard" in surface


def test_runtime_restore_uses_sqlite_base_over_stale_json(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store)
        host.on_request(_event("e1"), assessment=_assessment())
        host.flush()
        durable = tuple(_loaded(store).bundle.b.e)

        snapshot_path = tmp_path / "session.alpha.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["computation"]["engine"]["scar"]["base"] = [0.99] * 8
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

        restored = _brain_host(tmp_path, store)
        assert restored.kernel.computation.engine.scar_state.base == durable
        assert restored.kernel.snapshot()["brain_compute"]["mutation_seq"] == 1
    finally:
        store.close()


def test_computation_spine_brain_mode_bypasses_cache_and_commits_blank(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(config=_brain_config(), store=store, session_id="spine")
        spine = ComputationSpine(brain_context=context)

        spine.process("same", timestamp=1000.0, assessment=_assessment(), event_id="e1")
        spine.process("same", timestamp=1000.0, assessment=_assessment(), event_id="e2")
        spine.process("", timestamp=1000.0, assessment=None, event_id="e3")

        state = _loaded(store, "spine").bundle.b
        assert (state.tick_id, state.history_epoch, state.mutation_seq) == (3, 3, 3)
    finally:
        store.close()


def test_resonance_spine_commits_once_and_blank_is_neutral(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(config=_brain_config(), store=store, session_id="resonance")
        spine = ResonanceSpine(brain_context=context)

        spine.process("x", timestamp=1000.0, assessment=_assessment(), event_id="e1")
        before_blank = _loaded(store, "resonance").bundle.b
        spine.process("", timestamp=1000.0, assessment=None, event_id="e2")

        after_blank = _loaded(store, "resonance").bundle.b
        assert (after_blank.tick_id, after_blank.history_epoch, after_blank.mutation_seq) == (
            2,
            2,
            2,
        )
        assert after_blank.d_plus == before_blank.d_plus
        assert after_blank.d_minus == before_blank.d_minus
    finally:
        store.close()


def test_resonance_spine_strict_error_escapes_fail_soft_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id="resonance-strict",
        )
        spine = ResonanceSpine(brain_context=context)

        def fail_process(*_args: object, **_kwargs: object) -> None:
            raise BrainDurabilityError("resonance strict failure")

        monkeypatch.setattr(VoidScarEngine, "process", fail_process)
        with pytest.raises(BrainDurabilityError, match="resonance strict failure"):
            spine.process("x", timestamp=1000.0, event_id="strict-e1")
    finally:
        store.close()


def test_computation_spine_strict_error_escapes_void_circuit_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(config=_brain_config(), store=store, session_id="strict")
        spine = ComputationSpine(brain_context=context)

        def fail_process(*_args: object, **_kwargs: object) -> None:
            raise BrainDurabilityError("strict store failure")

        monkeypatch.setattr(VoidScarEngine, "process", fail_process)
        with pytest.raises(BrainDurabilityError, match="strict store failure"):
            spine.process("x", timestamp=1000.0, event_id="strict-e1")
    finally:
        store.close()


@pytest.mark.parametrize("failure_mode", ("open", "ordinary"))
def test_computation_spine_void_failure_still_commits_neutral_brain_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(config=_brain_config(), store=store, session_id=failure_mode)
        spine = ComputationSpine(brain_context=context)
        if failure_mode == "open":
            breaker = spine._circuit_breakers["void_scar"]
            breaker._failures = breaker._threshold
            breaker._open_since = time.time()
        else:

            def fail_process(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("ordinary void failure")

            monkeypatch.setattr(VoidScarEngine, "process", fail_process)

        spine.process("x", timestamp=1000.0, event_id=f"{failure_mode}-e1")

        state = _loaded(store, failure_mode).bundle.b
        assert (state.tick_id, state.history_epoch, state.mutation_seq) == (1, 1, 1)
    finally:
        store.close()


def test_host_serializes_two_concurrent_brain_events(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="threads")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(host.on_request, _event(f"thread-{index}"), _assessment())
                for index in range(2)
            ]
            for future in futures:
                future.result(timeout=10.0)

        state = _loaded(store, "threads").bundle.b
        assert (state.tick_id, state.history_epoch, state.mutation_seq) == (2, 2, 2)
    finally:
        store.close()


def test_post_ack_core_finalize_failure_recovers_store_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="core-finalize")
        original = BrainComputeCore.commit
        failed = False

        def fail_once(self: BrainComputeCore, candidate: Any) -> Any:
            nonlocal failed
            if not failed:
                failed = True
                raise BrainOwnershipError("core finalize")
            return original(self, candidate)

        monkeypatch.setattr(BrainComputeCore, "commit", fail_once)
        with pytest.raises(BrainOwnershipError, match="core finalize"):
            host.on_request(_event("core-e1"), assessment=_assessment())

        durable = _loaded(store, "core-finalize").bundle.b
        assert host.kernel.computation.engine.brain_state == durable
        assert host.kernel.computation.engine.scar_state.base == tuple(durable.e)
    finally:
        store.close()


def test_post_ack_base_finalize_failure_recovers_store_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="base-finalize")
        original = ScarredState._replace_base
        failed = False

        def fail_once(self: ScarredState, candidate: Any, capability: object | None) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise BrainOwnershipError("base finalize")
            original(self, candidate, capability)

        monkeypatch.setattr(ScarredState, "_replace_base", fail_once)
        with pytest.raises(BrainOwnershipError, match="base finalize"):
            host.on_request(_event("base-e1"), assessment=_assessment())

        durable = _loaded(store, "base-finalize").bundle.b
        assert host.kernel.computation.engine.brain_state == durable
        assert host.kernel.computation.engine.scar_state.base == tuple(durable.e)
    finally:
        store.close()


def test_post_ack_coordinator_finalize_failure_recovers_store_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="c-finalize", c_enabled=True)

        def fail_after_store_ack(
            self: BrainBackendCoordinator, candidate: Any, store_commit: Any
        ) -> Any:
            store_commit(candidate)
            raise BrainDurabilityError("coordinator finalize")

        monkeypatch.setattr(BrainBackendCoordinator, "commit", fail_after_store_ack)
        with pytest.raises(BrainDurabilityError, match="coordinator finalize"):
            host.on_request(_event("c-e1"), assessment=_assessment())

        durable = _loaded(store, "c-finalize").bundle.b
        assert host.kernel.computation.engine.brain_state == durable
        assert host.kernel.computation.engine.scar_state.base == tuple(durable.e)
    finally:
        store.close()


def test_targeted_feedback_updates_target_without_advancing_event_clock(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id="feedback",
        )
        VoidScarEngine(brain_context=context)
        context.process_event(
            event_id="event-e1",
            assessment=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            hdc=[0.0] * 8,
            wound_sum=[0.0] * 8,
            surprise=0.0,
            perception_acuity=0.5,
            route="normal",
        )
        context.process_event(
            event_id="event-e2",
            assessment=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            hdc=[0.0] * 8,
            wound_sum=[0.0] * 8,
            surprise=0.0,
            perception_acuity=0.5,
            route="normal",
        )
        before = _loaded(store, "feedback").bundle.b
        traces = {record.tick_id: tuple(record.b_trace) for record in before.eligibility_records}

        assert traces[1] != traces[2]

        receipt = context.apply_targeted_feedback(
            feedback_id="feedback-f1",
            target_tick=1,
            value=1.0,
            confidence=1.0,
        )

        after = _loaded(store, "feedback").bundle.b
        assert receipt["status"] in {"applied", "degraded"}
        assert after.tick_id == before.tick_id == 2
        assert after.history_epoch == before.history_epoch == 2
        assert (before.mutation_seq, after.mutation_seq) == (2, 3)

        gain_delta = tuple(new - old for new, old in zip(after.gain_b, before.gain_b, strict=True))
        nonzero = [
            (delta, trace)
            for delta, trace in zip(gain_delta, traces[1], strict=True)
            if trace > 0.0
        ]
        assert nonzero
        scale = nonzero[0][0] / nonzero[0][1]
        assert scale > 0.0
        for delta, trace in zip(gain_delta, traces[1], strict=True):
            assert delta == pytest.approx(scale * trace, rel=1e-12, abs=1e-15)
    finally:
        store.close()


def test_feedback_recovery_read_failure_preserves_original_durability_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id="feedback-recovery-failure",
        )
        engine = VoidScarEngine(brain_context=context)
        engine.process(
            event_vec=b"\x00" * 128,
            ssm_input=[0.0] * 8,
            surprise=0.0,
            event_id="feedback-recovery-e1",
        )
        original_load = store.load
        load_calls = 0

        def fail_commit(*_args: object, **_kwargs: object) -> None:
            raise BrainDurabilityError("feedback commit failed")

        def fail_recovery_load(*args: object, **kwargs: object) -> Any:
            nonlocal load_calls
            load_calls += 1
            if load_calls == 1:
                return original_load(*args, **kwargs)
            raise RuntimeError("feedback recovery read failed")

        monkeypatch.setattr(store, "commit_feedback", fail_commit)
        monkeypatch.setattr(store, "load", fail_recovery_load)

        with pytest.raises(BrainDurabilityError, match="feedback commit failed"):
            engine.apply_targeted_feedback(
                feedback_id="feedback-recovery-f1",
                target_tick=1,
                value=1.0,
                confidence=1.0,
            )
    finally:
        store.close()


def test_assessment_projection_precedes_void_and_preserves_exact_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    captured: dict[str, Any] = {}
    original = BrainSessionContext.process_event

    def capture(self: BrainSessionContext, **kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return original(self, **kwargs)

    def no_coupling(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"coupling_events": []}

    monkeypatch.setattr(BrainSessionContext, "process_event", capture)
    monkeypatch.setattr(VoidSpace, "process", no_coupling)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id="projection",
        )
        spine = ResonanceSpine(brain_context=context)
        spine.apply_personality({"neuroticism": 0.83})
        assessment = _assessment()
        spine.process(
            "projection",
            timestamp=1000.0,
            assessment=assessment,
            event_id="projection-e1",
        )

        projected, wound = project_brain_assessment(assessment)
        assert captured["assessment"] == projected
        assert captured["wound_sum"] == wound
        assert len(captured["hdc"]) == 8
        assert captured["perception_acuity"] == pytest.approx(0.83)
    finally:
        store.close()


@pytest.mark.parametrize("spine_kind", ("computation", "resonance"))
def test_brain_mode_bypasses_legacy_base_writers_for_event_and_broad_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spine_kind: str,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        context = BrainSessionContext(
            config=_brain_config(),
            store=store,
            session_id=f"writer-spy-{spine_kind}",
        )
        spine: ComputationSpine | ResonanceSpine
        if spine_kind == "computation":
            spine = ComputationSpine(brain_context=context)
        else:
            spine = ResonanceSpine(brain_context=context)

        def forbidden_writer(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("legacy base writer reached in brain mode")

        monkeypatch.setattr(ScarredState, "_affect_decay", forbidden_writer)
        monkeypatch.setattr(ScarredState, "_evolve_base", forbidden_writer)
        monkeypatch.setattr(ScarredState, "_replace_legacy_base", forbidden_writer)
        monkeypatch.setattr(ComputationSpine, "apply_assessment", forbidden_writer)
        monkeypatch.setattr(
            ResonanceSpine,
            "_apply_assessment_to_engine",
            forbidden_writer,
        )

        spine.process(
            "writer isolation",
            timestamp=1000.0,
            assessment=_assessment(),
            event_id=f"writer-spy-{spine_kind}-e1",
        )
        spine.feedback("rejected")

        state = _loaded(store, f"writer-spy-{spine_kind}").bundle.b
        assert (state.tick_id, state.history_epoch, state.mutation_seq) == (1, 1, 1)
    finally:
        store.close()


def test_multiple_coupling_wounds_are_aggregated_componentwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    captured: dict[str, Any] = {}
    original = BrainSessionContext.process_event

    def capture(self: BrainSessionContext, **kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return original(self, **kwargs)

    def two_couplings(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "coupling_events": [
                {"dim_hint": 2, "pressure": 1.0},
                {"dim_hint": 2, "pressure": 2.0},
            ]
        }

    monkeypatch.setattr(BrainSessionContext, "process_event", capture)
    monkeypatch.setattr(VoidSpace, "process", two_couplings)
    try:
        context = BrainSessionContext(config=_brain_config(), store=store, session_id="wounds")
        engine = VoidScarEngine(brain_context=context)

        engine.process(
            event_vec=b"\x00" * 128,
            ssm_input=[0.0] * 8,
            surprise=0.0,
            event_id="wounds-e1",
        )

        assert captured["wound_sum"] == pytest.approx(
            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            abs=1e-15,
        )
    finally:
        store.close()


def test_blank_brain_event_cannot_inherit_void_coupling_wounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forced_coupling(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"coupling_events": [{"dim_hint": 0, "pressure": 10.0}]}

    monkeypatch.setattr(VoidSpace, "process", forced_coupling)
    store = BrainStateStore.start(tmp_path)
    try:
        host = _brain_host(tmp_path, store, session_id="blank-neutral")
        host.on_request(_event("blank-e1", text=""), assessment=None)

        state = _loaded(store, "blank-neutral").bundle.b
        assert tuple(state.d_plus) == (0.0,) * 8
        assert tuple(state.d_minus) == (0.0,) * 8
    finally:
        store.close()


def test_c_backend_runtime_failure_degrades_to_durable_builtin(
    tmp_path: Path,
) -> None:
    backend_name = "integration-failing-backend"

    class FailingBackend:
        is_process_isolated = True

        def open(self, *_args: object, **_kwargs: object) -> None:
            return None

        def step(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("backend step failed")

        def apply_feedback(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("backend feedback failed")

        def checkpoint(self, *_args: object, **_kwargs: object) -> bytes:
            return b"unused"

        def abort(self, _reason: str) -> None:
            return None

        def close(self) -> None:
            return None

    def factory() -> FailingBackend:
        return FailingBackend()

    register_brain_backend(backend_name, factory)
    store = BrainStateStore.start(tmp_path)
    try:
        host = SylanneAlphaHost(
            root=tmp_path,
            session_key="degraded",
            brain_compute=BrainComputeConfig(
                enabled=True,
                c_enabled=True,
                c_backend=backend_name,
            ),
            brain_store=store,
        )
        host.on_request(_event("degraded-e1"), assessment=_assessment())

        loaded = _loaded(store, "degraded")
        duplicate = store.lookup_event_receipt(
            session_digest("degraded"),
            event_id_digest("degraded-e1"),
        )
        assert isinstance(duplicate, EventDuplicate)
        assert duplicate.receipt.status == "degraded"
        assert loaded.bundle.b.tick_id == 1
        assert len(loaded.bundle.c.eligibility_records) == 1
    finally:
        store.close()


def test_c_backend_healthy_primary_advances_across_events_without_stale(
    tmp_path: Path,
) -> None:
    # Regression (P0, optional-backend prerequisite): a HEALTHY isolated primary that
    # actually commits must be drivable across many events. The engine used to
    # hardcode BrainStepRequest.expected_state_version=0, so once the coordinator's
    # primary_state_version advanced past 0 (after the first committed checkpoint)
    # the caller-side stale guard rejected event 2+ with BrainValidationError
    # ('stale') OUTSIDE the degrade path — a hard engine failure. Only a failing
    # backend was ever exercised at the engine level, so this healthy path (the
    # entire reason a non-builtin backend exists) was uncovered and broken.
    from sylanne_core.compute.brain_backend import BrainFeedbackResult, BrainStepResult

    backend_name = "integration-healthy-backend"

    class HealthyBackend:
        is_process_isolated = True

        def open(self, *_args: object, **_kwargs: object) -> None:
            return None

        def step(self, request: Any, **_kwargs: object) -> BrainStepResult:
            return BrainStepResult(
                request_id=request.request_id,
                event_id=request.event_id,
                tick_id=request.tick_id,
                expected_state_version=request.expected_state_version,
                state_version=request.expected_state_version + 1,
                proposal=(0.0,) * 8,
                eligibility=(0.0,) * 128,
            )

        def apply_feedback(self, request: Any, **_kwargs: object) -> BrainFeedbackResult:
            return BrainFeedbackResult(
                request_id=request.request_id,
                feedback_id=request.feedback_id,
                target_tick=request.target_tick,
                expected_state_version=request.expected_state_version,
                state_version=request.expected_state_version,
                applied_synapses=0,
            )

        def checkpoint(self, **_kwargs: object) -> bytes:
            return b"healthy-checkpoint-token"

        def abort(self, _reason: str) -> None:
            return None

        def close(self) -> None:
            return None

    register_brain_backend(backend_name, HealthyBackend)
    store = BrainStateStore.start(tmp_path)
    try:
        host = SylanneAlphaHost(
            root=tmp_path,
            session_key="healthy",
            brain_compute=BrainComputeConfig(
                enabled=True,
                c_enabled=True,
                c_backend=backend_name,
            ),
            brain_store=store,
        )
        # Event 1 commits and advances primary_state_version 0 -> 1; without the
        # fix, event 2 raises BrainValidationError('stale'). Drive several to be sure.
        for index in range(1, 6):
            host.on_request(_event(f"healthy-e{index}"), assessment=_assessment())

        loaded = _loaded(store, "healthy")
        assert loaded.bundle.b.tick_id == 5
    finally:
        store.close()
