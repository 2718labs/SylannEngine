from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import pytest

import sylanne_core.compute.brain_backend_coordinator as coordinator_module
from sylanne_core.compute.brain_backend import (
    BrainBackend,
    BrainCheckpointMismatchError,
    BrainFeedbackRequest,
    BrainFeedbackResult,
    BrainStepRequest,
    BrainStepResult,
)
from sylanne_core.compute.brain_backend_coordinator import (
    BrainBackendCoordinator,
    CoordinatedFeedback,
    CoordinatedStep,
)
from sylanne_core.compute.brain_c import BrainCState
from sylanne_core.compute.brain_codec import BrainBundle
from sylanne_core.compute.brain_errors import BrainDurabilityError, BrainValidationError
from sylanne_core.compute.brain_state import (
    DEFAULT_LINEAGE_ID,
    BEligibilityRecord,
    BrainState,
    EventAllocation,
    FeedbackAllocation,
)
from sylanne_core.compute.brain_store import (
    AppliedFeedbackCommit,
    BackendCheckpoint,
    BrainStateStore,
    EventAllocated,
    EventCommit,
    EventCommitted,
    EventDuplicate,
    FeedbackAllocated,
    FeedbackCommitted,
    ReceiptOnlyFeedbackCommit,
    SessionLoaded,
    StoredReceipt,
    event_id_digest,
    feedback_id_digest,
)


def test_coordinator_module_exposes_bounded_coordinator() -> None:
    assert BrainBackendCoordinator is not None


SESSION_DIGEST = hashlib.sha256(b"session").digest()
ZERO8 = (0.0,) * 8
ZERO128 = (0.0,) * 128


def _step(
    tick_id: int = 1,
    *,
    request_id: str = "request-1",
    expected_state_version: int = 0,
) -> BrainStepRequest:
    return BrainStepRequest(
        request_id=request_id,
        event_id=f"event-{tick_id}",
        tick_id=tick_id,
        expected_state_version=expected_state_version,
        event=(0.5,) + ZERO8[1:],
    )


def _feedback(*, expected_state_version: int = 0) -> BrainFeedbackRequest:
    return BrainFeedbackRequest(
        request_id="feedback-request-1",
        feedback_id="feedback-1",
        target_tick=1,
        expected_state_version=expected_state_version,
        value=0.0,
        confidence=1.0,
    )


def _checkpoint(token: bytes = b"ack", *, version: int = 0) -> BackendCheckpoint:
    return BackendCheckpoint(
        generation=0,
        backend_name="primary",
        backend_state_version=version,
        acknowledged_mutation_seq=0,
        token=token,
        token_sha256=hashlib.sha256(token).digest(),
    )


def _event_allocation(tick_id: int = 1, mutation_seq: int = 1) -> EventAllocation:
    return EventAllocation(
        generation=0,
        lineage_id=DEFAULT_LINEAGE_ID,
        tick_id=tick_id,
        history_epoch=tick_id,
        mutation_seq=mutation_seq,
    )


def _feedback_allocation(
    *,
    target_tick: int = 1,
    expected_mutation_seq: int = 1,
    next_mutation_seq: int = 2,
) -> FeedbackAllocation:
    return FeedbackAllocation(
        generation=0,
        lineage_id=DEFAULT_LINEAGE_ID,
        target_tick=target_tick,
        expected_mutation_seq=expected_mutation_seq,
        next_mutation_seq=next_mutation_seq,
    )


def _event_ack(candidate: CoordinatedStep) -> EventCommitted:
    checkpoint_digest = (
        None
        if candidate.provisional_checkpoint is None
        else candidate.provisional_checkpoint.token_sha256
    )
    key = (
        candidate.allocation.lineage_id,
        "degraded" if candidate.degraded else "applied",
        checkpoint_digest,
    )
    sequence = _REAL_EVENT_ACK_SEQUENCES.get(key)
    if sequence is None:
        sequence = _RealEventAckSequence(candidate.allocation.lineage_id)
        _REAL_EVENT_ACK_SEQUENCES[key] = sequence
    return sequence.issue(candidate)


def _feedback_ack(candidate: CoordinatedFeedback) -> FeedbackCommitted:
    return _commit_feedback_candidate_in_real_store(candidate)


def _stored_event_material(
    allocated: EventAllocated,
    candidate: CoordinatedStep,
) -> tuple[BrainBundle, StoredReceipt]:
    return _stored_event_material_from_state(
        allocated,
        candidate.c_candidate.state,
        status="degraded" if candidate.degraded else "applied",
    )


def _stored_event_material_from_state(
    allocated: EventAllocated,
    c_state: BrainCState,
    *,
    status: str,
) -> tuple[BrainBundle, StoredReceipt]:
    old = allocated.bundle.b
    clock = float(allocated.allocation.tick_id)
    next_b = BrainState(
        generation=allocated.allocation.generation,
        lineage_id=allocated.allocation.lineage_id,
        e=old.e,
        d_plus=old.d_plus,
        d_minus=old.d_minus,
        gain_b=old.gain_b,
        theta_b=old.theta_b,
        clock=clock,
        tick_id=allocated.allocation.tick_id,
        history_epoch=allocated.allocation.history_epoch,
        mutation_seq=allocated.allocation.mutation_seq,
        eligibility_ring=(
            old.eligibility_records
            + (BEligibilityRecord(allocated.allocation.tick_id, clock, ZERO8),)
        )[-old.eligibility_horizon :],
        eligibility_horizon=old.eligibility_horizon,
        clock_regressions=old.clock_regressions,
    )
    bundle = BrainBundle(next_b, c_state)
    return bundle, StoredReceipt(
        kind="event",
        status=cast(Any, status),
        generation=next_b.generation,
        tick_id=next_b.tick_id,
        history_epoch=next_b.history_epoch,
        mutation_seq=next_b.mutation_seq,
    )


class _RealEventAckSequence:
    def __init__(self, lineage_id: str) -> None:
        self._lineage_id = lineage_id
        self._directory = tempfile.TemporaryDirectory(prefix="sylanne-event-ack-")
        self._issued: dict[int, EventCommitted] = {}
        self._last_tick = 0

    def issue(self, candidate: CoordinatedStep) -> EventCommitted:
        cached = self._issued.get(candidate.request.tick_id)
        if cached is not None:
            return cached
        store = BrainStateStore.start(self._directory.name)
        try:
            for tick in range(self._last_tick + 1, candidate.request.tick_id + 1):
                identifier = event_id_digest(f"event-{tick}")
                with patch(
                    "sylanne_core.compute.brain_store.uuid4",
                    return_value=UUID(self._lineage_id),
                ):
                    allocated = store.preflight_allocate(SESSION_DIGEST, identifier)
                assert isinstance(allocated, EventAllocated)
                is_target = tick == candidate.request.tick_id
                if is_target:
                    c_state = candidate.c_candidate.state
                    status = "degraded" if candidate.degraded else "applied"
                    checkpoint = candidate.provisional_checkpoint
                else:
                    c_state = allocated.bundle.c
                    status = "applied"
                    checkpoint = None
                bundle, receipt = _stored_event_material_from_state(
                    allocated,
                    c_state,
                    status=status,
                )
                committed = store.commit_event(
                    SESSION_DIGEST,
                    identifier,
                    EventCommit(allocated, bundle, receipt, checkpoint),
                )
                assert isinstance(committed, EventCommitted)
                self._issued[tick] = committed
                self._last_tick = tick
            return self._issued[candidate.request.tick_id]
        finally:
            store.close()


_REAL_EVENT_ACK_SEQUENCES: dict[
    tuple[str, str, bytes | None],
    _RealEventAckSequence,
] = {}


def _feedback_status(candidate: CoordinatedFeedback) -> str:
    if candidate.degraded:
        return "degraded"
    if candidate.combined_changed:
        return "applied"
    return candidate.c_candidate.status


def _commit_feedback_candidate_in_real_store(
    candidate: CoordinatedFeedback,
) -> FeedbackCommitted:
    with tempfile.TemporaryDirectory(prefix="sylanne-feedback-ack-") as data_dir:
        store = BrainStateStore.start(data_dir)
        try:
            event_key = event_id_digest(f"feedback-baseline-{candidate.state_tick}")
            with patch(
                "sylanne_core.compute.brain_store.uuid4",
                return_value=UUID(candidate.allocation.lineage_id),
            ):
                event_allocated = store.preflight_allocate(SESSION_DIGEST, event_key)
            assert isinstance(event_allocated, EventAllocated)
            event_bundle, event_receipt = _stored_event_material_from_state(
                event_allocated,
                candidate.c_candidate.state,
                status="applied",
            )
            event_result = store.commit_event(
                SESSION_DIGEST,
                event_key,
                EventCommit(event_allocated, event_bundle, event_receipt),
            )
            assert isinstance(event_result, EventCommitted)

            feedback_key = feedback_id_digest(candidate.request.feedback_id)
            allocated = store.preflight_feedback(
                SESSION_DIGEST,
                feedback_key,
                target_tick=candidate.request.target_tick,
            )
            assert isinstance(allocated, FeedbackAllocated)
            receipt = StoredReceipt(
                kind="feedback",
                status=cast(Any, _feedback_status(candidate)),
                generation=candidate.generation,
                tick_id=candidate.state_tick,
                history_epoch=candidate.state_tick,
                mutation_seq=candidate.candidate_mutation_seq,
                target_tick=candidate.request.target_tick,
            )
            if candidate.combined_changed:
                old = allocated.bundle.b
                next_b = BrainState(
                    generation=allocated.allocation.generation,
                    lineage_id=allocated.allocation.lineage_id,
                    e=old.e,
                    d_plus=old.d_plus,
                    d_minus=old.d_minus,
                    gain_b=old.gain_b,
                    theta_b=old.theta_b,
                    clock=old.clock,
                    tick_id=old.tick_id,
                    history_epoch=old.history_epoch,
                    mutation_seq=allocated.allocation.next_mutation_seq,
                    eligibility_ring=old.eligibility_records,
                    eligibility_horizon=old.eligibility_horizon,
                    clock_regressions=old.clock_regressions,
                )
                result = store.commit_feedback(
                    SESSION_DIGEST,
                    feedback_key,
                    AppliedFeedbackCommit(
                        allocated,
                        BrainBundle(next_b, candidate.c_candidate.state),
                        receipt,
                        candidate.provisional_checkpoint,
                    ),
                )
            else:
                result = store.commit_feedback(
                    SESSION_DIGEST,
                    feedback_key,
                    ReceiptOnlyFeedbackCommit(receipt),
                )
            assert isinstance(result, FeedbackCommitted)
            return result
        finally:
            store.close()


def _prepare_and_commit_candidate_in_real_store(
    data_dir: Any,
    coordinator: BrainBackendCoordinator,
    *,
    session_key: bytes = SESSION_DIGEST,
    event_key: bytes | None = None,
    wrong_checkpoint: bool = False,
) -> tuple[CoordinatedStep, EventCommitted]:
    store = BrainStateStore.start(data_dir)
    try:
        request = _step()
        identifier = event_id_digest(request.event_id) if event_key is None else event_key
        allocated = store.preflight_allocate(session_key, identifier)
        assert isinstance(allocated, EventAllocated)
        candidate = coordinator.prepare_step(
            request,
            allocation=allocated.allocation,
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
        bundle, receipt = _stored_event_material(allocated, candidate)
        selected_checkpoint = candidate.provisional_checkpoint
        if wrong_checkpoint:
            assert selected_checkpoint is not None
            wrong_token = b"wrong-checkpoint-token"
            selected_checkpoint = replace(
                selected_checkpoint,
                token=wrong_token,
                token_sha256=hashlib.sha256(wrong_token).digest(),
            )
        committed = store.commit_event(
            session_key,
            identifier,
            EventCommit(allocated, bundle, receipt, selected_checkpoint),
        )
        assert isinstance(committed, EventCommitted)
        return candidate, committed
    finally:
        store.close()


class ScriptedBackend:
    is_process_isolated = True

    def __init__(self, behavior: object = "ok", *, token: bytes = b"token") -> None:
        self.behavior = behavior
        self.token = token
        self.open_calls: list[tuple[bytes, bytes | None, int]] = []
        self.step_calls: list[tuple[BrainStepRequest, int]] = []
        self.feedback_calls: list[tuple[BrainFeedbackRequest, int]] = []
        self.checkpoint_calls: list[int] = []
        self.abort_reasons: list[str] = []
        self.close_calls = 0

    def open(
        self,
        session_digest: bytes,
        checkpoint_token: bytes | None,
        *,
        timeout_ms: int,
    ) -> None:
        self.open_calls.append((session_digest, checkpoint_token, timeout_ms))
        if self.behavior == "mismatch" and checkpoint_token is not None:
            raise BrainCheckpointMismatchError("checkpoint mismatch")
        if self.behavior == "open-error":
            raise RuntimeError("open failed")

    def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult:
        self.step_calls.append((request, timeout_ms))
        if isinstance(self.behavior, BaseException):
            raise self.behavior
        result = BrainStepResult(
            request_id=request.request_id,
            event_id=request.event_id,
            tick_id=request.tick_id,
            expected_state_version=request.expected_state_version,
            state_version=request.expected_state_version + 1,
            proposal=(0.25,) * 8,
            eligibility=ZERO128,
        )
        if callable(self.behavior):
            return self.behavior(result)
        if self.behavior == "unknown":
            return None  # type: ignore[return-value]
        return result

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult:
        self.feedback_calls.append((request, timeout_ms))
        if isinstance(self.behavior, BaseException):
            raise self.behavior
        return BrainFeedbackResult(
            request_id=request.request_id,
            feedback_id=request.feedback_id,
            target_tick=request.target_tick,
            expected_state_version=request.expected_state_version,
            state_version=request.expected_state_version,
            applied_synapses=0,
        )

    def checkpoint(self, *, timeout_ms: int) -> bytes:
        self.checkpoint_calls.append(timeout_ms)
        if self.behavior == "checkpoint-error":
            raise RuntimeError("checkpoint failed")
        return self.token

    def abort(self, reason: str) -> None:
        self.abort_reasons.append(reason)

    def close(self) -> None:
        self.close_calls += 1


class BackendFactory:
    def __init__(self, *behaviors: object, token: bytes = b"token") -> None:
        self.behaviors = list(behaviors) or ["ok"]
        self.token = token
        self.instances: list[ScriptedBackend] = []

    def __call__(self) -> BrainBackend:
        index = min(len(self.instances), len(self.behaviors) - 1)
        instance = ScriptedBackend(self.behaviors[index], token=self.token)
        self.instances.append(instance)
        return instance


ControlError = type[KeyboardInterrupt] | type[SystemExit]


class LifecycleFaultBackend(ScriptedBackend):
    def __init__(
        self,
        *,
        fault_stage: str | None = None,
        control_error: ControlError = KeyboardInterrupt,
    ) -> None:
        super().__init__()
        self.fault_stage = fault_stage
        self.control_error = control_error

    def _raise_control(self, stage: str) -> None:
        if self.fault_stage == stage:
            raise self.control_error(stage)

    def open(
        self,
        session_digest: bytes,
        checkpoint_token: bytes | None,
        *,
        timeout_ms: int,
    ) -> None:
        super().open(session_digest, checkpoint_token, timeout_ms=timeout_ms)
        self._raise_control("open")

    def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult:
        self._raise_control("step")
        return super().step(request, timeout_ms=timeout_ms)

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult:
        self._raise_control("feedback")
        return super().apply_feedback(request, timeout_ms=timeout_ms)

    def checkpoint(self, *, timeout_ms: int) -> bytes:
        self._raise_control("checkpoint")
        return super().checkpoint(timeout_ms=timeout_ms)

    def abort(self, reason: str) -> None:
        self.abort_reasons.append(reason)
        self._raise_control("abort")

    def close(self) -> None:
        self.close_calls += 1
        self._raise_control("close")


class ExtendedBaseFaultBackend(ScriptedBackend):
    def __init__(self, faults: dict[str, BaseException]) -> None:
        super().__init__()
        self.faults = faults

    def _raise_fault(self, stage: str) -> None:
        error = self.faults.get(stage)
        if error is not None:
            raise error

    def open(
        self,
        session_digest: bytes,
        checkpoint_token: bytes | None,
        *,
        timeout_ms: int,
    ) -> None:
        super().open(session_digest, checkpoint_token, timeout_ms=timeout_ms)
        self._raise_fault("open")

    def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult:
        self._raise_fault("step")
        return super().step(request, timeout_ms=timeout_ms)

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult:
        self._raise_fault("feedback")
        return super().apply_feedback(request, timeout_ms=timeout_ms)

    def checkpoint(self, *, timeout_ms: int) -> bytes:
        self._raise_fault("checkpoint")
        return super().checkpoint(timeout_ms=timeout_ms)

    def abort(self, reason: str) -> None:
        self.abort_reasons.append(reason)
        self._raise_fault("abort")

    def close(self) -> None:
        self.close_calls += 1
        self._raise_fault("close")


class IsolationProbeBackend(ScriptedBackend):
    def __init__(self, result: bool | BaseException) -> None:
        super().__init__()
        self._isolation_result = result

    @property
    def is_process_isolated(self) -> bool:
        if isinstance(self._isolation_result, BaseException):
            raise self._isolation_result
        return self._isolation_result


def _coordinator(
    factory: Callable[[], BrainBackend] | None = None,
    *,
    checkpoint: BackendCheckpoint | None = None,
    alpha: float = 0.075,
    promoted: bool = False,
) -> BrainBackendCoordinator:
    return BrainBackendCoordinator(
        session_digest=SESSION_DIGEST,
        c_state=BrainCState.fresh(),
        primary_name="builtin" if factory is None else "primary",
        primary_factory=factory,
        acknowledged_checkpoint=checkpoint,
        primary_alpha=alpha,
        promoted=promoted,
    )


def test_non_builtin_primary_requires_process_isolation_and_raw_digest() -> None:
    class InProcessBackend(ScriptedBackend):
        is_process_isolated = False

    with pytest.raises(BrainValidationError, match="raw 32 bytes"):
        BrainBackendCoordinator(
            session_digest=b"hex-not-raw",
            c_state=BrainCState.fresh(),
        )
    with pytest.raises(BrainValidationError, match="process isolated"):
        _coordinator(InProcessBackend)


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_primary_factory_control_exceptions_propagate(control_error: ControlError) -> None:
    def factory() -> BrainBackend:
        raise control_error("factory")

    with pytest.raises(control_error, match="factory"):
        _coordinator(factory)


@pytest.mark.parametrize("failure", [RuntimeError("factory"), MemoryError("factory OOM")])
def test_primary_factory_ordinary_failures_degrade_to_builtin(failure: Exception) -> None:
    def factory() -> BrainBackend:
        raise failure

    coordinator = _coordinator(factory)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert candidate.failure_reason is not None
    assert "construction failed" in candidate.failure_reason
    assert coordinator.primary_instance is None


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("fault_stage", ["open", "step", "checkpoint"])
def test_step_lifecycle_control_exceptions_discard_primary_then_propagate(
    control_error: ControlError,
    fault_stage: str,
) -> None:
    instance = LifecycleFaultBackend(fault_stage=fault_stage, control_error=control_error)
    coordinator = _coordinator(lambda: instance)
    with pytest.raises(control_error, match=fault_stage):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_feedback_control_exceptions_discard_primary_then_propagate(
    control_error: ControlError,
) -> None:
    instance = LifecycleFaultBackend(control_error=control_error)
    coordinator = _coordinator(lambda: instance)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    instance.fault_stage = "feedback"
    with pytest.raises(control_error, match="feedback"):
        coordinator.prepare_feedback(
            _feedback(expected_state_version=1),
            allocation=_feedback_allocation(),
            state_tick=1,
            trusted_now=1.0,
            state_clock=1.0,
            feedback_ttl_seconds=10.0,
            b_changed=False,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


@pytest.mark.parametrize("failure", [RuntimeError("feedback"), MemoryError("feedback OOM")])
def test_feedback_ordinary_failures_degrade_and_discard(failure: Exception) -> None:
    instance = ScriptedBackend()
    coordinator = _coordinator(lambda: instance)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    instance.behavior = failure
    candidate = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=20,
    )
    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert instance.abort_reasons
    assert instance.close_calls == 1
    assert coordinator.primary_instance is None


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("fault_stage", ["abort", "close"])
def test_cleanup_control_exceptions_propagate_after_best_effort_discard(
    control_error: ControlError,
    fault_stage: str,
) -> None:
    instance = LifecycleFaultBackend(fault_stage=fault_stage, control_error=control_error)
    instance.behavior = RuntimeError("ordinary primary crash")
    coordinator = _coordinator(lambda: instance)
    with pytest.raises(control_error, match=fault_stage):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_direct_close_control_exception_propagates_but_coordinator_stays_closed(
    control_error: ControlError,
) -> None:
    instance = LifecycleFaultBackend(fault_stage="close", control_error=control_error)
    coordinator = _coordinator(lambda: instance)
    with pytest.raises(control_error, match="close"):
        coordinator.close()
    assert coordinator.primary_instance is None
    with pytest.raises(BrainDurabilityError, match="closed"):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )


@pytest.mark.parametrize("fault_stage", ["open", "step", "checkpoint"])
def test_generator_exit_discards_primary_before_propagating(fault_stage: str) -> None:
    instance = ExtendedBaseFaultBackend({fault_stage: GeneratorExit(fault_stage)})
    coordinator = _coordinator(lambda: instance)
    with pytest.raises(GeneratorExit, match=fault_stage):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


def test_feedback_generator_exit_discards_primary_before_propagating() -> None:
    instance = ExtendedBaseFaultBackend({})
    coordinator = _coordinator(lambda: instance)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    instance.faults["feedback"] = GeneratorExit("feedback")
    with pytest.raises(GeneratorExit, match="feedback"):
        coordinator.prepare_feedback(
            _feedback(expected_state_version=1),
            allocation=_feedback_allocation(),
            state_tick=1,
            trusted_now=1.0,
            state_clock=1.0,
            feedback_ttl_seconds=10.0,
            b_changed=False,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


@pytest.mark.parametrize("cleanup_stage", ["abort", "close"])
def test_generator_exit_preserves_original_across_secondary_cleanup_baseexception(
    cleanup_stage: str,
) -> None:
    instance = ExtendedBaseFaultBackend(
        {
            "step": GeneratorExit("original-step"),
            cleanup_stage: GeneratorExit(f"secondary-{cleanup_stage}"),
        }
    )
    coordinator = _coordinator(lambda: instance)
    with pytest.raises(GeneratorExit, match="original-step"):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    assert coordinator.primary_instance is None
    assert instance.abort_reasons
    assert instance.close_calls == 1


@pytest.mark.parametrize(
    "isolation_result",
    [False, RuntimeError("getter"), GeneratorExit("getter-base")],
)
def test_isolation_probe_failure_cleans_owned_factory_result(
    isolation_result: bool | BaseException,
) -> None:
    instance = IsolationProbeBackend(isolation_result)
    expected_error: type[BaseException]
    if isolation_result is False:
        expected_error = BrainValidationError
    else:
        expected_error = type(isolation_result)
    with pytest.raises(expected_error):
        _coordinator(lambda: instance)
    assert instance.abort_reasons
    assert instance.close_calls == 1


def test_success_is_provisional_until_ack_and_builtin_always_advances_first() -> None:
    factory = BackendFactory()
    coordinator = _coordinator(factory)
    initial = coordinator.c_state

    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )

    assert isinstance(candidate, CoordinatedStep)
    assert candidate.source == "primary"
    assert candidate.degraded is False
    assert candidate.proposal == (0.25,) * 8
    assert candidate.c_candidate.state != initial
    assert coordinator.c_state == initial
    assert coordinator.acknowledged_checkpoint is None
    assert candidate.provisional_checkpoint is not None
    assert factory.instances[0].open_calls == [(SESSION_DIGEST, None, 20)]

    coordinator.acknowledge(candidate, _event_ack(candidate))
    assert coordinator.c_state == candidate.c_candidate.state
    assert coordinator.acknowledged_checkpoint == candidate.provisional_checkpoint


def test_reject_discards_builtin_and_primary_provisional_state() -> None:
    factory = BackendFactory()
    coordinator = _coordinator(factory)
    initial = coordinator.c_state
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )

    coordinator.reject(candidate, "store CAS failed")

    assert coordinator.c_state == initial
    assert coordinator.acknowledged_checkpoint is None
    assert factory.instances[0].abort_reasons == ["store CAS failed"]
    assert factory.instances[0].close_calls == 1
    assert coordinator.primary_instance is None
    assert coordinator.needs_reload is True
    with pytest.raises(BrainDurabilityError, match="reload"):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    coordinator.reload_authoritative(initial, None)
    assert coordinator.needs_reload is False


@pytest.mark.parametrize(
    "behavior",
    [
        TimeoutError("timed out"),
        RuntimeError("crash"),
        MemoryError("OOM"),
        "unknown",
        lambda result: replace(result, request_id="late-request"),
        lambda result: replace(result, event_id="wrong-event"),
        lambda result: replace(result, tick_id=999),
        lambda result: replace(result, expected_state_version=999),
        lambda result: replace(result, state_version=999),
        lambda result: replace(result, proposal=(0.0,) * 7),
        lambda result: replace(result, proposal=(math.nan,) + ZERO8[1:]),
        lambda result: replace(result, proposal=(2.0,) + ZERO8[1:]),
        lambda result: replace(result, eligibility=(0.0,) * 127),
        lambda result: replace(result, eligibility=(math.inf,) + ZERO128[1:]),
        lambda result: replace(result, eligibility=(-0.1,) + ZERO128[1:]),
        lambda result: replace(result, eligibility=(8.1,) + ZERO128[1:]),
        "open-error",
        "checkpoint-error",
    ],
)
def test_primary_failures_abort_close_never_reuse_and_return_current_builtin(
    behavior: Any,
) -> None:
    factory = BackendFactory(behavior)
    coordinator = _coordinator(factory)

    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=5,
    )

    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert candidate.proposal == candidate.c_candidate.proposal
    assert candidate.provisional_checkpoint is None
    assert factory.instances[0].abort_reasons
    assert factory.instances[0].close_calls == 1
    assert coordinator.primary_instance is None


def test_checkpoint_token_ceiling_is_raw_65536_bytes() -> None:
    accepted = BackendFactory(token=b"x" * 65536)
    coordinator = _coordinator(accepted)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert candidate.source == "primary"
    assert candidate.provisional_checkpoint is not None
    assert len(candidate.provisional_checkpoint.token) == 65536

    rejected = BackendFactory(token=b"x" * 65537)
    coordinator = _coordinator(rejected)
    fallback = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert fallback.source == "builtin"
    assert fallback.degraded is True
    assert rejected.instances[0].abort_reasons

    non_bytes = BackendFactory(token=cast(bytes, bytearray(b"not-raw")))
    coordinator = _coordinator(non_bytes)
    fallback = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert fallback.source == "builtin"
    assert fallback.degraded is True


def test_builtin_precedes_primary_and_no_primary_is_a_pure_provisional_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    real_evolve = coordinator_module.evolve_c_event

    def ordered_builtin(*args: Any, **kwargs: Any) -> Any:
        order.append("builtin")
        return real_evolve(*args, **kwargs)

    class OrderedBackend(ScriptedBackend):
        def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult:
            order.append("primary")
            return super().step(request, timeout_ms=timeout_ms)

    monkeypatch.setattr(coordinator_module, "evolve_c_event", ordered_builtin)
    coordinator = _coordinator(OrderedBackend)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert order == ["builtin", "primary"]
    coordinator.reject(candidate, "order observed")

    builtin_only = _coordinator()
    initial = builtin_only.c_state
    split = builtin_only.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert split.source == "builtin"
    assert split.degraded is False
    assert builtin_only.c_state == initial
    builtin_only.reject(split, "pure candidate discarded")
    assert builtin_only.c_state == initial
    assert builtin_only.needs_reload is False


def test_provisional_candidate_blocks_reentry_and_cannot_finalize_twice() -> None:
    coordinator = _coordinator(BackendFactory())
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    with pytest.raises(BrainDurabilityError, match="provisional"):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    coordinator.acknowledge(candidate, _event_ack(candidate))
    with pytest.raises(BrainDurabilityError, match="stale"):
        coordinator.acknowledge(candidate, _event_ack(candidate))
    with pytest.raises(BrainDurabilityError, match="stale"):
        coordinator.reject(candidate, "double reject")


def test_duplicate_store_result_never_acknowledges_provisional_primary_token() -> None:
    factory = BackendFactory(token=b"provisional")
    coordinator = _coordinator(factory)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    duplicate = EventDuplicate(
        _event_ack(candidate).receipt,
        BrainBundle(BrainState.fresh(), BrainCState.fresh()),
    )
    with pytest.raises(BrainDurabilityError, match="positive EventCommitted"):
        coordinator.acknowledge(candidate, cast(Any, duplicate))
    assert coordinator.acknowledged_checkpoint is None
    assert coordinator.needs_reload is True
    assert factory.instances[0].abort_reasons == ["store acknowledgement was not positive"]


def test_positive_store_acknowledgements_are_not_publicly_constructible() -> None:
    event_receipt = StoredReceipt(
        kind="event",
        status="applied",
        generation=0,
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
    )
    feedback_receipt = StoredReceipt(
        kind="feedback",
        status="no_effect",
        generation=0,
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        target_tick=1,
    )
    with pytest.raises(TypeError):
        EventCommitted(event_receipt)
    with pytest.raises(TypeError):
        FeedbackCommitted(feedback_receipt)


def test_real_store_acknowledgement_advances_builtin_and_primary_token(tmp_path: Any) -> None:
    coordinator = _coordinator(BackendFactory(token=b"real-token"))
    candidate, acknowledgement = _prepare_and_commit_candidate_in_real_store(
        tmp_path,
        coordinator,
    )
    coordinator.acknowledge(candidate, acknowledgement)
    assert coordinator.c_state == candidate.c_candidate.state
    assert coordinator.acknowledged_checkpoint == candidate.provisional_checkpoint


@pytest.mark.parametrize("mismatch", ["session", "event_id", "checkpoint"])
def test_real_store_acknowledgement_with_wrong_provenance_fails_closed(
    tmp_path: Any,
    mismatch: str,
) -> None:
    coordinator = _coordinator(BackendFactory(token=b"candidate-token"))
    session_key = SESSION_DIGEST
    identifier = event_id_digest("event-1")
    if mismatch == "session":
        session_key = hashlib.sha256(b"wrong-session").digest()
    elif mismatch == "event_id":
        identifier = event_id_digest("wrong-event")
    candidate, acknowledgement = _prepare_and_commit_candidate_in_real_store(
        tmp_path / mismatch,
        coordinator,
        session_key=session_key,
        event_key=identifier,
        wrong_checkpoint=mismatch == "checkpoint",
    )
    with pytest.raises(BrainDurabilityError, match="acknowledgement"):
        coordinator.acknowledge(candidate, acknowledgement)
    _assert_provisional_failed_closed(coordinator, candidate)


def test_store_acknowledgement_seal_binds_the_complete_payload(tmp_path: Any) -> None:
    coordinator = _coordinator(BackendFactory(token=b"candidate-token"))
    candidate, acknowledgement = _prepare_and_commit_candidate_in_real_store(
        tmp_path,
        coordinator,
        event_key=event_id_digest("wrong-event"),
    )
    tampered = copy(acknowledgement)
    object.__setattr__(tampered, "session_digest", SESSION_DIGEST)
    object.__setattr__(tampered, "id_digest", event_id_digest(candidate.request.event_id))
    object.__setattr__(
        tampered,
        "checkpoint_token_sha256",
        candidate.provisional_checkpoint.token_sha256,
    )

    with pytest.raises(BrainDurabilityError, match="seal"):
        coordinator.acknowledge(candidate, tampered)
    _assert_provisional_failed_closed(coordinator, candidate)


def _assert_provisional_failed_closed(
    coordinator: BrainBackendCoordinator,
    candidate: CoordinatedStep,
) -> None:
    assert coordinator.needs_reload is True
    assert coordinator.primary_instance is None
    assert coordinator.acknowledged_checkpoint is None
    with pytest.raises(BrainDurabilityError, match="stale"):
        coordinator.acknowledge(candidate, cast(Any, object()))


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_stage", ["abort", "close"])
def test_invalid_ack_fails_closed_before_cleanup_control_exception(
    control_error: ControlError,
    cleanup_stage: str,
) -> None:
    instance = LifecycleFaultBackend(control_error=control_error)
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    invalid_ack = copy(_event_ack(candidate))
    invalid_receipt = replace(invalid_ack.receipt, status="degraded")
    object.__setattr__(invalid_ack, "receipt", invalid_receipt)
    instance.fault_stage = cleanup_stage
    with pytest.raises(control_error, match=cleanup_stage):
        coordinator.acknowledge(candidate, invalid_ack)
    _assert_provisional_failed_closed(coordinator, candidate)


@pytest.mark.parametrize("callback_failure", ["ordinary", "control"])
@pytest.mark.parametrize("cleanup_stage", ["abort", "close"])
def test_commit_callback_failure_fails_closed_before_cleanup_control_exception(
    callback_failure: str,
    cleanup_stage: str,
) -> None:
    instance = LifecycleFaultBackend(control_error=SystemExit)
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    instance.fault_stage = cleanup_stage

    def fail_commit(_candidate: CoordinatedStep) -> EventCommitted:
        if callback_failure == "control":
            raise KeyboardInterrupt("callback-control")
        raise RuntimeError("callback-ordinary")

    if callback_failure == "control":
        expected_error: type[BaseException] = KeyboardInterrupt
        message = "callback-control"
    else:
        expected_error = SystemExit
        message = cleanup_stage
    with pytest.raises(expected_error, match=message):
        coordinator.commit(candidate, fail_commit)
    _assert_provisional_failed_closed(coordinator, candidate)


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_stage", ["abort", "close"])
def test_commit_nonpositive_ack_fails_closed_before_cleanup_control_exception(
    control_error: ControlError,
    cleanup_stage: str,
) -> None:
    instance = LifecycleFaultBackend(control_error=control_error)
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    duplicate = EventDuplicate(
        _event_ack(candidate).receipt,
        BrainBundle(BrainState.fresh(), BrainCState.fresh()),
    )
    instance.fault_stage = cleanup_stage
    with pytest.raises(control_error, match=cleanup_stage):
        coordinator.commit(candidate, lambda _candidate: cast(Any, duplicate))
    _assert_provisional_failed_closed(coordinator, candidate)


def test_reject_invalid_reason_keeps_candidate_retryable_and_primary_live() -> None:
    instance = ScriptedBackend()
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    with pytest.raises(BrainValidationError, match="reason"):
        coordinator.reject(candidate, "")
    assert coordinator.needs_reload is False
    assert coordinator.primary_instance is instance
    assert instance.abort_reasons == []

    coordinator.reject(candidate, "valid retry")
    assert coordinator.needs_reload is True
    assert coordinator.primary_instance is None
    assert instance.abort_reasons == ["valid retry"]


@pytest.mark.parametrize("forgery", ["history_epoch", "status"])
def test_event_ack_rejects_forged_history_and_status(forgery: str) -> None:
    factory = BackendFactory()
    coordinator = _coordinator(factory)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    invalid_ack = copy(_event_ack(candidate))
    receipt = invalid_ack.receipt
    if forgery == "history_epoch":
        receipt = replace(receipt, history_epoch=receipt.history_epoch + 1)
    else:
        receipt = replace(receipt, status="degraded")
    object.__setattr__(invalid_ack, "receipt", receipt)
    with pytest.raises(BrainDurabilityError, match="acknowledgement"):
        coordinator.acknowledge(candidate, invalid_ack)
    assert coordinator.acknowledged_checkpoint is None
    assert coordinator.needs_reload is True


def test_feedback_ack_rejects_forged_status() -> None:
    coordinator = _coordinator()
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    candidate = coordinator.prepare_feedback(
        _feedback(),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=20,
    )
    invalid_ack = copy(_feedback_ack(candidate))
    receipt = replace(invalid_ack.receipt, status="applied")
    object.__setattr__(invalid_ack, "receipt", receipt)
    with pytest.raises(BrainDurabilityError, match="acknowledgement"):
        coordinator.acknowledge(candidate, invalid_ack)
    assert coordinator.needs_reload is True


class FeedbackMutatingBackend(ScriptedBackend):
    def __init__(self, mutate: Callable[[BrainFeedbackResult], object]) -> None:
        super().__init__()
        self._mutate = mutate

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult:
        result = super().apply_feedback(request, timeout_ms=timeout_ms)
        return cast(BrainFeedbackResult, self._mutate(result))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: replace(result, request_id="late"),
        lambda result: replace(result, feedback_id="wrong"),
        lambda result: replace(result, target_tick=99),
        lambda result: replace(result, expected_state_version=99),
        lambda result: replace(result, state_version=99),
        lambda result: replace(result, applied_synapses=-1),
        lambda result: replace(result, applied_synapses=129),
        lambda _result: None,
    ],
)
def test_feedback_reply_fields_versions_and_synapse_count_are_exact(
    mutate: Callable[[BrainFeedbackResult], object],
) -> None:
    instance = FeedbackMutatingBackend(mutate)
    coordinator = _coordinator(lambda: instance)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    feedback = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=True,
        timeout_ms=20,
    )
    assert feedback.source == "builtin"
    assert feedback.degraded is True
    assert instance.abort_reasons
    assert instance.close_calls == 1


def test_allocation_identity_and_primary_alpha_are_strict() -> None:
    with pytest.raises(BrainValidationError, match=r"\[0, 0.1\]"):
        _coordinator(BackendFactory(), alpha=0.1001)
    coordinator = _coordinator()
    with pytest.raises(BrainValidationError, match="allocation tick_id"):
        coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(2, 1),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )


def test_commit_success_result_loss_reloads_new_token_from_real_store(tmp_path: Any) -> None:
    store = BrainStateStore.start(tmp_path)
    factory = BackendFactory(token=b"old-durable")
    coordinator = _coordinator(factory)
    try:
        first_key = event_id_digest("event-1")
        first_allocated = store.preflight_allocate(SESSION_DIGEST, first_key)
        assert isinstance(first_allocated, EventAllocated)
        first = coordinator.prepare_step(
            _step(),
            allocation=first_allocated.allocation,
            route="normal",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
        first_bundle, first_receipt = _stored_event_material(first_allocated, first)
        first_commit = EventCommit(
            first_allocated,
            first_bundle,
            first_receipt,
            first.provisional_checkpoint,
        )
        coordinator.commit(
            first,
            lambda _candidate: store.commit_event(SESSION_DIGEST, first_key, first_commit),
        )
        assert coordinator.acknowledged_checkpoint is not None
        assert coordinator.acknowledged_checkpoint.token == b"old-durable"

        factory.instances[0].token = b"newly-durable"
        second_key = event_id_digest("event-2")
        second_allocated = store.preflight_allocate(SESSION_DIGEST, second_key)
        assert isinstance(second_allocated, EventAllocated)
        second = coordinator.prepare_step(
            _step(2, expected_state_version=1),
            allocation=second_allocated.allocation,
            route="normal",
            created_at=2.0,
            delta_t=1.0,
            timeout_ms=20,
        )
        second_bundle, second_receipt = _stored_event_material(second_allocated, second)
        second_commit = EventCommit(
            second_allocated,
            second_bundle,
            second_receipt,
            second.provisional_checkpoint,
        )

        def commit_then_lose_result(_candidate: CoordinatedStep) -> EventCommitted:
            committed = store.commit_event(SESSION_DIGEST, second_key, second_commit)
            assert isinstance(committed, EventCommitted)
            raise BrainDurabilityError("commit result lost")

        with pytest.raises(BrainDurabilityError, match="result lost"):
            coordinator.commit(second, commit_then_lose_result)

        assert factory.instances[0].abort_reasons == ["store commit was not acknowledged"]
        assert coordinator.acknowledged_checkpoint is not None
        assert coordinator.acknowledged_checkpoint.token == b"old-durable"
        assert coordinator.needs_reload is True
        loaded = store.load(SESSION_DIGEST)
        assert isinstance(loaded, SessionLoaded)
        assert loaded.checkpoint is not None
        assert loaded.checkpoint.token == b"newly-durable"

        coordinator.reload_authoritative(loaded.bundle.c, loaded.checkpoint)
        third_key = event_id_digest("event-3")
        third_allocated = store.preflight_allocate(SESSION_DIGEST, third_key)
        assert isinstance(third_allocated, EventAllocated)
        resumed = coordinator.prepare_step(
            _step(3, expected_state_version=2),
            allocation=third_allocated.allocation,
            route="normal",
            created_at=3.0,
            delta_t=1.0,
            timeout_ms=20,
        )
        assert factory.instances[1].open_calls[0][1] == b"newly-durable"
        coordinator.reject(resumed, "test cleanup")
    finally:
        coordinator.close()
        store.close()


def test_feedback_combined_noop_aborts_invoked_primary() -> None:
    factory = BackendFactory()
    coordinator = _coordinator(factory)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))

    candidate = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(expected_mutation_seq=1, next_mutation_seq=2),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=20,
    )

    assert isinstance(candidate, CoordinatedFeedback)
    assert candidate.combined_changed is False
    assert candidate.provisional_checkpoint is None
    assert factory.instances[0].abort_reasons == ["feedback combined result was a no-op"]
    assert factory.instances[0].close_calls == 1
    assert coordinator.primary_instance is None
    coordinator.acknowledge(candidate, _feedback_ack(candidate))
    assert coordinator.c_state == candidate.c_candidate.state


def test_feedback_token_only_change_consumes_allocation_and_is_acknowledged() -> None:
    factory = BackendFactory(token=b"a")
    coordinator = _coordinator(factory)
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    assert coordinator.acknowledged_checkpoint is not None
    assert coordinator.acknowledged_checkpoint.token == b"a"

    factory.instances[0].token = b"b"
    candidate = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=20,
    )
    assert candidate.c_candidate.status == "no_effect"
    assert candidate.applied_synapses == 0
    assert candidate.combined_changed is True
    assert candidate.provisional_checkpoint is not None
    assert candidate.provisional_checkpoint.token == b"b"
    assert factory.instances[0].abort_reasons == []

    coordinator.acknowledge(candidate, _feedback_ack(candidate))
    assert coordinator.acknowledged_checkpoint is not None
    assert coordinator.acknowledged_checkpoint.token == b"b"
    assert coordinator.acknowledged_checkpoint.acknowledged_mutation_seq == 2


def test_reopen_uses_only_last_acknowledged_checkpoint() -> None:
    factory = BackendFactory("ok", TimeoutError("timeout"), "ok", token=b"new")
    coordinator = _coordinator(factory, checkpoint=_checkpoint(b"old"))
    committed_builtin = coordinator.c_state
    first = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    coordinator.reject(first, "rollback")
    assert coordinator.needs_reload is True
    coordinator.reload_authoritative(committed_builtin, _checkpoint(b"old"))
    second = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert second.source == "builtin"
    coordinator.acknowledge(second, _event_ack(second))
    third = coordinator.prepare_step(
        _step(2),
        allocation=_event_allocation(2, 2),
        route="normal",
        created_at=2.0,
        delta_t=0.0,
        timeout_ms=20,
    )

    assert factory.instances[1].open_calls[0][1] == b"old"
    assert factory.instances[2].open_calls[0][1] == b"old"
    assert third.source == "primary"


def test_checkpoint_mismatch_reinitializes_and_requires_32_accepted_zero_authority_events() -> None:
    factory = BackendFactory("mismatch", "ok")
    coordinator = _coordinator(factory, checkpoint=_checkpoint(b"bad"), alpha=0.08)
    committed_builtin = coordinator.c_state

    rolled_back = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    assert coordinator.warmup_remaining == 32
    coordinator.reject(rolled_back, "rollback does not count as accepted")
    assert coordinator.warmup_remaining == 32
    assert coordinator.needs_reload is True
    coordinator.reload_authoritative(committed_builtin, _checkpoint(b"bad"))

    for tick in range(1, 33):
        candidate = coordinator.prepare_step(
            _step(tick, expected_state_version=tick - 1),
            allocation=_event_allocation(tick, tick),
            route="fast",
            created_at=float(tick),
            delta_t=1.0,
            timeout_ms=20,
        )
        assert candidate.authority_alpha == 0.0
        coordinator.acknowledge(candidate, _event_ack(candidate))

    assert factory.instances[0].abort_reasons == ["checkpoint mismatch"]
    assert factory.instances[1].open_calls[0][1] is None
    assert coordinator.warmup_remaining == 0
    assert coordinator.canary_required is True

    candidate = coordinator.prepare_step(
        _step(33, expected_state_version=32),
        allocation=_event_allocation(33, 33),
        route="fast",
        created_at=33.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    assert candidate.authority_alpha == 0.0
    coordinator.reject(candidate, "canary not passed")


_CHILD_CRASH_EXIT = 77
_CHILD_OOM_EXIT = 88


def _backend_child(connection: Any, fault: str | None) -> None:
    while True:
        operation, payload = connection.recv()
        if operation == "close":
            connection.send((os.getpid(), None))
            return
        if fault == f"block_{operation}":
            while True:
                time.sleep(1.0)
        if operation == "step" and fault == "exit_step":
            os._exit(_CHILD_CRASH_EXIT)
        if operation == "step" and fault == "oom_step":
            os._exit(_CHILD_OOM_EXIT)
        if operation == "open":
            value: object = None
        elif operation == "step":
            request = payload
            value = BrainStepResult(
                request_id=request.request_id,
                event_id=request.event_id,
                tick_id=request.tick_id,
                expected_state_version=request.expected_state_version,
                state_version=request.expected_state_version + 1,
                proposal=(0.25,) * 8,
                eligibility=ZERO128,
            )
        elif operation == "feedback":
            request = payload
            value = BrainFeedbackResult(
                request_id=request.request_id,
                feedback_id=request.feedback_id,
                target_tick=request.target_tick,
                expected_state_version=request.expected_state_version,
                state_version=request.expected_state_version,
                applied_synapses=0,
            )
        elif operation == "checkpoint":
            value = b"ipc-token"
        else:  # pragma: no cover - supervisor sends a closed operation set
            raise AssertionError(f"unknown operation {operation}")
        connection.send((os.getpid(), value))


class BlockingProcessBackend:
    is_process_isolated = True

    def __init__(
        self,
        *,
        fault: str | None = "block_step",
        block_step: bool | None = None,
    ) -> None:
        if block_step is not None:
            fault = "block_step" if block_step else None
        context = multiprocessing.get_context("spawn")
        self._parent, child = context.Pipe()
        self.process = context.Process(target=_backend_child, args=(child, fault))
        self.abort_reasons: list[str] = []
        self.close_calls = 0
        self.rpc_pids: list[int] = []

    def _raise_child_exit(self, operation: str) -> None:
        self.process.join(timeout=1.0)
        if self.process.exitcode == _CHILD_OOM_EXIT:
            raise MemoryError(f"child reported OOM during {operation}")
        raise RuntimeError(f"child exited during {operation} with code {self.process.exitcode}")

    def _rpc(self, operation: str, payload: object, timeout_ms: int) -> object:
        self._parent.send((operation, payload))
        if not self._parent.poll(timeout_ms / 1000.0):
            if not self.process.is_alive():
                self._raise_child_exit(operation)
            raise TimeoutError(f"bounded child IPC timed out during {operation}")
        try:
            pid, value = self._parent.recv()
        except EOFError:
            self._raise_child_exit(operation)
        self.rpc_pids.append(pid)
        return value

    def open(
        self,
        session_digest: bytes,
        checkpoint_token: bytes | None,
        *,
        timeout_ms: int,
    ) -> None:
        self.process.start()
        self._rpc("open", (session_digest, checkpoint_token), timeout_ms)

    def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult:
        return cast(BrainStepResult, self._rpc("step", request, timeout_ms))

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult:
        return cast(BrainFeedbackResult, self._rpc("feedback", request, timeout_ms))

    def checkpoint(self, *, timeout_ms: int) -> bytes:
        return cast(bytes, self._rpc("checkpoint", None, timeout_ms))

    def abort(self, reason: str) -> None:
        self.abort_reasons.append(reason)
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2.0)

    def close(self) -> None:
        self.close_calls += 1
        if self.process.is_alive():
            try:
                self._rpc("close", None, 100)
            except (EOFError, OSError, TimeoutError):
                self.process.terminate()
            self.process.join(timeout=2.0)


def test_real_process_supervisor_timeout_terminates_child_synchronously() -> None:
    instance = BlockingProcessBackend()
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )

    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert instance.abort_reasons
    assert instance.close_calls == 1
    assert not instance.process.is_alive()


def test_real_process_supervisor_uses_child_ipc_for_full_backend_lifecycle() -> None:
    instance = BlockingProcessBackend(block_step=False)
    coordinator = _coordinator(lambda: instance)
    step = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=2000,
    )
    assert step.source == "primary"
    coordinator.acknowledge(step, _event_ack(step))

    feedback = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=2000,
    )
    assert feedback.combined_changed is False
    assert len(instance.rpc_pids) == 5
    assert set(instance.rpc_pids) == {instance.process.pid}
    assert instance.process.pid != os.getpid()
    assert not instance.process.is_alive()


@pytest.mark.parametrize("operation", ["open", "step", "feedback", "checkpoint"])
def test_real_process_supervisor_times_out_and_terminates_each_blocked_operation(
    operation: str,
) -> None:
    instance = BlockingProcessBackend(fault=f"block_{operation}")  # type: ignore[call-arg]
    coordinator = _coordinator(lambda: instance)
    if operation == "feedback":
        event = coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=2000,
        )
        coordinator.acknowledge(event, _event_ack(event))
        candidate = coordinator.prepare_feedback(
            _feedback(expected_state_version=1),
            allocation=_feedback_allocation(),
            state_tick=1,
            trusted_now=1.0,
            state_clock=1.0,
            feedback_ttl_seconds=10.0,
            b_changed=False,
            timeout_ms=20,
        )
    else:
        candidate = coordinator.prepare_step(
            _step(),
            allocation=_event_allocation(),
            route="fast",
            created_at=1.0,
            delta_t=0.0,
            timeout_ms=20,
        )
    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert instance.abort_reasons
    assert instance.close_calls == 1
    assert not instance.process.is_alive()
    assert coordinator.primary_instance is None


@pytest.mark.parametrize(
    ("fault", "classification"),
    [("exit_step", "RuntimeError"), ("oom_step", "MemoryError")],
)
def test_real_process_supervisor_classifies_child_exit_and_oom(
    fault: str,
    classification: str,
) -> None:
    instance = BlockingProcessBackend(fault=fault)  # type: ignore[call-arg]
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=2000,
    )
    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert candidate.failure_reason is not None
    assert classification in candidate.failure_reason
    assert instance.abort_reasons
    assert not instance.process.is_alive()


def test_timed_out_old_process_epoch_cannot_contaminate_reopened_primary() -> None:
    stale = BlockingProcessBackend(fault="block_step")  # type: ignore[call-arg]
    fresh = BlockingProcessBackend(fault=None)  # type: ignore[call-arg]
    instances = iter((stale, fresh))
    coordinator = _coordinator(lambda: next(instances))
    fallback = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )
    assert fallback.source == "builtin"
    coordinator.acknowledge(fallback, _event_ack(fallback))
    assert not stale.process.is_alive()

    reopened = coordinator.prepare_step(
        _step(2, expected_state_version=0),
        allocation=_event_allocation(2, 2),
        route="fast",
        created_at=2.0,
        delta_t=1.0,
        timeout_ms=2000,
    )
    assert reopened.source == "primary"
    assert fresh.rpc_pids
    assert set(fresh.rpc_pids) == {fresh.process.pid}
    assert stale.process.pid != fresh.process.pid
    coordinator.reject(reopened, "test cleanup")


class CountingValues:
    def __init__(self, length: int) -> None:
        self.length = length
        self.consumed = 0

    def __iter__(self) -> Iterator[float]:
        for _ in range(self.length):
            self.consumed += 1
            yield 0.0


def test_oversized_primary_result_consumes_only_one_value_past_expected_length() -> None:
    eligibility = CountingValues(10_000)

    def oversized(result: BrainStepResult) -> BrainStepResult:
        return replace(result, eligibility=cast(Any, eligibility))

    factory = BackendFactory(oversized)
    coordinator = _coordinator(factory)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="normal",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=20,
    )

    assert candidate.source == "builtin"
    assert candidate.degraded is True
    assert eligibility.consumed == 129
    coordinator.close()


def test_fresh_primary_requires_32_durable_event_acks_before_canary_promotion() -> None:
    factory = BackendFactory()
    coordinator = _coordinator(factory, alpha=0.08)

    assert coordinator.warmup_remaining == 32
    with pytest.raises(BrainDurabilityError, match="warmup"):
        coordinator.promote_primary()

    for tick in range(1, 33):
        candidate = coordinator.prepare_step(
            _step(tick, expected_state_version=tick - 1),
            allocation=_event_allocation(tick, tick),
            route="fast",
            created_at=float(tick),
            delta_t=1.0,
            timeout_ms=20,
        )
        assert candidate.authority_alpha == 0.0
        coordinator.acknowledge(candidate, _event_ack(candidate))

    assert coordinator.warmup_remaining == 0
    assert coordinator.canary_required is True
    coordinator.promote_primary()
    promoted = coordinator.prepare_step(
        _step(33, expected_state_version=32),
        allocation=_event_allocation(33, 33),
        route="fast",
        created_at=33.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    assert promoted.authority_alpha == 0.08
    coordinator.reject(promoted, "test cleanup")


def test_config_promoted_clears_canary_and_survives_reconstruction() -> None:
    # Step 0b: config.c_promoted is the DURABLE operator sign-off. It clears the
    # canary at construction WITHOUT a promote_primary() call (which has no
    # production caller), and because the engine threads it from config into every
    # _new_coordinator on rebase, its effect survives reconstruction — unlike
    # promote_primary(), which a fresh coordinator loses. Warmup still gates per
    # session: even a promoted backend earns alpha only after 32 committed events.
    factory = BackendFactory()
    coordinator = _coordinator(factory, alpha=0.08, promoted=True)

    assert coordinator.canary_required is False  # promotion survives (re)construction
    assert coordinator.warmup_remaining == 32

    for tick in range(1, 33):
        candidate = coordinator.prepare_step(
            _step(tick, expected_state_version=tick - 1),
            allocation=_event_allocation(tick, tick),
            route="fast",
            created_at=float(tick),
            delta_t=1.0,
            timeout_ms=20,
        )
        assert candidate.authority_alpha == 0.0  # warmup still gates during ramp-up
        coordinator.acknowledge(candidate, _event_ack(candidate))

    assert coordinator.warmup_remaining == 0
    earned = coordinator.prepare_step(
        _step(33, expected_state_version=32),
        allocation=_event_allocation(33, 33),
        route="fast",
        created_at=33.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    # No promote_primary() call needed anywhere: authority came from config.
    assert earned.authority_alpha == 0.08
    coordinator.reject(earned, "test cleanup")


def test_unpromoted_primary_is_gated_to_zero_authority_forever() -> None:
    # The safe default and the exact reason the un-silenced Brain C seam
    # (config.c_authority) is WIRED but not LIVE: with c_promoted=False and no
    # production caller of promote_primary(), the canary never clears, so even a
    # fully warmed-up primary stays at authority 0. Opt-in, off until an operator
    # promotes after the offline value-gate instrument passes.
    factory = BackendFactory()
    coordinator = _coordinator(factory, alpha=0.08, promoted=False)
    assert coordinator.canary_required is True

    for tick in range(1, 33):
        candidate = coordinator.prepare_step(
            _step(tick, expected_state_version=tick - 1),
            allocation=_event_allocation(tick, tick),
            route="fast",
            created_at=float(tick),
            delta_t=1.0,
            timeout_ms=20,
        )
        coordinator.acknowledge(candidate, _event_ack(candidate))

    assert coordinator.warmup_remaining == 0
    assert coordinator.canary_required is True  # never cleared: no promoter exists
    gated = coordinator.prepare_step(
        _step(33, expected_state_version=32),
        allocation=_event_allocation(33, 33),
        route="fast",
        created_at=33.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    assert gated.authority_alpha == 0.0
    coordinator.reject(gated, "test cleanup")


def test_fresh_primary_rollback_does_not_consume_warmup() -> None:
    coordinator = _coordinator(BackendFactory())
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=1.0,
        timeout_ms=20,
    )

    coordinator.reject(candidate, "rollback")
    assert coordinator.warmup_remaining == 32


def test_fresh_primary_duplicate_ack_does_not_consume_warmup() -> None:
    coordinator = _coordinator(BackendFactory())
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    acknowledgement = _event_ack(candidate)
    duplicate = EventDuplicate(
        acknowledgement.receipt,
        BrainBundle(BrainState.fresh(), BrainCState.fresh()),
    )

    with pytest.raises(BrainDurabilityError, match="positive EventCommitted"):
        coordinator.acknowledge(candidate, cast(Any, duplicate))
    assert coordinator.warmup_remaining == 32


def test_fresh_primary_builtin_fallback_does_not_consume_warmup() -> None:
    coordinator = _coordinator(BackendFactory(RuntimeError("primary failed")))
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=1.0,
        timeout_ms=20,
    )

    assert candidate.source == "builtin"
    coordinator.acknowledge(candidate, _event_ack(candidate))
    assert coordinator.warmup_remaining == 32


def test_fresh_primary_feedback_ack_does_not_consume_event_warmup() -> None:
    coordinator = _coordinator(BackendFactory())
    event = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=1.0,
        timeout_ms=20,
    )
    coordinator.acknowledge(event, _event_ack(event))
    assert coordinator.warmup_remaining == 31

    feedback = coordinator.prepare_feedback(
        _feedback(expected_state_version=1),
        allocation=_feedback_allocation(),
        state_tick=1,
        trusted_now=1.0,
        state_clock=1.0,
        feedback_ttl_seconds=10.0,
        b_changed=False,
        timeout_ms=20,
    )
    coordinator.acknowledge(feedback, _feedback_ack(feedback))
    assert coordinator.warmup_remaining == 31


@pytest.mark.parametrize(
    "failure",
    [MemoryError("post-ack-copy"), GeneratorExit("post-ack-copy")],
)
def test_post_ack_builtin_copy_baseexception_fails_closed_before_propagating(
    tmp_path: Any,
    failure: BaseException,
) -> None:
    instance = ScriptedBackend()
    coordinator = _coordinator(lambda: instance)
    candidate, acknowledgement = _prepare_and_commit_candidate_in_real_store(
        tmp_path,
        coordinator,
    )

    def fail_copy(_state: BrainCState) -> BrainCState:
        raise failure

    try:
        with (
            patch.object(BrainCState, "copy", fail_copy),
            pytest.raises(type(failure), match="post-ack-copy"),
        ):
            coordinator.acknowledge(candidate, acknowledgement)
        needs_reload = coordinator.needs_reload
        primary_after_failure = coordinator.primary_instance
        aborts_after_failure = tuple(instance.abort_reasons)
        try:
            coordinator.acknowledge(candidate, acknowledgement)
        except BrainDurabilityError as error:
            stale_after_failure = "stale" in str(error)
        else:
            stale_after_failure = False
    finally:
        if instance is coordinator.primary_instance:
            instance.abort("red-test cleanup")
            instance.close()

    assert needs_reload is True
    assert primary_after_failure is None
    assert aborts_after_failure
    assert stale_after_failure is True


class ReceiptIterationGeneratorExit:
    def __iter__(self) -> Iterator[int]:
        raise GeneratorExit("receipt-iteration")


@pytest.mark.parametrize("entrypoint", ["acknowledge", "commit"])
def test_ack_seal_generator_exit_fails_closed_before_propagating(
    tmp_path: Any,
    entrypoint: str,
) -> None:
    instance = ScriptedBackend()
    coordinator = _coordinator(lambda: instance)
    candidate, acknowledgement = _prepare_and_commit_candidate_in_real_store(
        tmp_path / entrypoint,
        coordinator,
    )
    invalid_receipt = copy(acknowledgement.receipt)
    object.__setattr__(
        invalid_receipt,
        "applied_dimensions",
        cast(Any, ReceiptIterationGeneratorExit()),
    )
    invalid_acknowledgement = copy(acknowledgement)
    object.__setattr__(invalid_acknowledgement, "receipt", invalid_receipt)

    try:
        with pytest.raises(GeneratorExit, match="receipt-iteration"):
            if entrypoint == "acknowledge":
                coordinator.acknowledge(candidate, invalid_acknowledgement)
            else:
                coordinator.commit(candidate, lambda _candidate: invalid_acknowledgement)
        needs_reload = coordinator.needs_reload
        primary_after_failure = coordinator.primary_instance
        aborts_after_failure = tuple(instance.abort_reasons)
        try:
            coordinator.acknowledge(candidate, acknowledgement)
        except BrainDurabilityError as error:
            stale_after_failure = "stale" in str(error)
        else:
            stale_after_failure = False
    finally:
        if instance is coordinator.primary_instance:
            instance.abort("red-test cleanup")
            instance.close()

    assert needs_reload is True
    assert primary_after_failure is None
    assert aborts_after_failure
    assert stale_after_failure is True


class CloseFaultProcessBackend(BlockingProcessBackend):
    def __init__(self, failure: BaseException) -> None:
        super().__init__(fault=None)
        self._close_failure = failure
        self._close_failed = False

    def close(self) -> None:
        if not self._close_failed:
            self._close_failed = True
            self.close_calls += 1
            raise self._close_failure
        super().close()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("close-first"), GeneratorExit("close-first")],
)
def test_direct_close_failure_falls_back_to_abort_and_synchronous_child_cleanup(
    failure: BaseException,
) -> None:
    instance = CloseFaultProcessBackend(failure)
    coordinator = _coordinator(lambda: instance)
    candidate = coordinator.prepare_step(
        _step(),
        allocation=_event_allocation(),
        route="fast",
        created_at=1.0,
        delta_t=0.0,
        timeout_ms=2000,
    )
    coordinator.acknowledge(candidate, _event_ack(candidate))
    assert instance.process.is_alive()

    try:
        if isinstance(failure, Exception):
            coordinator.close()
        else:
            with pytest.raises(type(failure), match="close-first"):
                coordinator.close()
        child_alive_after_close = instance.process.is_alive()
        primary_after_close = coordinator.primary_instance
        aborts_after_close = tuple(instance.abort_reasons)
    finally:
        if instance.process.is_alive():
            instance.abort("red-test cleanup")
            instance.close()

    assert child_alive_after_close is False
    assert primary_after_close is None
    assert aborts_after_close
