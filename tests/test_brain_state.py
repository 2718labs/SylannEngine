"""Authoritative B-state mathematics, allocation, and irreversibility contracts."""

from __future__ import annotations

import math
import sys
from array import array
from collections import deque
from dataclasses import FrozenInstanceError

import pytest

from sylanne_core.compute.brain_compute import (
    AXES,
    D_MAX,
    D_SCALE,
    ETA_B,
    ETA_THETA,
    BrainComputeCore,
    BrainEvent,
    EventCandidate,
    FeedbackCandidate,
    R,
    _feedback_candidate,
    compose_appraisal,
    evolve_b,
    evolve_feedback,
    project_oriented,
    sanitize_surprise,
    scar_view,
)
from sylanne_core.compute.brain_errors import (
    BrainAllocationError,
    BrainComputeError,
    BrainCounterExhaustedError,
    BrainDurabilityError,
    BrainNotificationBackpressureError,
    BrainOwnershipError,
    BrainValidationError,
)
from sylanne_core.compute.brain_state import (
    MAX_COUNTER,
    BEligibilityRecord,
    BrainState,
    EventAllocation,
    FeedbackAllocation,
)

ZERO8 = (0.0,) * 8
LINEAGE = "00000000-0000-0000-0000-000000000001"
OTHER_LINEAGE = "00000000-0000-0000-0000-000000000002"


def test_strict_brain_error_hierarchy_uses_plan_pinned_names() -> None:
    assert issubclass(BrainValidationError, BrainComputeError)
    assert issubclass(BrainDurabilityError, BrainComputeError)
    assert issubclass(BrainAllocationError, BrainDurabilityError)
    assert issubclass(BrainCounterExhaustedError, BrainDurabilityError)
    assert issubclass(BrainOwnershipError, BrainComputeError)
    assert issubclass(BrainNotificationBackpressureError, BrainComputeError)


def neutral_event(event_id: str = "neutral") -> BrainEvent:
    return BrainEvent(
        event_id=event_id,
        assessment=ZERO8,
        hdc=ZERO8,
        wound_sum=ZERO8,
        surprise=0.0,
        perception_acuity=0.5,
        proposal_c=ZERO8,
    )


def next_event_allocation(state: BrainState) -> EventAllocation:
    return EventAllocation(
        generation=state.generation,
        lineage_id=state.lineage_id,
        tick_id=state.tick_id + 1,
        history_epoch=state.history_epoch + 1,
        mutation_seq=state.mutation_seq + 1,
    )


def feedback_allocation(state: BrainState, target_tick: int) -> FeedbackAllocation:
    return FeedbackAllocation(
        generation=state.generation,
        lineage_id=state.lineage_id,
        target_tick=target_tick,
        expected_mutation_seq=state.mutation_seq,
        next_mutation_seq=state.mutation_seq + 1,
    )


def custom_state(
    *,
    e: tuple[float, ...] = ZERO8,
    d_plus: tuple[float, ...] = ZERO8,
    d_minus: tuple[float, ...] = ZERO8,
    gain_b: tuple[float, ...] = (0.5,) * 8,
    theta_b: tuple[float, ...] = (0.05,) * 8,
    clock: float = 0.0,
    tick_id: int = 0,
    history_epoch: int = 0,
    mutation_seq: int = 0,
    records: tuple[BEligibilityRecord, ...] = (),
    horizon: int = 8,
    clock_regressions: int = 0,
) -> BrainState:
    return BrainState(
        generation=3,
        lineage_id=LINEAGE,
        e=e,
        d_plus=d_plus,
        d_minus=d_minus,
        gain_b=gain_b,
        theta_b=theta_b,
        clock=clock,
        tick_id=tick_id,
        history_epoch=history_epoch,
        mutation_seq=mutation_seq,
        eligibility_ring=records,
        eligibility_horizon=horizon,
        clock_regressions=clock_regressions,
    )


def matrix_rank(rows: tuple[tuple[float, ...], ...], tolerance: float = 1e-12) -> int:
    work = [list(row) for row in rows]
    rank = 0
    for column in range(len(work[0])):
        pivot = next(
            (row for row in range(rank, len(work)) if abs(work[row][column]) > tolerance),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(len(work)):
            if row == rank:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
    return rank


def test_exact_axis_order_matrix_rank_and_basis_vectors() -> None:
    expected_r = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    assert AXES == (
        "warmth",
        "arousal",
        "valence",
        "tension",
        "curiosity",
        "repair_pressure",
        "expression_drive",
        "boundary_firmness",
    )
    assert expected_r == R
    assert matrix_rank(R) == 7

    for column in range(8):
        basis = tuple(1.0 if index == column else 0.0 for index in range(8))
        expected = tuple(row[column] for row in expected_r)
        assert tuple(project_oriented(basis)) == expected


def test_source_composition_sanitizes_then_clips_with_fixed_coefficients() -> None:
    actual = compose_appraisal(
        assessment=(0.2, math.nan, 0.4, math.inf, -math.inf, 0.5, 2.0, -2.0),
        hdc=(0.4, 0.8, -0.4, 0.2, 0.6, -0.8, 0.4, -0.4),
        wound_sum=(-0.2, 0.6, -0.8, 0.4, -0.2, 1.0, -0.6, 0.8),
    )
    assert tuple(actual) == pytest.approx((0.2, 0.5, -0.1, 0.25, 0.05, 0.8, 0.8, -0.7), abs=1e-15)
    assert tuple(compose_appraisal(None, ZERO8, ZERO8)) == ZERO8


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_surprise_becomes_zero_before_clipping(value: float) -> None:
    assert sanitize_surprise(value) == 0.0


@pytest.mark.parametrize(("value", "expected"), [(-1.0, 0.0), (0.25, 0.25), (2.0, 1.0)])
def test_finite_surprise_is_clipped(value: float, expected: float) -> None:
    assert sanitize_surprise(value) == expected


def test_brain_event_rejects_malformed_or_nonfinite_backend_proposal() -> None:
    kwargs = {
        "event_id": "event",
        "assessment": ZERO8,
        "hdc": ZERO8,
        "wound_sum": ZERO8,
        "surprise": 0.0,
        "perception_acuity": 0.5,
    }
    with pytest.raises(BrainValidationError, match="proposal_c.*eight"):
        BrainEvent(**kwargs, proposal_c=(0.0,) * 7)
    with pytest.raises(BrainValidationError, match="proposal_c.*finite"):
        BrainEvent(**kwargs, proposal_c=(0.0,) * 7 + (math.nan,))
    with pytest.raises(BrainValidationError, match="hdc.*eight"):
        BrainEvent(**{**kwargs, "hdc": (0.0,) * 7}, proposal_c=ZERO8)


def test_brain_event_is_immutable_and_defensively_copies_inputs() -> None:
    source = [0.0] * 8
    event = BrainEvent(
        event_id="event",
        assessment=source,
        hdc=source,
        wound_sum=source,
        surprise=0.0,
        perception_acuity=0.5,
        proposal_c=source,
    )
    source[0] = 1.0
    assert tuple(event.assessment or ()) == ZERO8
    assert tuple(event.hdc) == ZERO8
    assert tuple(event.wound_sum) == ZERO8
    assert tuple(event.proposal_c) == ZERO8
    with pytest.raises(FrozenInstanceError):
        event.event_id = "changed"  # type: ignore[misc]


def test_brain_event_normalizes_untrusted_appraisal_and_surprise_at_boundary() -> None:
    event = BrainEvent(
        event_id="sanitize",
        assessment=(math.nan, math.inf, -math.inf, 2.0, -2.0, 0.0, 0.5, -0.5),
        hdc=(math.nan,) * 8,
        wound_sum=(math.inf,) * 8,
        surprise=math.inf,
        perception_acuity=math.nan,
        proposal_c=ZERO8,
    )
    assert event.assessment == (0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.5, -0.5)
    assert event.hdc == ZERO8
    assert event.wound_sum == ZERO8
    assert event.surprise == 0.0


def test_complete_published_b_step_golden() -> None:
    state = custom_state(
        e=(0.2, -0.1, 0.3, -0.4, 0.05, -0.2, 0.4, -0.3),
        d_plus=(1.0, 2.0, 0.0, 4.0, 0.5, 0.0, 8.0, 1.5),
        d_minus=(0.5, 1.0, 3.0, 0.0, 0.5, 2.0, 1.0, 5.0),
        clock=1000.0,
        tick_id=41,
        history_epoch=99,
        mutation_seq=120,
    )
    core = BrainComputeCore(state)
    event = BrainEvent(
        event_id="golden",
        assessment=(0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4),
        hdc=(0.4, 0.8, -0.4, 0.2, 0.6, -0.8, 0.4, -0.4),
        wound_sum=(-0.2, 0.6, -0.8, 0.4, -0.2, 1.0, -0.6, 0.8),
        surprise=0.6,
        perception_acuity=0.8,
        proposal_c=(0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75),
    )
    allocation = EventAllocation(3, LINEAGE, 42, 100, 121)

    candidate = core.prepare_event(
        event,
        allocation=allocation,
        trusted_now=1120.0,
        alpha_c=0.08,
    )

    assert tuple(candidate.appraisal) == pytest.approx(
        (0.2, 0.4, -0.1, 0.55, -0.15, 0.8, -0.1, -0.1), rel=1e-12, abs=1e-12
    )
    assert tuple(candidate.oriented) == pytest.approx(
        (0.2, -0.325, -0.1, -0.55, -0.15, -0.8, -0.1, 0.1),
        rel=1e-12,
        abs=1e-12,
    )
    assert candidate.salience == pytest.approx(0.6, rel=1e-12, abs=1e-12)
    assert tuple(candidate.state.d_plus) == pytest.approx(
        (1.0870650074418082, 2.0, 0.0, 4.0, 0.5, 0.0, 8.0, 1.5285803508672435),
        rel=1e-12,
        abs=1e-12,
    )
    assert tuple(candidate.state.d_minus) == pytest.approx(
        (
            0.5,
            1.1594323602119467,
            3.0271747598409857,
            0.2985981342508491,
            0.5590071634969677,
            2.4189225472316864,
            1.0290488812093295,
            5.0,
        ),
        rel=1e-12,
        abs=1e-12,
    )
    assert tuple(candidate.state.e) == pytest.approx(
        (
            0.3394971724056208,
            0.06218447733587657,
            0.3211654077022184,
            -0.20892943833662075,
            -0.006513613002966236,
            0.1906838119371053,
            0.42063459750172827,
            -0.4055805834708768,
        ),
        rel=1e-12,
        abs=1e-12,
    )
    # b_trace is computed from the residual-free visible affect (decontaminated: the
    # C residual driven by proposal_c must never leak into the eligibility ring), so
    # these values differ from a naive abs(e_new - old_e) that includes the residual.
    assert tuple(candidate.b_trace) == pytest.approx(
        (
            0.10230572103277588,
            0.2073274566975523,
            0.05977165298195351,
            0.2804017533587183,
            0.07838685628858427,
            0.4246478968448282,
            0.04030246318244367,
            0.04651764415504871,
        ),
        rel=1e-12,
        abs=1e-12,
    )
    assert (
        candidate.state.tick_id,
        candidate.state.history_epoch,
        candidate.state.mutation_seq,
    ) == (
        42,
        100,
        121,
    )
    assert candidate.state.clock == 1120.0
    assert candidate.state.eligibility_records[-1].tick_id == 42
    assert candidate.state.eligibility_records[-1].created_at == 1120.0
    leaked_candidate_trace = candidate._b_trace
    leaked_candidate_trace[0] = 1.0
    assert candidate.b_trace[0] == pytest.approx(0.10230572103277588, abs=1e-12)


def test_stable_expm1_scar_view_preserves_tiny_positive_dose() -> None:
    tiny = 1e-18
    view = scar_view((tiny,) + (0.0,) * 7)
    expected = -math.expm1(-tiny / D_SCALE)
    assert view[0] == expected
    assert view[0] > 0.0
    assert 1.0 - math.exp(-tiny / D_SCALE) == 0.0


def test_every_accepted_event_advances_time_arrow_without_committing_early() -> None:
    core = BrainComputeCore.fresh(lineage_id=LINEAGE)
    before = core.state
    candidate = core.prepare_event(
        neutral_event("e1"), allocation=next_event_allocation(before), trusted_now=0.0
    )

    assert core.state == before
    assert candidate.state.history_epoch == before.history_epoch + 1
    assert candidate.state.tick_id == before.tick_id + 1
    assert candidate.state.mutation_seq == before.mutation_seq + 1
    assert candidate.state.d_plus == before.d_plus
    assert candidate.state.d_minus == before.d_minus

    committed = core.commit(candidate)
    assert committed == candidate.state
    assert core.state == candidate.state


def test_b_arrays_traces_and_ring_have_fixed_storage_without_mutable_aliases() -> None:
    trace_input = array("d", (0.25,) * 8)
    record = BEligibilityRecord(1, 10.0, trace_input)
    state = custom_state(
        records=(record,),
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        horizon=3,
        clock=10.0,
    )

    assert isinstance(state.e, array) and state.e.typecode == "d"
    assert isinstance(state.d_plus, array) and state.d_plus.typecode == "d"
    assert isinstance(state.d_minus, array) and state.d_minus.typecode == "d"
    assert isinstance(state.gain_b, array) and state.gain_b.typecode == "d"
    assert isinstance(state.theta_b, array) and state.theta_b.typecode == "d"
    assert isinstance(record.b_trace, array) and record.b_trace.typecode == "d"
    assert isinstance(state._eligibility_ring, deque)
    assert state._eligibility_ring.maxlen == 3

    trace_input[0] = 0.9
    leaked_trace = record.b_trace
    leaked_trace[1] = 0.8
    leaked_e = state.e
    leaked_e[0] = 1.0
    leaked_private_e = state._e
    leaked_private_e[1] = 1.0
    leaked_private_trace = record._b_trace
    leaked_private_trace[2] = 1.0
    leaked_ring = state._eligibility_ring
    leaked_ring.clear()
    exposed_record = state.eligibility_records[0]
    exposed_record.b_trace[2] = 0.7
    assert tuple(record.b_trace) == (0.25,) * 8
    assert tuple(state.e) == ZERO8
    assert tuple(state.eligibility_records[0].b_trace) == (0.25,) * 8


def test_standard_attribute_access_cannot_reach_any_mutable_storage() -> None:
    record = BEligibilityRecord(1, 0.0, (0.25,) * 8)
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    candidate = BrainComputeCore(state).prepare_event(
        neutral_event(), allocation=next_event_allocation(state)
    )

    hidden_state_storage = (
        "_BrainState__e",
        "_BrainState__d_plus",
        "_BrainState__d_minus",
        "_BrainState__gain_b",
        "_BrainState__theta_b",
        "_BrainState__eligibility_ring",
    )
    hidden_record_storage = ("_BEligibilityRecord__b_trace",)
    hidden_candidate_storage = (
        "_EventCandidate__appraisal",
        "_EventCandidate__oriented",
        "_EventCandidate__rho_plus",
        "_EventCandidate__rho_minus",
        "_EventCandidate__b_trace",
    )
    for owner, names in (
        (state, hidden_state_storage),
        (record, hidden_record_storage),
        (candidate, hidden_candidate_storage),
        (candidate.state, hidden_state_storage),
    ):
        for name in names:
            with pytest.raises(AttributeError, match=name):
                getattr(owner, name)

    before_dose = tuple(state.d_plus)
    committed = BrainComputeCore(state)
    prepared = committed.prepare_event(
        neutral_event("safe-commit"), allocation=next_event_allocation(state)
    )
    with pytest.raises(AttributeError):
        _ = prepared.state._BrainState__d_plus  # type: ignore[attr-defined]
    committed.commit(prepared)
    assert all(
        after >= before for after, before in zip(committed.state.d_plus, before_dose, strict=True)
    )


def test_hidden_eligibility_storage_does_not_break_safe_record_repr() -> None:
    record = BEligibilityRecord(1, 10.0, (0.25,) * 8)
    assert repr(record) == "BEligibilityRecord(tick_id=1, created_at=10.0, b_trace=<float64[8]>)"


@pytest.mark.parametrize("field", ["generation", "tick_id", "history_epoch", "mutation_seq"])
@pytest.mark.parametrize("bad", [-1, MAX_COUNTER + 1, True, 1.5])
def test_event_allocation_rejects_invalid_counter_domains(field: str, bad: object) -> None:
    values: dict[str, object] = {
        "generation": 3,
        "lineage_id": LINEAGE,
        "tick_id": 1,
        "history_epoch": 1,
        "mutation_seq": 1,
    }
    values[field] = bad
    with pytest.raises(BrainValidationError, match=field):
        EventAllocation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"generation": 4},
        {"lineage_id": OTHER_LINEAGE},
        {"tick_id": 2},
        {"history_epoch": 2},
        {"mutation_seq": 2},
    ],
)
def test_event_allocation_must_match_state_and_exact_successors(
    override: dict[str, object],
) -> None:
    state = custom_state()
    values: dict[str, object] = {
        "generation": state.generation,
        "lineage_id": state.lineage_id,
        "tick_id": 1,
        "history_epoch": 1,
        "mutation_seq": 1,
    }
    values.update(override)
    allocation = EventAllocation(**values)  # type: ignore[arg-type]
    with pytest.raises(BrainAllocationError):
        BrainComputeCore(state).prepare_event(neutral_event(), allocation=allocation)


@pytest.mark.parametrize("field", ["tick_id", "history_epoch", "mutation_seq"])
def test_event_counter_exhaustion_is_strict(field: str) -> None:
    kwargs = {"tick_id": 0, "history_epoch": 0, "mutation_seq": 0}
    kwargs[field] = MAX_COUNTER
    state = custom_state(**kwargs)
    allocation = EventAllocation(
        state.generation,
        state.lineage_id,
        min(state.tick_id + 1, MAX_COUNTER),
        min(state.history_epoch + 1, MAX_COUNTER),
        min(state.mutation_seq + 1, MAX_COUNTER),
    )
    with pytest.raises(BrainCounterExhaustedError, match=field):
        BrainComputeCore(state).prepare_event(neutral_event(), allocation=allocation)


def test_visible_affect_can_recover_while_dose_and_epoch_do_not() -> None:
    state = custom_state(
        e=(0.8,) + ZERO8[1:],
        d_plus=(4.0,) + ZERO8[1:],
        d_minus=(4.0,) + ZERO8[1:],
        clock=0.0,
    )
    candidate = BrainComputeCore(state).prepare_event(
        neutral_event(), allocation=next_event_allocation(state), trusted_now=300.0
    )
    assert abs(candidate.state.e[0]) < abs(state.e[0])
    assert candidate.state.d_plus == state.d_plus
    assert candidate.state.d_minus == state.d_minus
    assert candidate.state.history_epoch > state.history_epoch


def test_rank_projection_has_a_noninjective_witness_but_epoch_is_the_no_return_proof() -> None:
    first = (0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    second = (0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert first != second
    assert tuple(project_oriented(first)) == tuple(project_oriented(second))

    core = BrainComputeCore.fresh(lineage_id=LINEAGE)
    candidate = core.prepare_event(neutral_event(), allocation=next_event_allocation(core.state))
    assert candidate.state.e == core.state.e
    assert candidate.state.d_plus == core.state.d_plus
    assert candidate.state.d_minus == core.state.d_minus
    assert candidate.state != core.state
    assert candidate.state.history_epoch == core.state.history_epoch + 1


def test_b_trace_is_exact_delta_plus_both_dose_fractions() -> None:
    state = custom_state(e=(0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8))
    event = BrainEvent(
        event_id="trace",
        assessment=(0.3, 0.6, -0.4, 0.5, -0.2, 0.1, 0.8, -0.7),
        hdc=ZERO8,
        wound_sum=ZERO8,
        surprise=0.75,
        perception_acuity=0.5,
        proposal_c=ZERO8,
    )
    candidate = BrainComputeCore(state).prepare_event(
        event, allocation=next_event_allocation(state), trusted_now=20.0
    )
    expected = tuple(
        min(
            1.0,
            abs(candidate.state.e[index] - state.e[index])
            + candidate.rho_plus[index]
            + candidate.rho_minus[index],
        )
        for index in range(8)
    )
    assert tuple(candidate.b_trace) == pytest.approx(expected, rel=1e-15, abs=1e-15)
    assert tuple(candidate.state.eligibility_records[-1].b_trace) == pytest.approx(
        expected, rel=1e-15, abs=1e-15
    )


@pytest.mark.parametrize("trusted_now", [999.0, math.nan, math.inf, -math.inf])
def test_event_clock_regression_uses_old_clock_zero_dt_and_diagnostic(
    trusted_now: float,
) -> None:
    state = custom_state(e=(0.4,) + ZERO8[1:], clock=1000.0)
    candidate = BrainComputeCore(state).prepare_event(
        neutral_event(), allocation=next_event_allocation(state), trusted_now=trusted_now
    )
    baseline = BrainComputeCore(state).prepare_event(
        neutral_event(), allocation=next_event_allocation(state), trusted_now=1000.0
    )
    assert candidate.state.clock == state.clock
    assert candidate.state.clock_regressions == state.clock_regressions + 1
    assert candidate.state.e == baseline.state.e


@pytest.mark.parametrize("alpha_c", [-0.1, 0.1000000001, math.nan, math.inf])
def test_alpha_c_is_finite_and_in_closed_calibration_range(alpha_c: float) -> None:
    state = custom_state()
    with pytest.raises(BrainValidationError, match="alpha_c"):
        BrainComputeCore(state).prepare_event(
            neutral_event(), allocation=next_event_allocation(state), alpha_c=alpha_c
        )


def test_c_residual_is_component_clipped_without_overflow() -> None:
    huge = sys.float_info.max
    state = custom_state()
    event = BrainEvent(
        event_id="huge-proposal",
        assessment=ZERO8,
        hdc=ZERO8,
        wound_sum=ZERO8,
        surprise=0.0,
        perception_acuity=0.5,
        proposal_c=(huge, -huge) + ZERO8[2:],
    )
    candidate = BrainComputeCore(state).prepare_event(
        event, allocation=next_event_allocation(state), alpha_c=0.1
    )
    assert candidate.state.e[0] == 0.1
    assert candidate.state.e[1] == -0.1
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in candidate.state.e)


def test_c_residual_cap_is_wired_and_component_clipped_without_overflow() -> None:
    huge = sys.float_info.max
    state = custom_state()
    event = BrainEvent(
        event_id="huge-proposal-narrow-cap",
        assessment=ZERO8,
        hdc=ZERO8,
        wound_sum=ZERO8,
        surprise=0.0,
        perception_acuity=0.5,
        proposal_c=(huge, -huge) + ZERO8[2:],
    )
    candidate = BrainComputeCore(state).prepare_event(
        event,
        allocation=next_event_allocation(state),
        alpha_c=0.1,
        c_residual_cap=0.03,
    )
    assert candidate.state.e[0] == 0.03
    assert candidate.state.e[1] == -0.03
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in candidate.state.e)


@pytest.mark.parametrize("c_residual_cap", [-0.1, 0.1000000001, math.nan, math.inf])
def test_c_residual_cap_is_finite_and_in_closed_calibration_range(
    c_residual_cap: float,
) -> None:
    state = custom_state()
    with pytest.raises(BrainValidationError, match="c_residual_cap"):
        BrainComputeCore(state).prepare_event(
            neutral_event(),
            allocation=next_event_allocation(state),
            c_residual_cap=c_residual_cap,
        )


def test_dose_and_b_trace_are_proposal_c_invariant() -> None:
    """dose (d_plus/d_minus) and b_trace must be functions of appraisal/oriented/
    salience/old_theta ONLY -- proposal_c (the C residual) must appear in neither.

    dose feeds the future scar dose RATE (via theta_b, brain_compute.py ~802-806
    and the dose block ~596-606) and b_trace feeds the durable eligibility ring that
    later gates feedback -> gain_b/theta_b. If proposal_c ever reappears in either
    quantity, C gains a path to modulate scarring that it must never have. This test
    fails if that coupling is reintroduced, regardless of whether the code comments
    still claim it isn't there.
    """
    state = custom_state(e=(0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8))
    shared_kwargs = {
        "assessment": (0.3, 0.6, -0.4, 0.5, -0.2, 0.1, 0.8, -0.7),
        "hdc": ZERO8,
        "wound_sum": ZERO8,
        "surprise": 0.75,
        "perception_acuity": 0.5,
    }
    quiet = BrainEvent(event_id="quiet", proposal_c=ZERO8, **shared_kwargs)
    loud = BrainEvent(
        event_id="loud",
        proposal_c=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        **shared_kwargs,
    )

    quiet_candidate = BrainComputeCore(state).prepare_event(
        quiet, allocation=next_event_allocation(state), trusted_now=20.0, alpha_c=0.1
    )
    loud_candidate = BrainComputeCore(state).prepare_event(
        loud, allocation=next_event_allocation(state), trusted_now=20.0, alpha_c=0.1
    )

    # The residual actually moved the visible affect -- otherwise this test would be
    # vacuously true no matter what b_trace/dose are computed from.
    assert loud_candidate.state.e != quiet_candidate.state.e

    # ... but dose and b_trace (and therefore theta_b, which feedback derives from
    # b_trace) must be bit-identical regardless of proposal_c.
    assert loud_candidate.state.d_plus == quiet_candidate.state.d_plus
    assert loud_candidate.state.d_minus == quiet_candidate.state.d_minus
    assert tuple(loud_candidate.b_trace) == tuple(quiet_candidate.b_trace)
    assert tuple(loud_candidate.state.eligibility_records[-1].b_trace) == tuple(
        quiet_candidate.state.eligibility_records[-1].b_trace
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("generation", True),
        ("target_tick", -1),
        ("expected_mutation_seq", MAX_COUNTER + 1),
        ("next_mutation_seq", 1.5),
    ],
)
def test_feedback_allocation_rejects_invalid_counter_domains(field: str, bad: object) -> None:
    values: dict[str, object] = {
        "generation": 3,
        "lineage_id": LINEAGE,
        "target_tick": 1,
        "expected_mutation_seq": 1,
        "next_mutation_seq": 2,
    }
    values[field] = bad
    with pytest.raises(BrainValidationError, match=field):
        FeedbackAllocation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"generation": 4},
        {"lineage_id": OTHER_LINEAGE},
        {"target_tick": 2},
        {"expected_mutation_seq": 0},
        {"next_mutation_seq": 3},
    ],
)
def test_feedback_allocation_must_match_state_target_and_exact_successor(
    override: dict[str, object],
) -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(tick_id=2, history_epoch=2, mutation_seq=1, records=(record,), clock=0.0)
    values: dict[str, object] = {
        "generation": 3,
        "lineage_id": LINEAGE,
        "target_tick": 1,
        "expected_mutation_seq": 1,
        "next_mutation_seq": 2,
    }
    values.update(override)
    allocation = FeedbackAllocation(**values)  # type: ignore[arg-type]
    with pytest.raises(BrainAllocationError):
        BrainComputeCore(state).prepare_feedback(
            target_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=allocation,
        )


def test_feedback_counter_exhaustion_is_strict() -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=MAX_COUNTER,
        records=(record,),
    )
    allocation = FeedbackAllocation(3, LINEAGE, 1, MAX_COUNTER, MAX_COUNTER)
    with pytest.raises(BrainCounterExhaustedError, match="mutation_seq"):
        BrainComputeCore(state).prepare_feedback(
            target_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=allocation,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_feedback_value_must_be_finite(value: float) -> None:
    state = custom_state()
    with pytest.raises(BrainValidationError, match="value"):
        BrainComputeCore(state).prepare_feedback(
            target_tick=0,
            value=value,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=None,
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf])
def test_feedback_confidence_must_be_finite_and_bounded(confidence: float) -> None:
    state = custom_state()
    with pytest.raises(BrainValidationError, match="confidence"):
        BrainComputeCore(state).prepare_feedback(
            target_tick=0,
            value=1.0,
            confidence=confidence,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=None,
        )


@pytest.mark.parametrize("field_value", [(0.0, 1.0), (1.0, 0.0), (-0.0, 1.0)])
def test_zero_value_or_confidence_is_no_effect_without_allocation(
    field_value: tuple[float, float],
) -> None:
    value, confidence = field_value
    state = custom_state()
    result = BrainComputeCore(state).prepare_feedback(
        target_tick=0,
        value=value,
        confidence=confidence,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert result.status == "no_effect"
    assert result.state == state
    assert result.allocation is None
    assert result.applied_dimensions == ()


def test_exact_ttl_boundary_applies_decayed_target_trace_only() -> None:
    first = BEligibilityRecord(1, 0.0, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    second = BEligibilityRecord(2, 7000.0, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    state = custom_state(
        tick_id=2,
        history_epoch=2,
        mutation_seq=2,
        records=(first, second),
        clock=7100.0,
    )
    allocation = feedback_allocation(state, 1)
    result = BrainComputeCore(state).prepare_feedback(
        target_tick=1,
        value=2.0,
        confidence=0.5,
        trusted_now=7200.0,
        feedback_ttl_seconds=7200.0,
        allocation=allocation,
    )
    decay = math.exp(-7200.0 / 1800.0)
    assert result.status == "applied"
    assert result.allocation == allocation
    assert result.applied_dimensions == (0,)
    assert result.state.gain_b[0] == pytest.approx(0.5 + ETA_B * 0.5 * decay, abs=1e-15)
    assert result.state.theta_b[0] == pytest.approx(0.05 + ETA_THETA * 0.5 * decay, abs=1e-15)
    assert result.state.gain_b[1] == 0.5
    assert result.state.theta_b[1] == 0.05
    assert result.state.clock == state.clock
    assert result.state.tick_id == state.tick_id
    assert result.state.history_epoch == state.history_epoch
    assert result.state.mutation_seq == allocation.next_mutation_seq
    assert result.state.d_plus == state.d_plus
    assert result.state.d_minus == state.d_minus


def test_expired_or_missing_old_target_is_missed_without_guessing_or_allocation() -> None:
    record = BEligibilityRecord(2, 0.0, (0.0, 1.0) + ZERO8[2:])
    state = custom_state(
        tick_id=3, history_epoch=3, mutation_seq=4, records=(record,), clock=8000.0
    )
    core = BrainComputeCore(state)
    missing = core.prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=8000.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    expired = core.prepare_feedback(
        target_tick=2,
        value=1.0,
        confidence=1.0,
        trusted_now=8000.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert missing.status == expired.status == "missed"
    assert missing.applied_dimensions == expired.applied_dimensions == ()
    assert missing.allocation is expired.allocation is None
    assert missing.state == expired.state == state


def test_feedback_rejects_future_target_and_wrong_target_allocation() -> None:
    first = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    second = BEligibilityRecord(2, 0.0, (0.0, 1.0) + ZERO8[2:])
    state = custom_state(tick_id=2, history_epoch=2, mutation_seq=2, records=(first, second))
    core = BrainComputeCore(state)
    with pytest.raises(BrainValidationError, match="future"):
        core.prepare_feedback(
            target_tick=3,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=None,
        )
    with pytest.raises(BrainAllocationError, match="target"):
        core.prepare_feedback(
            target_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=feedback_allocation(state, 2),
        )


def test_feedback_clock_regression_decays_from_state_clock_without_writing_clock() -> None:
    record = BEligibilityRecord(1, 900.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
        clock=1000.0,
        clock_regressions=2,
    )
    result = BrainComputeCore(state).prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=500.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(state, 1),
    )
    decay = math.exp(-100.0 / 1800.0)
    assert result.state.gain_b[0] == pytest.approx(0.5 + ETA_B * decay, abs=1e-15)
    assert result.state.clock == 1000.0
    assert result.state.clock_regressions == 3


def test_feedback_receipt_only_regressions_leave_exact_state_unchanged() -> None:
    missing_state = custom_state(
        tick_id=2,
        history_epoch=2,
        mutation_seq=2,
        records=(BEligibilityRecord(2, 100.0, (1.0,) + ZERO8[1:]),),
        clock=200.0,
        clock_regressions=4,
    )
    missing = BrainComputeCore(missing_state).prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=math.nan,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert missing.status == "missed"
    assert missing.state == missing_state

    expired_state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:]),),
        clock=8000.0,
        clock_regressions=5,
    )
    expired = BrainComputeCore(expired_state).prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=-math.inf,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert expired.status == "missed"
    assert expired.state == expired_state

    clipped_state = custom_state(
        gain_b=(1.0,) + (0.5,) * 7,
        theta_b=(0.95,) + (0.05,) * 7,
        tick_id=1,
        history_epoch=1,
        mutation_seq=5,
        records=(BEligibilityRecord(1, 100.0, (1.0,) + ZERO8[1:]),),
        clock=200.0,
        clock_regressions=6,
    )
    clipped = BrainComputeCore(clipped_state).prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=100.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(clipped_state, 1),
    )
    assert clipped.status == "no_effect"
    assert clipped.allocation is None
    assert clipped.state == clipped_state


@pytest.mark.parametrize(
    "lineage_id",
    [
        "x",
        "not-a-uuid",
        "123E4567-E89B-12D3-A456-426614174000",
        "123e4567e89b12d3a456426614174000",
        "{123e4567-e89b-12d3-a456-426614174000}",
    ],
)
@pytest.mark.parametrize("record_type", ["state", "event_allocation", "feedback_allocation"])
def test_lineage_id_requires_canonical_lowercase_hyphenated_uuid(
    lineage_id: str,
    record_type: str,
) -> None:
    with pytest.raises(BrainValidationError, match="lineage_id.*canonical UUID"):
        if record_type == "state":
            BrainState.fresh(lineage_id=lineage_id)
        elif record_type == "event_allocation":
            EventAllocation(0, lineage_id, 1, 1, 1)
        else:
            FeedbackAllocation(0, lineage_id, 0, 0, 1)


def test_clipped_feedback_no_change_discards_allocation_and_reports_no_dimension() -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        gain_b=(1.0,) + (0.5,) * 7,
        theta_b=(0.95,) + (0.05,) * 7,
        tick_id=1,
        history_epoch=1,
        mutation_seq=5,
        records=(record,),
    )
    result = BrainComputeCore(state).prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(state, 1),
    )
    assert result.status == "no_effect"
    assert result.allocation is None
    assert result.applied_dimensions == ()
    assert result.state == state
    assert result.state.mutation_seq == 5


def test_feedback_reports_only_positive_trace_dimensions_with_actual_change() -> None:
    record = BEligibilityRecord(7, 0.0, (1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    state = custom_state(
        gain_b=(1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        theta_b=(0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        tick_id=7,
        history_epoch=7,
        mutation_seq=10,
        records=(record,),
    )
    result = BrainComputeCore(state).prepare_feedback(
        target_tick=7,
        value=1.0,
        confidence=0.5,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(state, 7),
    )
    assert result.status == "applied"
    assert result.applied_dimensions == (2,)
    assert result.state.gain_b[0] == 1.0
    assert result.state.theta_b[0] == 0.95
    assert result.state.gain_b[2] == pytest.approx(0.5 + ETA_B * 0.5 * 0.5)
    assert result.state.theta_b[2] == pytest.approx(0.05 + ETA_THETA * 0.5 * 0.5)


def test_ring_horizon_discards_oldest_record_and_keeps_c_out_of_b_state() -> None:
    core = BrainComputeCore.fresh(lineage_id=LINEAGE, feedback_horizon=2)
    for index in range(3):
        state = core.state
        candidate = core.prepare_event(
            neutral_event(str(index)),
            allocation=next_event_allocation(state),
            trusted_now=float(index),
        )
        core.commit(candidate)
    state = core.state
    assert tuple(record.tick_id for record in state.eligibility_records) == (2, 3)
    assert state._eligibility_ring.maxlen == 2
    assert not hasattr(state, "c_trace")
    assert not hasattr(state, "dedup")


def test_dose_is_componentwise_monotone_finite_and_bounded_for_hostile_steps() -> None:
    core = BrainComputeCore.fresh(lineage_id=LINEAGE, feedback_horizon=8)
    finite_huge = sys.float_info.max
    times = (math.nan, math.inf, -math.inf, -1.0, 0.0, finite_huge)
    surprises = (math.nan, math.inf, -math.inf, -2.0, 0.25, 2.0)
    raw = (math.nan, math.inf, -math.inf, -2.0, -0.25, 0.0, 0.25, 2.0)
    proposals = (finite_huge, -finite_huge, 1.0, -1.0, 0.0, 0.5, -0.5, 2.0)

    for step in range(10_000):
        before = core.state
        rotated = raw[step % 8 :] + raw[: step % 8]
        event = BrainEvent(
            event_id=f"hostile-{step}",
            assessment=rotated,
            hdc=rotated[::-1],
            wound_sum=rotated[::2] + rotated[1::2],
            surprise=surprises[step % len(surprises)],
            perception_acuity=raw[(step + 3) % len(raw)],
            proposal_c=proposals[step % 8 :] + proposals[: step % 8],
        )
        candidate = core.prepare_event(
            event,
            allocation=next_event_allocation(before),
            trusted_now=times[step % len(times)],
            alpha_c=0.1,
        )
        after = candidate.state
        assert all(math.isfinite(value) and 0.0 <= value <= D_MAX for value in after.d_plus)
        assert all(math.isfinite(value) and 0.0 <= value <= D_MAX for value in after.d_minus)
        assert all(new >= old for new, old in zip(after.d_plus, before.d_plus, strict=True))
        assert all(new >= old for new, old in zip(after.d_minus, before.d_minus, strict=True))
        assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in after.e)
        assert all(math.isfinite(value) and 0.05 <= value <= 1.0 for value in after.gain_b)
        assert all(math.isfinite(value) and 0.0 <= value <= 0.95 for value in after.theta_b)
        assert after.clock >= before.clock
        core.commit(candidate)


def test_brain_state_rejects_future_eligibility_created_at() -> None:
    record = BEligibilityRecord(1, 101.0, (1.0,) + ZERO8[1:])
    with pytest.raises(BrainValidationError, match="created_at"):
        custom_state(
            tick_id=1,
            history_epoch=1,
            mutation_seq=1,
            records=(record,),
            clock=100.0,
        )


def test_brain_state_rejects_descending_eligibility_created_at() -> None:
    records = (
        BEligibilityRecord(1, 20.0, (1.0,) + ZERO8[1:]),
        BEligibilityRecord(2, 10.0, (0.0, 1.0) + ZERO8[2:]),
    )
    with pytest.raises(BrainValidationError, match="created_at"):
        custom_state(
            tick_id=2,
            history_epoch=2,
            mutation_seq=2,
            records=records,
            clock=20.0,
        )


def test_brain_state_accepts_eligibility_created_at_equal_to_clock() -> None:
    record = BEligibilityRecord(1, 100.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
        clock=100.0,
    )
    assert state.eligibility_records == (record,)


def test_brain_state_rejects_none_eligibility_ring_as_validation_error() -> None:
    with pytest.raises(BrainValidationError, match="eligibility_ring"):
        BrainState(
            generation=3,
            lineage_id=LINEAGE,
            e=ZERO8,
            d_plus=ZERO8,
            d_minus=ZERO8,
            gain_b=(0.5,) * 8,
            theta_b=(0.05,) * 8,
            clock=0.0,
            tick_id=0,
            history_epoch=0,
            mutation_seq=0,
            eligibility_ring=None,  # type: ignore[arg-type]
        )


def test_event_allocation_check_rejects_wrong_object_as_validation_error() -> None:
    state = custom_state()
    with pytest.raises(BrainValidationError, match="EventAllocation"):
        evolve_b(
            state,
            neutral_event(),
            allocation=object(),  # type: ignore[arg-type]
        )


def test_feedback_allocation_check_rejects_wrong_object_as_validation_error() -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    with pytest.raises(BrainValidationError, match="FeedbackAllocation"):
        evolve_feedback(
            state,
            target_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=object(),  # type: ignore[arg-type]
        )


def test_publicly_constructed_feedback_candidate_cannot_roll_back_core() -> None:
    committed = custom_state(tick_id=1, history_epoch=1, mutation_seq=1)
    rollback = custom_state()
    core = BrainComputeCore(committed)
    forged = FeedbackCandidate(
        status="missed",
        state=rollback,
        target_tick=0,
        applied_dimensions=(),
        allocation=None,
        base_generation=committed.generation,
        base_lineage_id=committed.lineage_id,
        base_mutation_seq=committed.mutation_seq,
    )
    with pytest.raises(BrainOwnershipError):
        core.commit(forged)
    assert core.state == committed


def test_publicly_constructed_event_candidate_cannot_be_committed() -> None:
    state = custom_state()
    prepared = evolve_b(
        state,
        neutral_event("forged"),
        allocation=next_event_allocation(state),
    )
    forged = EventCandidate(
        event_id=prepared.event_id,
        state=prepared.state,
        allocation=prepared.allocation,
        salience=prepared.salience,
        base_generation=prepared.base_generation,
        base_lineage_id=prepared.base_lineage_id,
        base_mutation_seq=prepared.base_mutation_seq,
        appraisal=prepared.appraisal,
        oriented=prepared.oriented,
        rho_plus=prepared.rho_plus,
        rho_minus=prepared.rho_minus,
        b_trace=prepared.b_trace,
    )
    with pytest.raises(BrainOwnershipError):
        BrainComputeCore(state).commit(forged)


def test_foreign_same_version_candidate_is_rejected_without_consuming_local_candidate() -> None:
    state = custom_state()
    local = BrainComputeCore(state)
    foreign = BrainComputeCore(state)
    local_candidate = local.prepare_event(
        neutral_event("local"), allocation=next_event_allocation(state)
    )
    foreign_candidate = foreign.prepare_event(
        neutral_event("foreign"), allocation=next_event_allocation(state)
    )

    with pytest.raises(BrainOwnershipError):
        local.commit(foreign_candidate)
    assert local.commit(local_candidate) == local_candidate.state


def test_receipt_only_candidate_is_single_use() -> None:
    core = BrainComputeCore(custom_state())
    before = core.state
    receipt = core.prepare_feedback(
        target_tick=0,
        value=0.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert receipt.allocation is None
    assert core.commit(receipt) == before
    with pytest.raises(BrainOwnershipError):
        core.commit(receipt)


def test_new_successful_prepare_invalidates_old_candidate() -> None:
    core = BrainComputeCore(custom_state())
    allocation = next_event_allocation(core.state)
    old = core.prepare_event(neutral_event("old"), allocation=allocation)
    current = core.prepare_event(neutral_event("current"), allocation=allocation)

    with pytest.raises(BrainOwnershipError):
        core.commit(old)
    assert core.commit(current) == current.state


@pytest.mark.parametrize("candidate_kind", ["event", "feedback"])
def test_pure_evolution_candidate_cannot_be_committed(candidate_kind: str) -> None:
    if candidate_kind == "event":
        state = custom_state()
        candidate = evolve_b(
            state,
            neutral_event(),
            allocation=next_event_allocation(state),
        )
    else:
        record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
        state = custom_state(
            tick_id=1,
            history_epoch=1,
            mutation_seq=1,
            records=(record,),
        )
        candidate = evolve_feedback(
            state,
            target_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=0.0,
            feedback_ttl_seconds=7200.0,
            allocation=feedback_allocation(state, 1),
        )
    with pytest.raises(BrainOwnershipError):
        BrainComputeCore(state).commit(candidate)


def test_commit_retains_stale_candidate_guard_and_consumes_candidate() -> None:
    core = BrainComputeCore(custom_state())
    candidate = core.prepare_event(neutral_event(), allocation=next_event_allocation(core.state))
    core._state = custom_state(mutation_seq=1)  # type: ignore[attr-defined]

    with pytest.raises(BrainAllocationError, match="stale"):
        core.commit(candidate)
    with pytest.raises(BrainOwnershipError):
        core.commit(candidate)


@pytest.mark.parametrize("violation", ["counter", "dose", "gain", "theta"])
def test_commit_rejects_invalid_event_transition(violation: str) -> None:
    state = custom_state(d_plus=(1.0,) + ZERO8[1:])
    core = BrainComputeCore(state)
    candidate = core.prepare_event(neutral_event(), allocation=next_event_allocation(state))
    if violation == "counter":
        object.__setattr__(candidate.state, "history_epoch", state.history_epoch)
    elif violation == "dose":
        object.__setattr__(candidate.state, "_BrainState__d_plus", array("d", (0.5,) + ZERO8[1:]))
    elif violation == "gain":
        object.__setattr__(candidate.state, "_BrainState__gain_b", array("d", (0.6,) * 8))
    else:
        object.__setattr__(candidate.state, "_BrainState__theta_b", array("d", (0.1,) * 8))

    with pytest.raises(BrainValidationError, match="event transition"):
        core.commit(candidate)


@pytest.mark.parametrize(
    "violation",
    ["e", "d_plus", "d_minus", "tick", "history", "clock", "mutation"],
)
def test_commit_rejects_invalid_applied_feedback_transition(violation: str) -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    core = BrainComputeCore(state)
    candidate = core.prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(state, 1),
    )
    if violation == "e":
        object.__setattr__(candidate.state, "_BrainState__e", array("d", (0.1,) + ZERO8[1:]))
    elif violation == "d_plus":
        object.__setattr__(candidate.state, "_BrainState__d_plus", array("d", (0.1,) + ZERO8[1:]))
    elif violation == "d_minus":
        object.__setattr__(candidate.state, "_BrainState__d_minus", array("d", (0.1,) + ZERO8[1:]))
    elif violation == "tick":
        object.__setattr__(candidate.state, "tick_id", state.tick_id + 1)
    elif violation == "history":
        object.__setattr__(candidate.state, "history_epoch", state.history_epoch + 1)
    elif violation == "clock":
        object.__setattr__(candidate.state, "clock", state.clock + 1.0)
    else:
        object.__setattr__(candidate.state, "mutation_seq", state.mutation_seq)

    with pytest.raises(BrainValidationError, match="feedback transition"):
        core.commit(candidate)


@pytest.mark.parametrize("violation", ["state", "allocation"])
def test_commit_rejects_invalid_receipt_only_transition(violation: str) -> None:
    state = custom_state()
    core = BrainComputeCore(state)
    candidate = core.prepare_feedback(
        target_tick=0,
        value=0.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    if violation == "state":
        object.__setattr__(candidate.state, "mutation_seq", 1)
    else:
        object.__setattr__(candidate, "allocation", feedback_allocation(state, 0))

    with pytest.raises(BrainValidationError, match="receipt-only"):
        core.commit(candidate)


def test_event_candidate_keeps_immutable_state_without_constructor_copy() -> None:
    state = custom_state()
    allocation = next_event_allocation(state)
    candidate = EventCandidate(
        event_id="identity",
        state=state,
        allocation=allocation,
        salience=0.0,
        base_generation=state.generation,
        base_lineage_id=state.lineage_id,
        base_mutation_seq=state.mutation_seq,
        appraisal=ZERO8,
        oriented=ZERO8,
        rho_plus=ZERO8,
        rho_minus=ZERO8,
        b_trace=ZERO8,
    )
    assert candidate.state is state


def test_commit_pointer_swaps_candidate_state_but_returns_defensive_copy() -> None:
    core = BrainComputeCore(custom_state())
    candidate = core.prepare_event(neutral_event(), allocation=next_event_allocation(core.state))
    candidate_state = candidate.state
    committed = core.commit(candidate)

    assert object.__getattribute__(core, "_state") is candidate_state
    assert committed == candidate_state
    assert committed is not candidate_state
    assert core.state is not candidate_state


def test_commit_rejects_event_clock_below_base_before_pointer_swap() -> None:
    state = custom_state(clock=100.0)
    core = BrainComputeCore(state)
    raw_before = object.__getattribute__(core, "_state")
    candidate = core.prepare_event(
        neutral_event(),
        allocation=next_event_allocation(state),
        trusted_now=200.0,
    )
    object.__setattr__(candidate.state, "clock", 99.0)

    with pytest.raises(BrainValidationError, match="event transition.*clock"):
        core.commit(candidate)

    raw_after = object.__getattribute__(core, "_state")
    assert raw_after is raw_before
    assert raw_after.clock == 100.0
    with pytest.raises(BrainOwnershipError):
        core.commit(candidate)


@pytest.mark.parametrize("violation", ["ring", "horizon"])
def test_commit_rejects_applied_feedback_eligibility_change_before_pointer_swap(
    violation: str,
) -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    core = BrainComputeCore(state)
    raw_before = object.__getattribute__(core, "_state")
    candidate = core.prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(state, 1),
    )
    if violation == "ring":
        replacement = deque((), maxlen=candidate.state.eligibility_horizon)
    else:
        replacement = deque(
            candidate.state.eligibility_records,
            maxlen=candidate.state.eligibility_horizon + 1,
        )
    object.__setattr__(candidate.state, "_BrainState__eligibility_ring", replacement)

    with pytest.raises(BrainValidationError, match="feedback transition.*eligibility"):
        core.commit(candidate)

    assert object.__getattribute__(core, "_state") is raw_before
    assert core.state == state
    with pytest.raises(BrainOwnershipError):
        core.commit(candidate)


def _assert_integrity_rejection_is_atomic(
    core: BrainComputeCore,
    candidate: EventCandidate | FeedbackCandidate,
    raw_before: BrainState,
    snapshot: BrainState,
    *,
    match: str = "integrity",
) -> None:
    with pytest.raises(BrainValidationError, match=match):
        core.commit(candidate)

    raw_after = object.__getattribute__(core, "_state")
    assert raw_after is raw_before
    assert raw_after == snapshot
    with pytest.raises(BrainOwnershipError):
        core.commit(candidate)


@pytest.mark.parametrize(
    "violation",
    [
        "state_e_nan",
        "eligibility",
        "clock_regressions",
        "candidate_trace",
        "candidate_trace_storage",
        "event_id",
    ],
)
def test_event_integrity_seal_rejects_complete_state_or_metadata_tampering(
    violation: str,
) -> None:
    core = BrainComputeCore(custom_state())
    raw_before = object.__getattribute__(core, "_state")
    snapshot = raw_before.copy()
    candidate = core.prepare_event(
        neutral_event("sealed-event"),
        allocation=next_event_allocation(raw_before),
    )

    if violation == "state_e_nan":
        raw_e = object.__getattribute__(candidate.state, "_BrainState__e")
        raw_e[0] = math.nan
    elif violation == "eligibility":
        raw_ring = object.__getattribute__(candidate.state, "_BrainState__eligibility_ring")
        raw_trace = object.__getattribute__(raw_ring[-1], "_BEligibilityRecord__b_trace")
        raw_trace[0] = 0.5
    elif violation == "clock_regressions":
        object.__setattr__(candidate.state, "clock_regressions", 1)
    elif violation == "candidate_trace":
        raw_trace = object.__getattribute__(candidate, "_EventCandidate__b_trace")
        raw_trace[0] = 0.5
    elif violation == "candidate_trace_storage":
        object.__setattr__(candidate, "_EventCandidate__b_trace", object())
    else:
        object.__setattr__(candidate, "event_id", "tampered-event")

    _assert_integrity_rejection_is_atomic(core, candidate, raw_before, snapshot)


@pytest.mark.parametrize("violation", ["gain", "theta", "applied_dimensions"])
def test_applied_feedback_integrity_seal_rejects_result_or_metadata_tampering(
    violation: str,
) -> None:
    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    core = BrainComputeCore(state)
    raw_before = object.__getattribute__(core, "_state")
    snapshot = raw_before.copy()
    candidate = core.prepare_feedback(
        target_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=feedback_allocation(raw_before, 1),
    )
    assert candidate.status == "applied"

    if violation == "gain":
        raw_gain = object.__getattribute__(candidate.state, "_BrainState__gain_b")
        raw_gain[0] += 0.001
    elif violation == "theta":
        raw_theta = object.__getattribute__(candidate.state, "_BrainState__theta_b")
        raw_theta[0] += 0.001
    else:
        object.__setattr__(candidate, "applied_dimensions", (7,))

    _assert_integrity_rejection_is_atomic(core, candidate, raw_before, snapshot)


@pytest.mark.parametrize(
    "violation",
    ["state_e", "eligibility", "clock_regressions", "status", "target_tick", "dimensions"],
)
def test_receipt_integrity_seal_rejects_isolated_state_or_metadata_tampering(
    violation: str,
) -> None:
    record = BEligibilityRecord(1, 0.0, (0.25,) + ZERO8[1:])
    state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    core = BrainComputeCore(state)
    raw_before = object.__getattribute__(core, "_state")
    snapshot = raw_before.copy()
    candidate = core.prepare_feedback(
        target_tick=1,
        value=0.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert candidate.status == "no_effect"
    assert candidate.state is not raw_before

    if violation == "state_e":
        raw_e = object.__getattribute__(candidate.state, "_BrainState__e")
        raw_e[0] = 0.5
    elif violation == "eligibility":
        raw_ring = object.__getattribute__(candidate.state, "_BrainState__eligibility_ring")
        raw_trace = object.__getattribute__(raw_ring[-1], "_BEligibilityRecord__b_trace")
        raw_trace[0] = 0.75
    elif violation == "clock_regressions":
        object.__setattr__(candidate.state, "clock_regressions", 1)
    elif violation == "status":
        object.__setattr__(candidate, "status", "missed")
    elif violation == "target_tick":
        object.__setattr__(candidate, "target_tick", 0)
    else:
        object.__setattr__(candidate, "applied_dimensions", (0,))

    _assert_integrity_rejection_is_atomic(
        core,
        candidate,
        raw_before,
        snapshot,
        match="integrity|receipt-only",
    )


def test_feedback_candidate_copy_policy_preserves_atomicity_without_copying_applied_state() -> None:
    base = custom_state()
    receipt = _feedback_candidate(
        status="no_effect",
        state=base,
        target_tick=0,
        base=base,
    )
    assert receipt.state == base
    assert receipt.state is not base

    record = BEligibilityRecord(1, 0.0, (1.0,) + ZERO8[1:])
    next_state = custom_state(
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
        records=(record,),
    )
    allocation = feedback_allocation(next_state, 1)
    applied = _feedback_candidate(
        status="applied",
        state=next_state,
        target_tick=1,
        applied_dimensions=(0,),
        allocation=allocation,
        base=next_state,
    )
    assert applied.state is next_state


def test_receipt_commit_keeps_raw_state_pointer_and_returns_defensive_copy() -> None:
    core = BrainComputeCore(custom_state())
    raw_before = object.__getattribute__(core, "_state")
    receipt = core.prepare_feedback(
        target_tick=0,
        value=0.0,
        confidence=1.0,
        trusted_now=0.0,
        feedback_ttl_seconds=7200.0,
        allocation=None,
    )
    assert receipt.state is not raw_before

    committed = core.commit(receipt)

    assert object.__getattribute__(core, "_state") is raw_before
    assert committed == raw_before
    assert committed is not raw_before
