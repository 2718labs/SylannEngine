from __future__ import annotations

import ast
import hashlib
import inspect
import math
import struct
from array import array
from collections import deque
from itertools import repeat
from pathlib import Path
from typing import Iterator, TypeVar

import pytest

import sylanne_core.compute.brain_c as brain_c
from sylanne_core.compute.brain_c import (
    DEFAULT_W_OUT,
    E_MAX,
    ETA_C,
    ETA_OUT,
    FEEDBACK_DECAY_SECONDS,
    MODEL_SEED,
    N_AXES,
    N_BASIS,
    N_CHANNELS,
    N_EDGES,
    N_NEURONS,
    TOPOLOGY,
    TOPOLOGY_DIGEST,
    W_MAX,
    W_OUT_MAX,
    BrainCState,
    CEligibilityRecord,
    build_topology,
    evolve_c_event,
    evolve_c_feedback,
    split_signed_input,
)
from sylanne_core.compute.brain_compute import BrainEvent, evolve_b
from sylanne_core.compute.brain_errors import BrainValidationError
from sylanne_core.compute.brain_state import BrainState, EventAllocation

ZERO8 = (0.0,) * 8
T = TypeVar("T")


def _guarded_repeat(value: T, allowed_reads: int, consumed: list[int]) -> Iterator[T]:
    for _ in repeat(None, allowed_reads):
        consumed[0] += 1
        yield value
    raise AssertionError("iterable boundary consumed more than expected + 1 values")


def _raw_state_array(state: BrainCState, field: str) -> array[float]:
    return object.__getattribute__(state, f"_BrainCState__{field}")


def _raw_state_ring(state: BrainCState) -> deque[CEligibilityRecord]:
    return object.__getattribute__(state, "_BrainCState__eligibility_ring")


def _raw_record_trace(record: CEligibilityRecord) -> array[float]:
    return object.__getattribute__(record, "_CEligibilityRecord__c_trace")


def _state(
    *,
    v: tuple[float, ...] = (0.0,) * 32,
    adaptation: tuple[float, ...] = (0.0,) * 32,
    filtered: tuple[float, ...] = (0.0,) * 32,
    weights: tuple[float, ...] = (0.0,) * 128,
    w_out: tuple[float, ...] = DEFAULT_W_OUT,
    records: tuple[CEligibilityRecord, ...] = (),
    horizon: int = 8,
) -> BrainCState:
    return BrainCState(
        v=v,
        adaptation=adaptation,
        filtered=filtered,
        weights=weights,
        w_out=w_out,
        eligibility_ring=records,
        eligibility_horizon=horizon,
    )


def _one_hot_basis(slot: int) -> tuple[float, ...]:
    basis = [0.0] * N_BASIS
    basis[slot] = 1.0
    return tuple(basis)


def _one_hot_axis(axis: int) -> tuple[float, ...]:
    values = [0.0] * N_AXES
    values[axis] = 1.0
    return tuple(values)


def test_topology_weight_and_digest_contract() -> None:
    topology = build_topology(MODEL_SEED)
    assert MODEL_SEED == 42
    assert N_NEURONS == 32
    assert N_CHANNELS == 16
    assert N_EDGES == 128
    assert topology == TOPOLOGY
    assert len(topology.edges) == N_EDGES
    assert all(
        topology.incoming[post] == tuple((post + offset) % N_NEURONS for offset in (1, 5, 13, 17))
        for post in range(N_NEURONS)
    )
    assert all(pre != post for pre, post in topology.edges)
    assert all(len(set(topology.incoming[post])) == 4 for post in range(N_NEURONS))

    encoded = b"".join(struct.pack(">BB", pre, post) for pre, post in topology.edges)
    assert hashlib.sha256(encoded).hexdigest() == TOPOLOGY_DIGEST
    assert TOPOLOGY_DIGEST == ("1a4d0d707929a190184daf9f47857a271bc5ce2752ea8b65a354c9ca3e06d04e")
    expected = tuple(
        ((17 * pre + 31 * post + MODEL_SEED) % 2001 - 1000) / 5000.0 for pre, post in topology.edges
    )
    assert topology.initial_weights == expected
    assert expected[:8] == pytest.approx(
        (-0.1882, -0.1746, -0.1474, -0.1338, -0.1786, -0.165, -0.1378, -0.1242)
    )


def test_signed_input_flip_swaps_only_one_pair_and_sanitizes_hostile_values() -> None:
    positive = split_signed_input((0.5,) + ZERO8[1:])
    negative = split_signed_input((-0.5,) + ZERO8[1:])
    assert positive[:2] == (0.5, 0.0)
    assert negative[:2] == (0.0, 0.5)
    assert positive[2:] == negative[2:]
    assert split_signed_input((math.nan, math.inf, -math.inf, 5.0, -5.0, 0.0, 0.0, 0.0)) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_float32_storage_defensive_copies_and_constructor_validation() -> None:
    v = array("f", (0.25,) * N_NEURONS)
    weights = array("f", (0.125,) * N_EDGES)
    trace = array("f", (0.5,) * N_EDGES)
    record = CEligibilityRecord(1, 2.0, trace)
    state = BrainCState(
        v=v,
        adaptation=(0.0,) * N_NEURONS,
        filtered=(0.0,) * N_NEURONS,
        weights=weights,
        eligibility_ring=(record,),
        eligibility_horizon=2,
    )
    v[0] = 1.0
    weights[0] = 1.0
    trace[0] = 1.0
    exposed_v = state.v
    exposed_weights = state.weights
    exposed_trace = state.eligibility_records[0].c_trace
    exposed_v[0] = -1.0
    exposed_weights[0] = -1.0
    exposed_trace[0] = 2.0

    assert state.v.typecode == "f" and state.v[0] == 0.25
    assert state.adaptation.typecode == "f"
    assert state.filtered.typecode == "f"
    assert state.weights.typecode == "f" and state.weights[0] == 0.125
    assert state.eligibility_records[0].c_trace.typecode == "f"
    assert state.eligibility_records[0].c_trace[0] == 0.5

    with pytest.raises(BrainValidationError, match="v.*32"):
        _state(v=(0.0,) * 31)
    with pytest.raises(BrainValidationError, match="weights.*128"):
        _state(weights=(0.0,) * 127)
    with pytest.raises(BrainValidationError, match="finite"):
        _state(filtered=(math.nan,) + (0.0,) * 31)
    with pytest.raises(BrainValidationError, match="outside"):
        _state(adaptation=(1.1,) + (0.0,) * 31)
    with pytest.raises(BrainValidationError, match="horizon"):
        _state(horizon=0)
    with pytest.raises(BrainValidationError, match="ordered"):
        _state(
            records=(
                CEligibilityRecord(2, 2.0, (0.0,) * N_EDGES),
                CEligibilityRecord(1, 3.0, (0.0,) * N_EDGES),
            )
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    (("v", N_NEURONS), ("adaptation", N_NEURONS), ("filtered", N_NEURONS), ("weights", N_EDGES)),
)
def test_state_array_iterables_consume_at_most_expected_plus_one(
    field: str,
    expected: int,
) -> None:
    consumed = [0]
    kwargs = {
        "v": (0.0,) * N_NEURONS,
        "adaptation": (0.0,) * N_NEURONS,
        "filtered": (0.0,) * N_NEURONS,
        "weights": (0.0,) * N_EDGES,
    }
    kwargs[field] = _guarded_repeat(0.0, expected + 1, consumed)
    with pytest.raises(BrainValidationError, match=rf"{field}.*{expected}"):
        BrainCState(**kwargs)
    assert consumed == [expected + 1]


def test_appraisal_iterable_consumes_at_most_expected_plus_one() -> None:
    appraisal_reads = [0]
    with pytest.raises(BrainValidationError, match="eight"):
        split_signed_input(_guarded_repeat(0.0, N_AXES + 1, appraisal_reads))
    assert appraisal_reads == [N_AXES + 1]


def test_c_trace_iterable_consumes_at_most_expected_plus_one() -> None:
    trace_reads = [0]
    with pytest.raises(BrainValidationError, match="c_trace.*128"):
        CEligibilityRecord(1, 0.0, _guarded_repeat(0.0, N_EDGES + 1, trace_reads))
    assert trace_reads == [N_EDGES + 1]


def test_eligibility_ring_iterable_consumes_at_most_horizon_plus_one() -> None:
    ring_reads = [0]
    record = CEligibilityRecord(1, 0.0, (0.0,) * N_EDGES)
    with pytest.raises(BrainValidationError, match="eligibility_ring exceeds"):
        BrainCState(
            v=(0.0,) * N_NEURONS,
            adaptation=(0.0,) * N_NEURONS,
            filtered=(0.0,) * N_NEURONS,
            weights=(0.0,) * N_EDGES,
            eligibility_ring=_guarded_repeat(record, 3, ring_reads),
            eligibility_horizon=2,
        )
    assert ring_reads == [3]


def test_fast_route_performs_only_analytic_decay_and_records_zero_local_trace() -> None:
    state = _state(
        v=(1.0,) * N_NEURONS,
        adaptation=(1.0,) * N_NEURONS,
        filtered=(1.0,) * N_NEURONS,
    )
    candidate = evolve_c_event(
        state,
        (1.0,) * 8,
        route="fast",
        tick_id=7,
        created_at=100.0,
        delta_t=10.0,
    )

    assert candidate.relaxations == 0
    assert candidate.state.v == pytest.approx(array("f", (math.exp(-1.0),) * 32))
    assert candidate.state.adaptation == pytest.approx(array("f", (math.exp(-1 / 6),) * 32))
    assert candidate.state.filtered == pytest.approx(array("f", (math.exp(-2.0),) * 32))
    assert candidate.c_trace == array("f", (0.0,) * N_EDGES)
    # basis = filtered(post-decay, exp(-2) on every neuron) ++
    # split_signed_input((1.0,)*8) (each axis: positive channel 1.0, negative 0.0).
    expected_basis = array("f", (math.exp(-2.0),) * N_NEURONS)
    expected_basis.extend((1.0, 0.0) * 8)
    assert candidate.state.eligibility_records[-1] == CEligibilityRecord(
        7, 100.0, (0.0,) * N_EDGES, expected_basis
    )
    assert state.v == array("f", (1.0,) * N_NEURONS)
    assert state.eligibility_records == ()


@pytest.mark.parametrize(
    ("tick_id", "created_at", "message"),
    ((7, 101.0, "tick_id"), (8, 99.0, "created_at")),
)
def test_event_rejects_stale_order_before_appraisal_or_dynamics(
    monkeypatch: pytest.MonkeyPatch,
    tick_id: int,
    created_at: float,
    message: str,
) -> None:
    state = _state(records=(CEligibilityRecord(7, 100.0, (0.0,) * N_EDGES),))
    split_calls = 0
    decay_calls = 0

    def explode_split(_values: object) -> tuple[float, ...]:
        nonlocal split_calls
        split_calls += 1
        raise AssertionError("stale event reached appraisal splitting")

    def explode_decay(*_args: object, **_kwargs: object) -> array[float]:
        nonlocal decay_calls
        decay_calls += 1
        raise AssertionError("stale event reached dynamics")

    monkeypatch.setattr(brain_c, "split_signed_input", explode_split)
    monkeypatch.setattr(brain_c, "_decay", explode_decay)
    with pytest.raises(BrainValidationError, match=message):
        evolve_c_event(
            state,
            _guarded_repeat(0.0, N_AXES + 1, [0]),
            route="normal",
            tick_id=tick_id,
            created_at=created_at,
            delta_t=0.0,
        )
    assert split_calls == 0
    assert decay_calls == 0


def test_owned_event_path_rejects_non_float32_internal_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def wrong_type_decay(
        values: array[float],
        _delta_t: float,
        _tau: float,
        _lower: float,
    ) -> array[float]:
        return array("d", values)

    monkeypatch.setattr(brain_c, "_decay", wrong_type_decay)
    with pytest.raises(BrainValidationError, match="float32"):
        evolve_c_event(
            _state(),
            ZERO8,
            route="fast",
            tick_id=1,
            created_at=0.0,
            delta_t=0.0,
        )


def test_normal_route_executes_exactly_four_synchronous_steps() -> None:
    candidate = evolve_c_event(
        _state(),
        (1.0,) + ZERO8[1:],
        route="normal",
        tick_id=1,
        created_at=0.0,
        delta_t=0.0,
    )
    # A strong positive appraisal on axis 0 drives its two positive-channel
    # neurons (0, 1) across threshold within the 4-step relaxation (tuned
    # INPUT_GAIN_EVEN/ODD); the negative-channel neurons (2, 3) and every other
    # axis stay silent. Values are the deterministic 4-step result.
    assert candidate.relaxations == 4
    assert candidate.state.v[:4] == pytest.approx(array("f", (0.0, 0.49375, 0.0, 0.0)), abs=1e-6)
    assert candidate.state.v[4:] == array("f", (0.0,) * 28)
    # Neurons 0 and 1 spiked (nonzero low-pass filtered trace + adaptation);
    # neurons 2, 3 and all other axes did not.
    assert candidate.state.filtered[:4] == pytest.approx(
        array("f", (0.328, 0.16, 0.0, 0.0)), abs=1e-6
    )
    assert candidate.state.filtered[4:] == array("f", (0.0,) * 28)
    assert candidate.state.adaptation[:4] == pytest.approx(
        array("f", (0.19025, 0.095, 0.0, 0.0)), abs=1e-6
    )
    assert candidate.state.adaptation[4:] == array("f", (0.0,) * 28)
    # Bounded, sign-correct proposal on the driven axis only.
    assert candidate.proposal[0] == pytest.approx(0.23927034, abs=1e-6)
    assert candidate.proposal[1:] == pytest.approx(ZERO8[1:])
    assert all(-1.0 <= value <= 1.0 for value in candidate.proposal)


def test_hard_reset_uses_pre_reset_pseudo_and_synchronous_spikes() -> None:
    # Isolate the spike/eligibility/reset mechanism from input tuning: inject a
    # uniform supra-threshold membrane (v = 1.4) with a ZERO appraisal, so every
    # neuron crosses synchronously in step one from the same pre-reset value and
    # the eligibility trace is uniform across all edges.
    candidate = evolve_c_event(
        _state(v=(1.4,) * N_NEURONS),
        ZERO8,
        route="full",
        tick_id=1,
        created_at=0.0,
        delta_t=0.0,
    )
    # Every presynaptic unit crosses in step one; eligibility observes those
    # spikes synchronously with the pseudo-derivative taken from the pre-reset v
    # (0.25 * (1 - 0.05/0.5) = 0.225), then decays 0.95 per remaining step.
    expected = 0.225 * 0.95**3
    assert candidate.c_trace == pytest.approx(array("f", (expected,) * N_EDGES), rel=1e-5)
    # Hard reset: v is driven to 0 at the spike, then re-integrates toward the
    # small negative adaptation-only drive over the remaining steps.
    assert all(-0.5 < value <= 0.0 for value in candidate.state.v)
    assert all(value > 0.0 for value in candidate.state.adaptation)
    assert all(value > 0.0 for value in candidate.state.filtered)


def test_fast_zero_trace_and_normal_trace_are_event_local_not_inherited() -> None:
    seeded = evolve_c_event(
        _state(v=(1.4,) * N_NEURONS),
        (1.0,) + ZERO8[1:],
        route="normal",
        tick_id=1,
        created_at=1.0,
        delta_t=0.0,
    ).state
    assert any(value > 0.0 for value in seeded.eligibility_records[-1].c_trace)

    fast = evolve_c_event(
        seeded,
        ZERO8,
        route="fast",
        tick_id=2,
        created_at=2.0,
        delta_t=1.0,
    )
    assert fast.c_trace == array("f", (0.0,) * N_EDGES)

    without_history = BrainCState(
        v=seeded.v,
        adaptation=seeded.adaptation,
        filtered=seeded.filtered,
        weights=seeded.weights,
        eligibility_horizon=seeded.eligibility_horizon,
    )
    continued = evolve_c_event(
        seeded,
        ZERO8,
        route="normal",
        tick_id=2,
        created_at=2.0,
        delta_t=0.0,
    )
    isolated = evolve_c_event(
        without_history,
        ZERO8,
        route="normal",
        tick_id=2,
        created_at=2.0,
        delta_t=0.0,
    )
    assert continued.c_trace == isolated.c_trace


def test_feedback_targets_exact_old_tick_honors_ttl_boundary_and_decay() -> None:
    old = CEligibilityRecord(3, 0.0, (2.0,) + (0.0,) * (N_EDGES - 1))
    current = CEligibilityRecord(4, 100.0, (0.0, 3.0) + (0.0,) * (N_EDGES - 2))
    state = _state(records=(old, current))
    candidate = evolve_c_feedback(
        state,
        target_tick=3,
        state_tick=4,
        value=1.0,
        confidence=1.0,
        trusted_now=7200.0,
        state_clock=7200.0,
        feedback_ttl_seconds=7200.0,
    )

    # New e-prop rule: dw = ETA_C * certainty * L_post * trace * decay, where
    # L_post = W_out[post][axis] * axis_values[axis] (post=0 sits in DEFAULT_W_OUT's
    # axis-0 block at +0.5, and the scalar value=1.0 broadcasts to every axis).
    expected = array("f", (ETA_C * 0.5 * 2.0 * math.exp(-7200.0 / FEEDBACK_DECAY_SECONDS),))[0]
    assert candidate.status == "applied"
    assert candidate.applied_synapses == (0,)
    assert candidate.state.weights[0] == expected
    assert candidate.state.weights[1:] == state.weights[1:]
    assert state.weights == array("f", (0.0,) * N_EDGES)

    expired = evolve_c_feedback(
        state,
        target_tick=3,
        state_tick=4,
        value=1.0,
        confidence=1.0,
        trusted_now=7200.0001,
        state_clock=7200.0001,
        feedback_ttl_seconds=7200.0,
    )
    assert expired.status == "missed"
    assert expired.applied_synapses == ()
    assert expired.state is state


def test_feedback_missing_target_clipped_and_float32_noops_are_not_reported() -> None:
    record = CEligibilityRecord(1, 0.0, (E_MAX,) + (1.0,) * (N_EDGES - 1))
    saturated = _state(weights=(W_MAX,) + (0.5,) * (N_EDGES - 1), records=(record,))
    clipped = evolve_c_feedback(
        saturated,
        target_tick=1,
        state_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        state_clock=0.0,
        feedback_ttl_seconds=1.0,
    )
    assert clipped.status == "applied"
    assert 0 not in clipped.applied_synapses
    assert clipped.state.weights[0] == W_MAX

    no_float32_change = evolve_c_feedback(
        _state(weights=(0.5,) * N_EDGES, records=(record,)),
        target_tick=1,
        state_tick=1,
        value=1e-12,
        confidence=1.0,
        trusted_now=0.0,
        state_clock=0.0,
        feedback_ttl_seconds=1.0,
    )
    assert no_float32_change.status == "no_effect"
    assert no_float32_change.applied_synapses == ()

    missing = evolve_c_feedback(
        saturated,
        target_tick=999,
        state_tick=999,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        state_clock=0.0,
        feedback_ttl_seconds=1.0,
    )
    assert missing.status == "missed"
    assert missing.state is saturated


def test_feedback_requires_caller_owned_state_tick_and_rejects_future_target() -> None:
    state_tick = inspect.signature(evolve_c_feedback).parameters["state_tick"]
    assert state_tick.kind is inspect.Parameter.KEYWORD_ONLY
    assert state_tick.default is inspect.Parameter.empty

    state = _state(records=(CEligibilityRecord(1, 0.0, (1.0,) * N_EDGES),))
    common = {
        "state": state,
        "target_tick": 1,
        "value": 1.0,
        "confidence": 1.0,
        "trusted_now": 0.0,
        "state_clock": 0.0,
        "feedback_ttl_seconds": 1.0,
    }
    with pytest.raises(BrainValidationError, match="state_tick.*non-boolean integer"):
        evolve_c_feedback(**common, state_tick=True)
    with pytest.raises(BrainValidationError, match="target_tick.*state_tick"):
        evolve_c_feedback(**(common | {"target_tick": 2}), state_tick=1)


def test_feedback_rejects_target_record_created_after_authoritative_clock() -> None:
    state = _state(records=(CEligibilityRecord(1, 10_000.0, (1.0,) * N_EDGES),))
    with pytest.raises(BrainValidationError, match="created_at.*state_clock"):
        evolve_c_feedback(
            state,
            target_tick=1,
            state_tick=1,
            value=1.0,
            confidence=1.0,
            trusted_now=10_000.0,
            state_clock=100.0,
            feedback_ttl_seconds=7200.0,
        )


def test_feedback_clips_finite_value_but_keeps_confidence_strict() -> None:
    state = _state(records=(CEligibilityRecord(1, 0.0, (1.0,) * N_EDGES),))
    common = {
        "state": state,
        "target_tick": 1,
        "state_tick": 1,
        "confidence": 1.0,
        "trusted_now": 0.0,
        "state_clock": 0.0,
        "feedback_ttl_seconds": 1.0,
    }
    assert evolve_c_feedback(**common, value=2.0) == evolve_c_feedback(**common, value=1.0)
    with pytest.raises(BrainValidationError, match=r"confidence.*\[0, 1\]"):
        evolve_c_feedback(**(common | {"confidence": 2.0}), value=1.0)


def test_public_construction_copies_but_event_reuses_only_owned_immutable_storage() -> None:
    source_v = array("f", (0.25,) * N_NEURONS)
    source_record = CEligibilityRecord(1, 1.0, (0.0,) * N_EDGES)
    public_state = BrainCState(
        v=source_v,
        adaptation=(0.0,) * N_NEURONS,
        filtered=(0.0,) * N_NEURONS,
        weights=(0.0,) * N_EDGES,
        eligibility_ring=(source_record,),
        eligibility_horizon=32,
    )
    public_ring = _raw_state_ring(public_state)
    assert _raw_state_array(public_state, "v") is not source_v
    assert public_ring[0] is not source_record
    assert _raw_record_trace(public_ring[0]) is not _raw_record_trace(source_record)

    records = tuple(
        CEligibilityRecord(tick, float(tick), (float(tick % 2),) * N_EDGES) for tick in range(1, 33)
    )
    old = _state(records=records, horizon=32)
    old_snapshot = old.copy()
    old_ring = _raw_state_ring(old)
    candidate = evolve_c_event(
        old,
        ZERO8,
        route="fast",
        tick_id=33,
        created_at=33.0,
        delta_t=0.0,
    )
    new = candidate.state
    new_snapshot = new.copy()
    new_ring = _raw_state_ring(new)

    assert all(new_ring[index] is old_ring[index + 1] for index in range(31))
    assert all(
        _raw_record_trace(new_ring[index]) is _raw_record_trace(old_ring[index + 1])
        for index in range(31)
    )
    assert new_ring[-1] is not old_ring[-1]
    assert all(
        _raw_record_trace(new_ring[-1]) is not _raw_record_trace(record) for record in old_ring
    )
    assert _raw_state_array(new, "weights") is _raw_state_array(old, "weights")
    assert _raw_state_array(new, "v") is not _raw_state_array(old, "v")
    assert _raw_state_array(new, "adaptation") is not _raw_state_array(old, "adaptation")
    assert _raw_state_array(new, "filtered") is not _raw_state_array(old, "filtered")

    old.v[0] = -1.0
    old.weights[0] = -1.0
    old.eligibility_records[-1].c_trace[0] = E_MAX
    new.v[0] = -1.0
    new.weights[0] = -1.0
    new.eligibility_records[-1].c_trace[0] = E_MAX
    assert old == old_snapshot
    assert new == new_snapshot


def test_applied_feedback_reuses_unchanged_owned_state_storage() -> None:
    state = _state(records=(CEligibilityRecord(1, 0.0, (1.0,) * N_EDGES),))
    candidate = evolve_c_feedback(
        state,
        target_tick=1,
        state_tick=1,
        value=1.0,
        confidence=1.0,
        trusted_now=0.0,
        state_clock=0.0,
        feedback_ttl_seconds=1.0,
    )
    assert candidate.status == "applied"
    assert _raw_state_array(candidate.state, "v") is _raw_state_array(state, "v")
    assert _raw_state_array(candidate.state, "adaptation") is _raw_state_array(state, "adaptation")
    assert _raw_state_array(candidate.state, "filtered") is _raw_state_array(state, "filtered")
    assert _raw_state_array(candidate.state, "weights") is not _raw_state_array(state, "weights")
    assert _raw_state_ring(candidate.state) is _raw_state_ring(state)


@pytest.mark.parametrize("trusted_now", [99.0, math.nan, math.inf, -math.inf])
def test_feedback_nonfinite_or_backward_clock_uses_authoritative_state_clock(
    trusted_now: float,
) -> None:
    state = _state(records=(CEligibilityRecord(1, 90.0, (1.0,) * N_EDGES),))
    common = {
        "state": state,
        "target_tick": 1,
        "state_tick": 1,
        "value": 1.0,
        "confidence": 1.0,
        "state_clock": 100.0,
        "feedback_ttl_seconds": 20.0,
    }
    expected = evolve_c_feedback(**common, trusted_now=100.0)
    assert evolve_c_feedback(**common, trusted_now=trusted_now) == expected


def test_restore_continuation_is_bit_identical() -> None:
    first = evolve_c_event(
        BrainCState.fresh(feedback_horizon=4),
        (0.4, -0.3, 0.2, -0.1, 0.8, -0.7, 0.6, -0.5),
        route="full",
        tick_id=1,
        created_at=10.0,
        delta_t=2.0,
    ).state
    restored = BrainCState(
        v=first.v,
        adaptation=first.adaptation,
        filtered=first.filtered,
        weights=first.weights,
        eligibility_ring=first.eligibility_records,
        eligibility_horizon=first.eligibility_horizon,
    )
    kwargs = {
        "route": "normal",
        "tick_id": 2,
        "created_at": 13.0,
        "delta_t": 3.0,
    }
    uninterrupted = evolve_c_event(first, ZERO8, **kwargs)
    continued = evolve_c_event(restored, ZERO8, **kwargs)
    assert continued == uninterrupted


def test_ten_thousand_hostile_events_stay_finite_bounded_and_bounded_work() -> None:
    state = BrainCState.fresh(feedback_horizon=3)
    hostile = (math.nan, math.inf, -math.inf, 1e300, -1e300, 0.0, 0.25, -0.25)
    for tick in range(1, 10_001):
        delta_t = (math.nan, -1.0, math.inf, 1e300, 0.01)[tick % 5]
        candidate = evolve_c_event(
            state,
            hostile,
            route="fast" if tick % 3 == 0 else "normal",
            tick_id=tick,
            created_at=float(tick),
            delta_t=delta_t,
        )
        assert candidate.relaxations in (0, 4)
        state = candidate.state

    assert len(state.eligibility_records) == 3
    assert all(math.isfinite(value) and -2.0 <= value <= 2.0 for value in state.v)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in state.adaptation)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in state.filtered)
    assert all(math.isfinite(value) and -W_MAX <= value <= W_MAX for value in state.weights)
    assert all(
        math.isfinite(value) and 0.0 <= value <= E_MAX
        for record in state.eligibility_records
        for value in record.c_trace
    )


def test_alpha_zero_keeps_b_output_equal_while_c_advances_in_shadow() -> None:
    b_state = BrainState.fresh()
    allocation = EventAllocation(
        generation=0,
        lineage_id=b_state.lineage_id,
        tick_id=1,
        history_epoch=1,
        mutation_seq=1,
    )
    common = {
        "event_id": "shadow",
        "assessment": (0.2,) + ZERO8[1:],
        "hdc": ZERO8,
        "wound_sum": ZERO8,
        "surprise": 0.0,
        "perception_acuity": 0.5,
    }
    without_c = evolve_b(
        b_state,
        BrainEvent(proposal_c=ZERO8, **common),
        allocation=allocation,
        alpha_c=0.0,
    )
    with_c = evolve_b(
        b_state,
        BrainEvent(proposal_c=(1.0,) * 8, **common),
        allocation=allocation,
        alpha_c=0.0,
    )
    assert with_c.state == without_c.state
    assert (
        evolve_c_event(
            BrainCState.fresh(),
            (1.0,) + ZERO8[1:],
            route="normal",
            tick_id=1,
            created_at=0.0,
            delta_t=0.0,
        ).state
        != BrainCState.fresh()
    )


def test_brain_c_import_does_not_load_optional_numeric_stacks() -> None:
    source = Path("sylanne_core/compute/brain_c.py").read_text(encoding="utf-8")
    imported_roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert not ({"numpy", "torch", "cupy", "numba"} & imported_roots)


# --- Plastic readout (W_out) -------------------------------------------------


def _legacy_proposal(filtered: array[float]) -> tuple[float, ...]:
    """The pre-plastic-readout hardcoded contrast, kept verbatim so the day-0
    tests below prove bit-for-bit equality against the SHIPPED behaviour
    rather than against the new implementation's own formula."""
    return tuple(
        math.tanh(
            0.5
            * (
                filtered[4 * axis]
                + filtered[4 * axis + 1]
                - filtered[4 * axis + 2]
                - filtered[4 * axis + 3]
            )
        )
        for axis in range(N_AXES)
    )


def test_eta_out_defaults_inside_the_measured_viable_range() -> None:
    # Measured this session: 0.02-0.05 is viable; above that the shuffled-control
    # per-session pass rate balloons to 44% (noise, not signal). ETA_OUT must
    # default to the conservative end, not an invented value.
    assert 0.02 <= ETA_OUT <= 0.05
    assert ETA_OUT == 0.02


def test_default_w_out_matches_the_legacy_contrast_shape() -> None:
    assert len(DEFAULT_W_OUT) == N_BASIS * N_AXES
    for axis in range(N_AXES):
        for j in range(N_BASIS):
            expected = 0.0
            if j == 4 * axis or j == 4 * axis + 1:
                expected = 0.5
            elif j == 4 * axis + 2 or j == 4 * axis + 3:
                expected = -0.5
            assert DEFAULT_W_OUT[j * N_AXES + axis] == expected
    # Channel rows (the appraisal half of the basis) start at exactly zero.
    for j in range(N_NEURONS, N_BASIS):
        for axis in range(N_AXES):
            assert DEFAULT_W_OUT[j * N_AXES + axis] == 0.0


@pytest.mark.parametrize(
    ("appraisal", "route", "delta_t"),
    (
        (ZERO8, "fast", 0.0),
        ((1.0,) + ZERO8[1:], "normal", 0.0),
        ((-1.0, 0.5, -0.25, 0.75, -0.9, 0.1, -0.6, 0.3), "full", 2.0),
        ((0.9,) * 8, "fast", 5.0),
        ((0.3, -0.3, 0.6, -0.6, 0.1, -0.1, 0.99, -0.99), "normal", 10.0),
    ),
)
def test_plastic_readout_day0_matches_legacy_contrast_bit_for_bit(
    appraisal: tuple[float, ...],
    route: str,
    delta_t: float,
) -> None:
    """PROOF (not approx): a freshly-initialised W_out (never touched by
    feedback) reproduces the old hardcoded tanh(0.5*(f[4a]+f[4a+1]-f[4a+2]
    -f[4a+3])) contrast exactly, chained across several ticks/routes so the
    equality holds on real post-relaxation filtered values, not just zeros."""
    state = BrainCState.fresh(feedback_horizon=4)
    candidate = evolve_c_event(
        state, appraisal, route=route, tick_id=1, created_at=0.0, delta_t=delta_t
    )
    assert candidate.proposal == _legacy_proposal(candidate.state.filtered)
    assert all(-1.0 <= value <= 1.0 for value in candidate.proposal)


def test_plastic_readout_day0_matches_legacy_contrast_across_a_chained_sequence() -> None:
    scenarios = (
        (ZERO8, "fast", 0.0),
        ((1.0,) + ZERO8[1:], "normal", 0.0),
        ((-1.0, 0.5, -0.25, 0.75, -0.9, 0.1, -0.6, 0.3), "full", 2.0),
        ((0.9,) * 8, "fast", 5.0),
        ((0.3, -0.3, 0.6, -0.6, 0.1, -0.1, 0.99, -0.99), "normal", 10.0),
    )
    state = BrainCState.fresh(feedback_horizon=4)
    for tick, (appraisal, route, delta_t) in enumerate(scenarios, start=1):
        candidate = evolve_c_event(
            state, appraisal, route=route, tick_id=tick, created_at=float(tick), delta_t=delta_t
        )
        assert candidate.proposal == _legacy_proposal(candidate.state.filtered)
        state = candidate.state


def test_positive_vs_negative_feedback_move_the_proposal_in_opposite_directions() -> None:
    """The old code applied one scalar `delta` identically to every recurrent
    synapse, which cancelled in common mode under the antisymmetric readout
    (measured: +1 vs -1 feedback were 77% bit-identical on the proposal).
    With a plastic readout, +1 and -1 feedback on the SAME record must push
    the next proposal in opposite directions relative to the untouched
    baseline."""
    fresh = BrainCState.fresh(feedback_horizon=4)
    driven = evolve_c_event(
        fresh, (0.9,) + ZERO8[1:], route="full", tick_id=1, created_at=0.0, delta_t=0.0
    )
    baseline_followup = evolve_c_event(
        driven.state, ZERO8, route="fast", tick_id=2, created_at=1.0, delta_t=1.0
    ).proposal

    common = {
        "target_tick": 1,
        "state_tick": 1,
        "confidence": 1.0,
        "trusted_now": 0.0,
        "state_clock": 0.0,
        "feedback_ttl_seconds": 10.0,
    }
    plus = evolve_c_feedback(driven.state, value=1.0, **common)
    minus = evolve_c_feedback(driven.state, value=-1.0, **common)
    assert plus.status == "applied"
    assert minus.status == "applied"

    plus_followup = evolve_c_event(
        plus.state, ZERO8, route="fast", tick_id=2, created_at=1.0, delta_t=1.0
    ).proposal
    minus_followup = evolve_c_event(
        minus.state, ZERO8, route="fast", tick_id=2, created_at=1.0, delta_t=1.0
    ).proposal

    assert plus_followup[0] > baseline_followup[0] > minus_followup[0]
    assert all(-1.0 <= value <= 1.0 for value in plus_followup + minus_followup)


def test_shuffled_feedback_pairing_produces_no_directed_drift_on_the_proposal() -> None:
    """Theater-killer, asserted ON THE PROPOSAL (not on weights): 8 synthetic
    ticks each have a clean one-hot basis (as if exactly one axis fired).
    TREATED reinforces each tick's own axis (correct credit assignment).
    SHUFFLED applies the identical set of feedback calls but rotated by one
    axis, so every reward lands on an axis whose basis was exactly zero for
    that tick. Because the construction is exactly one-hot, this is not a
    statistical near-zero -- the shuffled run must leave W_out (and hence
    the proposal) EXACTLY at its pre-feedback value."""
    records = tuple(
        CEligibilityRecord(tick, float(tick), (0.0,) * N_EDGES, _one_hot_basis(4 * (tick - 1)))
        for tick in range(1, N_AXES + 1)
    )
    base_state = _state(records=records, horizon=N_AXES)
    common = {
        "state_tick": N_AXES,
        "confidence": 1.0,
        "trusted_now": float(N_AXES),
        "state_clock": float(N_AXES),
        "feedback_ttl_seconds": 10.0,
        "value": 0.0,
    }

    treated = base_state
    for tick in range(1, N_AXES + 1):
        treated = evolve_c_feedback(
            treated, target_tick=tick, values=_one_hot_axis(tick - 1), **common
        ).state

    shuffled = base_state
    for tick in range(1, N_AXES + 1):
        rotated_axis = tick % N_AXES  # deliberately mismatched vs. this tick's own basis
        shuffled = evolve_c_feedback(
            shuffled, target_tick=tick, values=_one_hot_axis(rotated_axis), **common
        ).state

    basis0 = array("f", _one_hot_basis(0))
    baseline_proposal0 = brain_c._proposal(basis0, array("f", DEFAULT_W_OUT))[0]
    treated_proposal0 = brain_c._proposal(basis0, treated.w_out)[0]
    shuffled_proposal0 = brain_c._proposal(basis0, shuffled.w_out)[0]

    assert treated_proposal0 > baseline_proposal0
    assert shuffled_proposal0 == baseline_proposal0
    # Column 0 (axis 0's own readout weights) never received reward under the
    # rotated/mismatched pairing: no tick whose basis was nonzero at a slot in
    # axis 0's block (j=0..3) ever targeted axis 0 in the shuffled run.
    assert all(shuffled.w_out[j * N_AXES + 0] == DEFAULT_W_OUT[j * N_AXES + 0] for j in range(4))


def test_feedback_learning_is_bit_reproducible_for_a_fixed_sequence() -> None:
    def _run() -> BrainCState:
        state = BrainCState.fresh(feedback_horizon=4)
        appraisals = (
            (0.8, -0.2, 0.4, -0.6, 0.1, -0.9, 0.5, -0.3),
            ZERO8,
            (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        )
        for tick, appraisal in enumerate(appraisals, start=1):
            state = evolve_c_event(
                state,
                appraisal,
                route="full",
                tick_id=tick,
                created_at=float(tick),
                delta_t=1.0,
            ).state
            state = evolve_c_feedback(
                state,
                target_tick=tick,
                state_tick=tick,
                value=0.7 if tick % 2 else -0.4,
                confidence=0.6,
                trusted_now=float(tick),
                state_clock=float(tick),
                feedback_ttl_seconds=100.0,
                values=_one_hot_axis(tick % N_AXES),
            ).state
        return state

    first = _run()
    second = _run()
    assert first == second
    assert first.weights == second.weights
    assert first.w_out == second.w_out


def test_w_out_storage_defensive_copies_constructor_validation_and_seal() -> None:
    w_out = array("f", (0.1,) * (N_BASIS * N_AXES))
    state = _state(w_out=w_out)
    w_out[0] = 1.0
    exposed = state.w_out
    exposed[0] = -1.0
    assert state.w_out.typecode == "f" and state.w_out[0] == pytest.approx(0.1)
    with pytest.raises(AttributeError):
        _ = state._BrainCState__w_out  # noqa: SLF001 - verifying the seal blocks direct access

    with pytest.raises(BrainValidationError, match="w_out.*384"):
        _state(w_out=(0.0,) * (N_BASIS * N_AXES - 1))
    with pytest.raises(BrainValidationError, match="outside"):
        _state(w_out=(W_OUT_MAX + 0.1,) + (0.0,) * (N_BASIS * N_AXES - 1))
    with pytest.raises(BrainValidationError, match="finite"):
        _state(w_out=(math.nan,) + (0.0,) * (N_BASIS * N_AXES - 1))


def test_hostile_feedback_sweep_keeps_proposal_and_w_out_bounded() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(BrainValidationError, match="finite"):
            evolve_c_feedback(
                _state(records=(CEligibilityRecord(1, 0.0, (E_MAX,) * N_EDGES, (1.0,) * N_BASIS),)),
                target_tick=1,
                state_tick=1,
                value=0.0,
                confidence=1.0,
                trusted_now=0.0,
                state_clock=0.0,
                feedback_ttl_seconds=1e9,
                values=(bad,) * N_AXES,
            )

    record = CEligibilityRecord(1, 0.0, (E_MAX,) * N_EDGES, (1.0,) * N_BASIS)
    state = _state(records=(record,), horizon=1)
    for tick in range(1, 2001):
        sign = 1e300 if tick % 2 == 0 else -1e300
        candidate = evolve_c_feedback(
            state,
            target_tick=1,
            state_tick=1,
            value=0.0,
            confidence=1.0,
            trusted_now=0.0,
            state_clock=0.0,
            feedback_ttl_seconds=1e9,
            values=(sign,) * N_AXES,
        )
        state = candidate.state

    assert all(-W_MAX <= w <= W_MAX and math.isfinite(w) for w in state.weights)
    assert all(-W_OUT_MAX <= w <= W_OUT_MAX and math.isfinite(w) for w in state.w_out)

    proposal = evolve_c_event(
        state, ZERO8, route="fast", tick_id=2, created_at=1.0, delta_t=0.0
    ).proposal
    assert all(-1.0 <= value <= 1.0 and math.isfinite(value) for value in proposal)
