"""Pure binary64 B operator and candidate-only in-memory coordination."""

from __future__ import annotations

import math
from array import array
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from hmac import compare_digest
from struct import Struct
from typing import Any, Literal, Protocol, SupportsFloat, SupportsIndex, cast

from .brain_errors import (
    BrainAllocationError,
    BrainCounterExhaustedError,
    BrainOwnershipError,
    BrainValidationError,
)
from .brain_state import (
    MAX_COUNTER,
    BEligibilityRecord,
    BrainState,
    EventAllocation,
    FeedbackAllocation,
)

N_B = 8
D_MAX = 32.0
D_SCALE = 4.0
ETA_D = 0.03125
ETA_B = 0.0005
ETA_THETA = 0.00025
GAIN_INIT = 0.5
THETA_INIT = 0.05
FEEDBACK_DECAY_SECONDS = 1800.0
EVENT_DT_MAX = 300.0

# Single source of truth for the C-authority ceiling: both the alpha_c mixing weight
# (config.py BrainComputeConfig.c_authority) and the residual clip bound
# (config.py BrainComputeConfig.c_residual_cap) are bounded by this same value.
# brain_backend_coordinator.MAX_PRIMARY_ALPHA imports this rather than redeclaring it.
C_RESIDUAL_CAP_MAX = 0.1

AXES = (
    "warmth",
    "arousal",
    "valence",
    "tension",
    "curiosity",
    "repair_pressure",
    "expression_drive",
    "boundary_firmness",
)

R = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
)

H_BASE_SECONDS = (5400.0, 1800.0, 3600.0, 2700.0, 2400.0, 3000.0, 1500.0, 7200.0)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _as_float(name: str, value: object, *, finite: bool = True) -> float:
    if isinstance(value, bool):
        raise BrainValidationError(f"{name} must be finite")
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise BrainValidationError(f"{name} must be finite") from error
    if finite and not math.isfinite(converted):
        raise BrainValidationError(f"{name} must be finite")
    return converted


def _materialize_eight(name: str, values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise BrainValidationError(f"{name} must contain exactly eight values") from error
    if len(materialized) != N_B:
        raise BrainValidationError(f"{name} must contain exactly eight values")
    converted: list[float] = []
    for value in materialized:
        try:
            item = float(value)
        except (TypeError, ValueError, OverflowError):
            item = math.nan
        converted.append(item)
    return tuple(converted)


def _sanitize_source(name: str, values: Sequence[float] | Iterable[float]) -> array[float]:
    result = array("d")
    for value in _materialize_eight(name, values):
        result.append(_clip(value, -1.0, 1.0) if math.isfinite(value) else 0.0)
    return result


def _strict_proposal(values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    materialized = _materialize_eight("proposal_c", values)
    if not all(math.isfinite(value) for value in materialized):
        raise BrainValidationError("proposal_c values must be finite")
    return materialized


def sanitize_surprise(value: object) -> float:
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(converted):
        return 0.0
    return _clip(converted, 0.0, 1.0)


def compose_appraisal(
    assessment: Sequence[float] | Iterable[float] | None,
    hdc: Sequence[float] | Iterable[float],
    wound_sum: Sequence[float] | Iterable[float],
) -> array[float]:
    assess = (
        array("d", (0.0,) * N_B)
        if assessment is None
        else _sanitize_source("assessment", assessment)
    )
    hdc_values = _sanitize_source("hdc", hdc)
    wound_values = _sanitize_source("wound_sum", wound_sum)
    return array(
        "d",
        (
            _clip(assess[index] + 0.25 * hdc_values[index] + 0.5 * wound_values[index], -1.0, 1.0)
            for index in range(N_B)
        ),
    )


def project_oriented(appraisal: Sequence[float] | Iterable[float]) -> array[float]:
    values = _sanitize_source("appraisal", appraisal)
    return array(
        "d",
        (
            math.fsum(coefficient * value for coefficient, value in zip(row, values, strict=True))
            for row in R
        ),
    )


def scar_view(dose: Sequence[float] | Iterable[float]) -> array[float]:
    values = _materialize_eight("dose", dose)
    if any(not math.isfinite(value) or not 0.0 <= value <= D_MAX for value in values):
        raise BrainValidationError("dose values must be finite and in [0, 32]")
    return array("d", (-math.expm1(-value / D_SCALE) for value in values))


def _normalized_transpose() -> tuple[tuple[float, ...], ...]:
    transpose = tuple(tuple(R[column][row] for column in range(N_B)) for row in range(N_B))
    norms = tuple(
        math.sqrt(math.fsum(transpose[row][column] ** 2 for row in range(N_B)))
        for column in range(N_B)
    )
    return tuple(
        tuple(
            0.0 if norms[column] == 0.0 else transpose[row][column] / norms[column]
            for column in range(N_B)
        )
        for row in range(N_B)
    )


_NORMALIZED_TRANSPOSE = _normalized_transpose()

_EVENT_CANDIDATE_STORAGE_NAMES = frozenset(
    {
        "_EventCandidate__appraisal",
        "_EventCandidate__oriented",
        "_EventCandidate__rho_plus",
        "_EventCandidate__rho_minus",
        "_EventCandidate__b_trace",
    }
)


def _event_candidate_array(candidate: EventCandidate, name: str) -> array[float]:
    return cast("array[float]", object.__getattribute__(candidate, name))


@dataclass(frozen=True, slots=True, init=False)
class BrainEvent:
    event_id: str
    assessment: tuple[float, ...] | None
    hdc: tuple[float, ...]
    wound_sum: tuple[float, ...]
    surprise: float
    perception_acuity: float
    proposal_c: tuple[float, ...]

    def __init__(
        self,
        *,
        event_id: str,
        assessment: Sequence[float] | Iterable[float] | None,
        hdc: Sequence[float] | Iterable[float],
        wound_sum: Sequence[float] | Iterable[float],
        surprise: float,
        perception_acuity: float,
        proposal_c: Sequence[float] | Iterable[float],
    ) -> None:
        if not isinstance(event_id, str) or not event_id:
            raise BrainValidationError("event_id must be a nonempty string")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "assessment",
            None if assessment is None else tuple(_sanitize_source("assessment", assessment)),
        )
        object.__setattr__(self, "hdc", tuple(_sanitize_source("hdc", hdc)))
        object.__setattr__(self, "wound_sum", tuple(_sanitize_source("wound_sum", wound_sum)))
        try:
            raw_acuity = float(perception_acuity)
        except (TypeError, ValueError, OverflowError):
            raw_acuity = math.nan
        object.__setattr__(self, "surprise", sanitize_surprise(surprise))
        object.__setattr__(self, "perception_acuity", raw_acuity)
        object.__setattr__(self, "proposal_c", _strict_proposal(proposal_c))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class EventCandidate:
    event_id: str
    state: BrainState
    allocation: EventAllocation
    salience: float
    base_generation: int
    base_lineage_id: str
    base_mutation_seq: int
    __appraisal: array[float]
    __oriented: array[float]
    __rho_plus: array[float]
    __rho_minus: array[float]
    __b_trace: array[float]

    def __init__(
        self,
        *,
        event_id: str,
        state: BrainState,
        allocation: EventAllocation,
        salience: float,
        base_generation: int,
        base_lineage_id: str,
        base_mutation_seq: int,
        appraisal: Sequence[float],
        oriented: Sequence[float],
        rho_plus: Sequence[float],
        rho_minus: Sequence[float],
        b_trace: Sequence[float],
    ) -> None:
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "salience", salience)
        object.__setattr__(self, "base_generation", base_generation)
        object.__setattr__(self, "base_lineage_id", base_lineage_id)
        object.__setattr__(self, "base_mutation_seq", base_mutation_seq)
        object.__setattr__(self, "_EventCandidate__appraisal", array("d", appraisal))
        object.__setattr__(self, "_EventCandidate__oriented", array("d", oriented))
        object.__setattr__(self, "_EventCandidate__rho_plus", array("d", rho_plus))
        object.__setattr__(self, "_EventCandidate__rho_minus", array("d", rho_minus))
        object.__setattr__(self, "_EventCandidate__b_trace", array("d", b_trace))

    def __getattribute__(self, name: str) -> Any:
        if name in _EVENT_CANDIDATE_STORAGE_NAMES:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @property
    def appraisal(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__appraisal"))

    @property
    def _appraisal(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__appraisal"))

    @property
    def oriented(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__oriented"))

    @property
    def _oriented(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__oriented"))

    @property
    def rho_plus(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__rho_plus"))

    @property
    def _rho_plus(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__rho_plus"))

    @property
    def rho_minus(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__rho_minus"))

    @property
    def _rho_minus(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__rho_minus"))

    @property
    def b_trace(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__b_trace"))

    @property
    def _b_trace(self) -> array[float]:
        return array("d", _event_candidate_array(self, "_EventCandidate__b_trace"))


FeedbackStatus = Literal["applied", "missed", "no_effect"]


@dataclass(frozen=True, slots=True)
class FeedbackCandidate:
    status: FeedbackStatus
    state: BrainState
    target_tick: int
    applied_dimensions: tuple[int, ...]
    allocation: FeedbackAllocation | None
    base_generation: int
    base_lineage_id: str
    base_mutation_seq: int


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def digest(self) -> bytes: ...


_SEAL_SIZE = 32
_SEAL_I64 = Struct("<q")
_SEAL_U64 = Struct("<Q")
_SEAL_F64 = Struct("<d")
_SEAL_F64X8 = Struct("<8d")


def _seal_int(digest: _Digest, value: object) -> None:
    if type(value) is not int:
        raise TypeError("integrity integer has the wrong type")
    digest.update(_SEAL_I64.pack(value))


def _seal_float(digest: _Digest, value: object) -> None:
    if type(value) is not float:
        raise TypeError("integrity float has the wrong type")
    digest.update(_SEAL_F64.pack(value))


def _seal_bool(digest: _Digest, value: object) -> None:
    if type(value) is not bool:
        raise TypeError("integrity boolean has the wrong type")
    digest.update(b"\x01" if value else b"\x00")


def _seal_text(digest: _Digest, value: object) -> None:
    if type(value) is not str:
        raise TypeError("integrity text has the wrong type")
    encoded = value.encode("utf-8")
    digest.update(_SEAL_U64.pack(len(encoded)))
    digest.update(encoded)


def _seal_float64x8(digest: _Digest, value: object) -> None:
    if type(value) is not array:
        raise TypeError("integrity vector has the wrong type")
    vector = cast("array[float]", value)
    if vector.typecode != "d" or len(vector) != N_B:
        raise ValueError("integrity vector must be float64[8]")
    digest.update(_SEAL_F64X8.pack(*vector))


def _seal_brain_state(digest: _Digest, value: object) -> None:
    if type(value) is not BrainState:
        raise TypeError("integrity state has the wrong type")
    state = value
    digest.update(b"S")
    _seal_int(digest, object.__getattribute__(state, "generation"))
    _seal_text(digest, object.__getattribute__(state, "lineage_id"))
    for name in (
        "_BrainState__e",
        "_BrainState__d_plus",
        "_BrainState__d_minus",
        "_BrainState__gain_b",
        "_BrainState__theta_b",
    ):
        _seal_float64x8(digest, object.__getattribute__(state, name))
    _seal_float(digest, object.__getattribute__(state, "clock"))
    _seal_int(digest, object.__getattribute__(state, "tick_id"))
    _seal_int(digest, object.__getattribute__(state, "history_epoch"))
    _seal_int(digest, object.__getattribute__(state, "mutation_seq"))
    _seal_int(digest, object.__getattribute__(state, "clock_regressions"))
    _seal_bool(digest, object.__getattribute__(state, "_sealed"))

    raw_ring = object.__getattribute__(state, "_BrainState__eligibility_ring")
    if type(raw_ring) is not deque:
        raise TypeError("integrity eligibility ring has the wrong type")
    ring = cast(deque[object], raw_ring)
    _seal_int(digest, ring.maxlen)
    _seal_int(digest, len(ring))
    for raw_record in ring:
        if type(raw_record) is not BEligibilityRecord:
            raise TypeError("integrity eligibility record has the wrong type")
        record = raw_record
        _seal_int(digest, object.__getattribute__(record, "tick_id"))
        _seal_float(digest, object.__getattribute__(record, "created_at"))
        _seal_float64x8(
            digest,
            object.__getattribute__(record, "_BEligibilityRecord__b_trace"),
        )


def _seal_event_allocation(digest: _Digest, value: object) -> None:
    if type(value) is not EventAllocation:
        raise TypeError("integrity event allocation has the wrong type")
    allocation = value
    digest.update(b"E")
    _seal_int(digest, object.__getattribute__(allocation, "generation"))
    _seal_text(digest, object.__getattribute__(allocation, "lineage_id"))
    _seal_int(digest, object.__getattribute__(allocation, "tick_id"))
    _seal_int(digest, object.__getattribute__(allocation, "history_epoch"))
    _seal_int(digest, object.__getattribute__(allocation, "mutation_seq"))


def _seal_feedback_allocation(digest: _Digest, value: object) -> None:
    if value is None:
        digest.update(b"\x00")
        return
    if type(value) is not FeedbackAllocation:
        raise TypeError("integrity feedback allocation has the wrong type")
    allocation = value
    digest.update(b"\x01")
    _seal_int(digest, object.__getattribute__(allocation, "generation"))
    _seal_text(digest, object.__getattribute__(allocation, "lineage_id"))
    _seal_int(digest, object.__getattribute__(allocation, "target_tick"))
    _seal_int(digest, object.__getattribute__(allocation, "expected_mutation_seq"))
    _seal_int(digest, object.__getattribute__(allocation, "next_mutation_seq"))


def _seal_base_fields(digest: _Digest, candidate: object) -> None:
    _seal_int(digest, object.__getattribute__(candidate, "base_generation"))
    _seal_text(digest, object.__getattribute__(candidate, "base_lineage_id"))
    _seal_int(digest, object.__getattribute__(candidate, "base_mutation_seq"))


def _seal_event_candidate(digest: _Digest, candidate: EventCandidate) -> None:
    digest.update(b"CANDIDATE-EVENT-V1")
    _seal_text(digest, object.__getattribute__(candidate, "event_id"))
    _seal_brain_state(digest, object.__getattribute__(candidate, "state"))
    _seal_event_allocation(digest, object.__getattribute__(candidate, "allocation"))
    _seal_float(digest, object.__getattribute__(candidate, "salience"))
    _seal_base_fields(digest, candidate)
    for name in (
        "_EventCandidate__appraisal",
        "_EventCandidate__oriented",
        "_EventCandidate__rho_plus",
        "_EventCandidate__rho_minus",
        "_EventCandidate__b_trace",
    ):
        _seal_float64x8(digest, object.__getattribute__(candidate, name))


def _seal_feedback_candidate(digest: _Digest, candidate: FeedbackCandidate) -> None:
    digest.update(b"CANDIDATE-FEEDBACK-V1")
    _seal_text(digest, object.__getattribute__(candidate, "status"))
    _seal_brain_state(digest, object.__getattribute__(candidate, "state"))
    _seal_int(digest, object.__getattribute__(candidate, "target_tick"))
    raw_dimensions = object.__getattribute__(candidate, "applied_dimensions")
    if type(raw_dimensions) is not tuple:
        raise TypeError("integrity applied dimensions have the wrong type")
    dimensions = cast(tuple[object, ...], raw_dimensions)
    _seal_int(digest, len(dimensions))
    for dimension in dimensions:
        _seal_int(digest, dimension)
    _seal_feedback_allocation(digest, object.__getattribute__(candidate, "allocation"))
    _seal_base_fields(digest, candidate)


def _candidate_integrity_seal(candidate: EventCandidate | FeedbackCandidate) -> bytes:
    try:
        digest: _Digest = blake2b(digest_size=_SEAL_SIZE, person=b"SylannCand.v1")
        if type(candidate) is EventCandidate:
            _seal_event_candidate(digest, candidate)
        elif type(candidate) is FeedbackCandidate:
            _seal_feedback_candidate(digest, candidate)
        else:
            raise TypeError("integrity candidate has the wrong type")
        seal = digest.digest()
        if type(seal) is not bytes or len(seal) != _SEAL_SIZE:
            raise ValueError("integrity seal has the wrong size")
        return seal
    except Exception as error:
        raise BrainValidationError("candidate integrity seal could not be computed") from error


def _integrity_mismatch_message(candidate: EventCandidate | FeedbackCandidate) -> str:
    if isinstance(candidate, EventCandidate):
        return "event transition candidate integrity seal does not match issuance"
    status = object.__getattribute__(candidate, "status")
    if status in ("missed", "no_effect"):
        return "receipt-only feedback transition candidate integrity seal does not match issuance"
    return "feedback transition candidate integrity seal does not match issuance"


def _check_event_allocation(state: BrainState, allocation: EventAllocation) -> None:
    if not isinstance(allocation, EventAllocation):
        raise BrainValidationError("allocation must be an EventAllocation")
    for field in ("tick_id", "history_epoch", "mutation_seq"):
        if getattr(state, field) == MAX_COUNTER:
            raise BrainCounterExhaustedError(f"{field} is exhausted")
    if allocation.generation != state.generation:
        raise BrainAllocationError("event allocation generation does not match state")
    if allocation.lineage_id != state.lineage_id:
        raise BrainAllocationError("event allocation lineage does not match state")
    expected = (
        state.tick_id + 1,
        state.history_epoch + 1,
        state.mutation_seq + 1,
    )
    actual = (allocation.tick_id, allocation.history_epoch, allocation.mutation_seq)
    if actual != expected:
        raise BrainAllocationError("event allocation counters are not exact successors")


def _clock_result(state: BrainState, trusted_now: float | None) -> tuple[float, bool]:
    if trusted_now is None:
        return state.clock, False
    try:
        proposed = float(trusted_now)
    except (TypeError, ValueError, OverflowError):
        return state.clock, True
    if not math.isfinite(proposed) or proposed < state.clock:
        return state.clock, True
    return proposed, False


def _next_regression_count(state: BrainState, regressed: bool) -> int:
    if not regressed:
        return state.clock_regressions
    if state.clock_regressions == MAX_COUNTER:
        raise BrainCounterExhaustedError("clock_regressions is exhausted")
    return state.clock_regressions + 1


def _residual(alpha_c: float, proposal: float, cap: float) -> float:
    if alpha_c == 0.0 or proposal == 0.0:
        return 0.0
    threshold = cap / alpha_c
    if proposal >= threshold:
        return cap
    if proposal <= -threshold:
        return -cap
    return _clip(alpha_c * proposal, -cap, cap)


def evolve_b(
    state: BrainState,
    event: BrainEvent,
    *,
    allocation: EventAllocation,
    trusted_now: float | None = None,
    alpha_c: float = 0.0,
    c_residual_cap: float = C_RESIDUAL_CAP_MAX,
) -> EventCandidate:
    _check_event_allocation(state, allocation)
    alpha = _as_float("alpha_c", alpha_c)
    if not 0.0 <= alpha <= C_RESIDUAL_CAP_MAX:
        raise BrainValidationError("alpha_c must be in [0, 0.1]")
    cap = _as_float("c_residual_cap", c_residual_cap)
    if not 0.0 <= cap <= C_RESIDUAL_CAP_MAX:
        raise BrainValidationError("c_residual_cap must be in [0, 0.1]")

    appraisal = compose_appraisal(event.assessment, event.hdc, event.wound_sum)
    oriented = project_oriented(appraisal)
    salience = _clip(
        0.25 + 0.5 * abs(appraisal[1]) + 0.25 * sanitize_surprise(event.surprise),
        0.0,
        1.0,
    )
    old_d_plus = state.d_plus
    old_d_minus = state.d_minus
    old_theta = state.theta_b
    rho_plus = array("d")
    rho_minus = array("d")
    d_plus = array("d")
    d_minus = array("d")
    for index in range(N_B):
        q_plus = max(0.0, oriented[index] - old_theta[index]) * salience
        q_minus = max(0.0, -oriented[index] - old_theta[index]) * salience
        plus_fraction = -math.expm1(-ETA_D * q_plus)
        minus_fraction = -math.expm1(-ETA_D * q_minus)
        rho_plus.append(plus_fraction)
        rho_minus.append(minus_fraction)
        plus_value = old_d_plus[index] + (D_MAX - old_d_plus[index]) * plus_fraction
        minus_value = old_d_minus[index] + (D_MAX - old_d_minus[index]) * minus_fraction
        d_plus.append(_clip(plus_value, old_d_plus[index], D_MAX))
        d_minus.append(_clip(minus_value, old_d_minus[index], D_MAX))

    signed_scar = array("d", (math.tanh((d_plus[i] - d_minus[i]) / D_SCALE) for i in range(N_B)))
    scar_mass = array("d", (-math.expm1(-(d_plus[i] + d_minus[i]) / D_SCALE) for i in range(N_B)))
    bias = array(
        "d",
        (
            _clip(
                math.fsum(
                    _NORMALIZED_TRANSPOSE[row][column] * signed_scar[column]
                    for column in range(N_B)
                ),
                -1.0,
                1.0,
            )
            for row in range(N_B)
        ),
    )

    effective_now, regressed = _clock_result(state, trusted_now)
    elapsed = effective_now - state.clock
    delta_t = min(EVENT_DT_MAX, max(0.0, elapsed))
    acuity = event.perception_acuity if math.isfinite(event.perception_acuity) else 0.5
    perception = _clip(acuity, 0.0, 1.0)
    old_e = state.e
    gain = state.gain_b
    e_new = array("d")
    # Residual-free visible affect: dose/b_trace decontamination requires that no
    # quantity derived from proposal_c (the C residual) can reach the eligibility
    # ring or the scar dose rate. e_clean tracks what e_new would be at alpha=0.
    e_clean = array("d")
    for index in range(N_B):
        trait_gain = _clip(2.0 * perception, 0.3, 2.0) if index == 3 else 1.0
        half_life = H_BASE_SECONDS[index] * trait_gain * (1.0 + 2.0 * scar_mass[index])
        tau = half_life / math.log(2.0)
        retention = math.exp(-delta_t / tau)
        visible = (
            bias[index] + retention * (old_e[index] - bias[index]) + gain[index] * appraisal[index]
        )
        residual = _residual(alpha, event.proposal_c[index], cap)
        e_clean.append(_clip(visible, -1.0, 1.0))
        e_new.append(_clip(visible + residual, -1.0, 1.0))

    b_trace = array(
        "d",
        (
            _clip(
                abs(e_clean[index] - old_e[index]) + rho_plus[index] + rho_minus[index],
                0.0,
                1.0,
            )
            for index in range(N_B)
        ),
    )
    records: deque[BEligibilityRecord] = deque(
        state.eligibility_records, maxlen=state.eligibility_horizon
    )
    records.append(BEligibilityRecord(allocation.tick_id, effective_now, b_trace))
    next_state = BrainState(
        generation=allocation.generation,
        lineage_id=allocation.lineage_id,
        e=e_new,
        d_plus=d_plus,
        d_minus=d_minus,
        gain_b=gain,
        theta_b=old_theta,
        clock=effective_now,
        tick_id=allocation.tick_id,
        history_epoch=allocation.history_epoch,
        mutation_seq=allocation.mutation_seq,
        eligibility_ring=records,
        eligibility_horizon=state.eligibility_horizon,
        clock_regressions=_next_regression_count(state, regressed),
    )
    return EventCandidate(
        event_id=event.event_id,
        state=next_state,
        allocation=allocation,
        salience=salience,
        base_generation=state.generation,
        base_lineage_id=state.lineage_id,
        base_mutation_seq=state.mutation_seq,
        appraisal=appraisal,
        oriented=oriented,
        rho_plus=rho_plus,
        rho_minus=rho_minus,
        b_trace=b_trace,
    )


def _feedback_candidate(
    *,
    status: FeedbackStatus,
    state: BrainState,
    target_tick: int,
    applied_dimensions: tuple[int, ...] = (),
    allocation: FeedbackAllocation | None = None,
    base: BrainState,
) -> FeedbackCandidate:
    # Receipt copies isolate the committed snapshot; applied state is already newly constructed.
    candidate_state = state.copy() if status != "applied" else state
    return FeedbackCandidate(
        status=status,
        state=candidate_state,
        target_tick=target_tick,
        applied_dimensions=applied_dimensions,
        allocation=allocation,
        base_generation=base.generation,
        base_lineage_id=base.lineage_id,
        base_mutation_seq=base.mutation_seq,
    )


def _check_feedback_allocation(
    state: BrainState,
    target_tick: int,
    allocation: FeedbackAllocation,
) -> None:
    if not isinstance(allocation, FeedbackAllocation):
        raise BrainValidationError("allocation must be a FeedbackAllocation")
    if state.mutation_seq == MAX_COUNTER:
        raise BrainCounterExhaustedError("mutation_seq is exhausted")
    if allocation.generation != state.generation:
        raise BrainAllocationError("feedback allocation generation does not match state")
    if allocation.lineage_id != state.lineage_id:
        raise BrainAllocationError("feedback allocation lineage does not match state")
    if allocation.target_tick != target_tick:
        raise BrainAllocationError("feedback allocation target does not match target record")
    if allocation.expected_mutation_seq != state.mutation_seq:
        raise BrainAllocationError("feedback allocation expected mutation does not match state")
    if allocation.next_mutation_seq != allocation.expected_mutation_seq + 1:
        raise BrainAllocationError("feedback allocation next mutation is not the exact successor")


def evolve_feedback(
    state: BrainState,
    *,
    target_tick: int,
    value: float,
    confidence: float,
    trusted_now: float,
    feedback_ttl_seconds: float,
    allocation: FeedbackAllocation | None,
) -> FeedbackCandidate:
    if (
        isinstance(target_tick, bool)
        or not isinstance(target_tick, int)
        or not 0 <= target_tick <= MAX_COUNTER
    ):
        raise BrainValidationError("target_tick must be a non-boolean persisted counter")
    if target_tick > state.tick_id:
        raise BrainValidationError("future target_tick is not eligible for feedback")
    feedback_value = _clip(_as_float("value", value), -1.0, 1.0)
    feedback_confidence = _as_float("confidence", confidence)
    if not 0.0 <= feedback_confidence <= 1.0:
        raise BrainValidationError("confidence must be in [0, 1]")
    ttl = _as_float("feedback_ttl_seconds", feedback_ttl_seconds)
    if ttl <= 0.0:
        raise BrainValidationError("feedback_ttl_seconds must be positive")
    if feedback_value == 0.0 or feedback_confidence == 0.0:
        if allocation is not None:
            raise BrainAllocationError("no_effect feedback must not receive an allocation")
        return _feedback_candidate(
            status="no_effect", state=state, target_tick=target_tick, base=state
        )

    effective_now, regressed = _clock_result(state, trusted_now)
    record = next(
        (item for item in state.eligibility_records if item.tick_id == target_tick),
        None,
    )
    if record is None:
        if allocation is not None:
            raise BrainAllocationError("missed feedback must not receive an allocation")
        return _feedback_candidate(
            status="missed", state=state, target_tick=target_tick, base=state
        )
    age = max(0.0, effective_now - record.created_at)
    if age > ttl:
        if allocation is not None:
            raise BrainAllocationError("expired feedback must not receive an allocation")
        return _feedback_candidate(
            status="missed", state=state, target_tick=target_tick, base=state
        )
    if allocation is None:
        raise BrainAllocationError("eligible feedback requires a store allocation")
    _check_feedback_allocation(state, target_tick, allocation)

    decay = math.exp(-age / FEEDBACK_DECAY_SECONDS)
    delta = feedback_value * feedback_confidence
    trace = record.b_trace
    old_gain = state.gain_b
    old_theta = state.theta_b
    gain = array("d", old_gain)
    theta = array("d", old_theta)
    applied: list[int] = []
    for index in range(N_B):
        decayed_trace = trace[index] * decay
        gain[index] = _clip(old_gain[index] + ETA_B * delta * decayed_trace, 0.05, 1.0)
        theta[index] = _clip(
            old_theta[index] + ETA_THETA * abs(delta) * decayed_trace,
            0.0,
            0.95,
        )
        if decayed_trace > 0.0 and (
            gain[index] != old_gain[index] or theta[index] != old_theta[index]
        ):
            applied.append(index)
    if not applied:
        return _feedback_candidate(
            status="no_effect",
            state=state,
            target_tick=target_tick,
            base=state,
        )

    regression_count = _next_regression_count(state, regressed)
    next_state = BrainState(
        generation=state.generation,
        lineage_id=state.lineage_id,
        e=state.e,
        d_plus=state.d_plus,
        d_minus=state.d_minus,
        gain_b=gain,
        theta_b=theta,
        clock=state.clock,
        tick_id=state.tick_id,
        history_epoch=state.history_epoch,
        mutation_seq=allocation.next_mutation_seq,
        eligibility_ring=state.eligibility_records,
        eligibility_horizon=state.eligibility_horizon,
        clock_regressions=regression_count,
    )
    return _feedback_candidate(
        status="applied",
        state=next_state,
        target_tick=target_tick,
        applied_dimensions=tuple(applied),
        allocation=allocation,
        base=state,
    )


def _validate_event_transition(state: BrainState, candidate: EventCandidate) -> None:
    if not isinstance(candidate.state, BrainState):
        raise BrainValidationError("event transition state must be a BrainState")
    _check_event_allocation(state, candidate.allocation)
    allocation = candidate.allocation
    state_counters = (
        candidate.state.generation,
        candidate.state.lineage_id,
        candidate.state.tick_id,
        candidate.state.history_epoch,
        candidate.state.mutation_seq,
    )
    allocation_counters = (
        allocation.generation,
        allocation.lineage_id,
        allocation.tick_id,
        allocation.history_epoch,
        allocation.mutation_seq,
    )
    if state_counters != allocation_counters:
        raise BrainValidationError("event transition counters do not match allocation")
    if not math.isfinite(candidate.state.clock) or candidate.state.clock < state.clock:
        raise BrainValidationError("event transition clock must not regress")
    if any(new < old for new, old in zip(candidate.state.d_plus, state.d_plus, strict=True)) or any(
        new < old for new, old in zip(candidate.state.d_minus, state.d_minus, strict=True)
    ):
        raise BrainValidationError("event transition must not decrease dose")
    if candidate.state.gain_b != state.gain_b or candidate.state.theta_b != state.theta_b:
        raise BrainValidationError("event transition must not change gain or theta")


def _validate_feedback_transition(state: BrainState, candidate: FeedbackCandidate) -> None:
    if not isinstance(candidate.state, BrainState):
        raise BrainValidationError("feedback transition state must be a BrainState")
    if candidate.status != "applied":
        if candidate.status not in ("missed", "no_effect"):
            raise BrainValidationError("receipt-only feedback status is invalid")
        if candidate.allocation is not None or candidate.state != state:
            raise BrainValidationError(
                "receipt-only feedback transition must keep exact state and no allocation"
            )
        return

    if not isinstance(candidate.allocation, FeedbackAllocation):
        raise BrainValidationError("applied feedback transition requires a FeedbackAllocation")
    _check_feedback_allocation(state, candidate.target_tick, candidate.allocation)
    if (
        candidate.state.eligibility_horizon != state.eligibility_horizon
        or candidate.state.eligibility_records != state.eligibility_records
    ):
        raise BrainValidationError("feedback transition must not change eligibility history")
    unchanged = (
        candidate.state.generation == state.generation
        and candidate.state.lineage_id == state.lineage_id
        and candidate.state.e == state.e
        and candidate.state.d_plus == state.d_plus
        and candidate.state.d_minus == state.d_minus
        and candidate.state.tick_id == state.tick_id
        and candidate.state.history_epoch == state.history_epoch
        and candidate.state.clock == state.clock
    )
    if not unchanged or candidate.state.mutation_seq != candidate.allocation.next_mutation_seq:
        raise BrainValidationError("feedback transition violates the applied mutation contract")


class BrainComputeCore:
    """Own one committed snapshot; prepare methods only produce candidates."""

    __slots__ = ("_issued_candidate", "_issued_seal", "_state")

    def __init__(self, state: BrainState) -> None:
        if not isinstance(state, BrainState):
            raise BrainValidationError("state must be a BrainState")
        self._state = state.copy()
        self._issued_candidate: EventCandidate | FeedbackCandidate | None = None
        self._issued_seal: bytes | None = None

    @classmethod
    def fresh(
        cls,
        *,
        generation: int = 0,
        lineage_id: str = "00000000-0000-0000-0000-000000000000",
        clock: float = 0.0,
        feedback_horizon: int = 8,
    ) -> BrainComputeCore:
        return cls(
            BrainState.fresh(
                generation=generation,
                lineage_id=lineage_id,
                clock=clock,
                feedback_horizon=feedback_horizon,
            )
        )

    @property
    def state(self) -> BrainState:
        return self._state.copy()

    def prepare_event(
        self,
        event: BrainEvent,
        *,
        allocation: EventAllocation,
        trusted_now: float | None = None,
        alpha_c: float = 0.0,
        c_residual_cap: float = C_RESIDUAL_CAP_MAX,
    ) -> EventCandidate:
        if not isinstance(event, BrainEvent):
            raise BrainValidationError("event must be a BrainEvent")
        candidate = evolve_b(
            self._state,
            event,
            allocation=allocation,
            trusted_now=trusted_now,
            alpha_c=alpha_c,
            c_residual_cap=c_residual_cap,
        )
        seal = _candidate_integrity_seal(candidate)
        self._issued_candidate = candidate
        self._issued_seal = seal
        return candidate

    def prepare_feedback(
        self,
        *,
        target_tick: int,
        value: float,
        confidence: float,
        trusted_now: float,
        feedback_ttl_seconds: float,
        allocation: FeedbackAllocation | None,
    ) -> FeedbackCandidate:
        candidate = evolve_feedback(
            self._state,
            target_tick=target_tick,
            value=value,
            confidence=confidence,
            trusted_now=trusted_now,
            feedback_ttl_seconds=feedback_ttl_seconds,
            allocation=allocation,
        )
        seal = _candidate_integrity_seal(candidate)
        self._issued_candidate = candidate
        self._issued_seal = seal
        return candidate

    def commit(self, candidate: EventCandidate | FeedbackCandidate) -> BrainState:
        if not isinstance(candidate, (EventCandidate, FeedbackCandidate)):
            raise BrainValidationError("candidate must be prepared by BrainComputeCore")
        if candidate is not self._issued_candidate:
            raise BrainOwnershipError("candidate was not issued by this BrainComputeCore")
        issued_seal = self._issued_seal
        self._issued_candidate = None
        self._issued_seal = None
        if issued_seal is None:
            raise BrainOwnershipError("candidate integrity seal is unavailable")
        actual_seal = _candidate_integrity_seal(candidate)
        integrity_matches = compare_digest(actual_seal, issued_seal)
        current_key = (
            self._state.generation,
            self._state.lineage_id,
            self._state.mutation_seq,
        )
        candidate_key = (
            candidate.base_generation,
            candidate.base_lineage_id,
            candidate.base_mutation_seq,
        )
        if current_key != candidate_key:
            raise BrainAllocationError("candidate was prepared from a stale committed state")
        if isinstance(candidate, EventCandidate):
            _validate_event_transition(self._state, candidate)
        else:
            _validate_feedback_transition(self._state, candidate)
        if not integrity_matches:
            raise BrainValidationError(_integrity_mismatch_message(candidate))
        if isinstance(candidate, EventCandidate) or candidate.status == "applied":
            self._state = candidate.state
        return self.state
