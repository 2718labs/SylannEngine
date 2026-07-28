"""Immutable-by-interface records for the authoritative B state."""

from __future__ import annotations

import math
from array import array
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import Any, SupportsFloat, SupportsIndex, cast
from uuid import UUID

from .brain_errors import BrainValidationError

N_B = 8
MAX_COUNTER = 2**63 - 2
DEFAULT_LINEAGE_ID = "00000000-0000-0000-0000-000000000000"

_B_RECORD_STORAGE_NAME = "_BEligibilityRecord__b_trace"
_B_STATE_ARRAY_STORAGE_NAMES = frozenset(
    {
        "_BrainState__e",
        "_BrainState__d_plus",
        "_BrainState__d_minus",
        "_BrainState__gain_b",
        "_BrainState__theta_b",
    }
)
_B_STATE_RING_STORAGE_NAME = "_BrainState__eligibility_ring"
_B_STATE_STORAGE_NAMES = _B_STATE_ARRAY_STORAGE_NAMES | {_B_STATE_RING_STORAGE_NAME}


def _record_trace_storage(record: BEligibilityRecord) -> array[float]:
    return cast(array[float], object.__getattribute__(record, _B_RECORD_STORAGE_NAME))


def _state_array_storage(state: BrainState, name: str) -> array[float]:
    return cast(array[float], object.__getattribute__(state, name))


def _state_ring_storage(state: BrainState) -> deque[BEligibilityRecord]:
    return cast(
        deque[BEligibilityRecord],
        object.__getattribute__(state, _B_STATE_RING_STORAGE_NAME),
    )


def _validate_counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BrainValidationError(f"{name} must be a non-boolean integer")
    if not 0 <= value <= MAX_COUNTER:
        raise BrainValidationError(f"{name} is outside the persisted counter domain")
    return value


def _validate_lineage(value: object) -> str:
    if not isinstance(value, str):
        raise BrainValidationError("lineage_id must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as error:
        raise BrainValidationError("lineage_id must be a canonical UUID string") from error
    if str(parsed) != value:
        raise BrainValidationError("lineage_id must be a canonical UUID string")
    return value


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise BrainValidationError(f"{name} must be finite")
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise BrainValidationError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise BrainValidationError(f"{name} must be finite")
    return converted


def _fixed_array(
    name: str,
    values: Sequence[float] | Iterable[float],
    *,
    lower: float,
    upper: float,
) -> array[float]:
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise BrainValidationError(f"{name} must contain exactly eight values") from error
    if len(materialized) != N_B:
        raise BrainValidationError(f"{name} must contain exactly eight values")
    converted = array("d")
    for index, value in enumerate(materialized):
        item = _finite_float(f"{name}[{index}]", value)
        if not lower <= item <= upper:
            raise BrainValidationError(f"{name}[{index}] is outside [{lower}, {upper}]")
        converted.append(item)
    return converted


@dataclass(frozen=True, slots=True)
class EventAllocation:
    generation: int
    lineage_id: str
    tick_id: int
    history_epoch: int
    mutation_seq: int

    def __post_init__(self) -> None:
        _validate_counter("generation", self.generation)
        _validate_lineage(self.lineage_id)
        _validate_counter("tick_id", self.tick_id)
        _validate_counter("history_epoch", self.history_epoch)
        _validate_counter("mutation_seq", self.mutation_seq)


@dataclass(frozen=True, slots=True)
class FeedbackAllocation:
    generation: int
    lineage_id: str
    target_tick: int
    expected_mutation_seq: int
    next_mutation_seq: int

    def __post_init__(self) -> None:
        _validate_counter("generation", self.generation)
        _validate_lineage(self.lineage_id)
        _validate_counter("target_tick", self.target_tick)
        _validate_counter("expected_mutation_seq", self.expected_mutation_seq)
        _validate_counter("next_mutation_seq", self.next_mutation_seq)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BEligibilityRecord:
    tick_id: int
    created_at: float
    __b_trace: array[float]

    def __init__(
        self,
        tick_id: int,
        created_at: float,
        b_trace: Sequence[float] | Iterable[float],
    ) -> None:
        object.__setattr__(self, "tick_id", _validate_counter("tick_id", tick_id))
        object.__setattr__(self, "created_at", _finite_float("created_at", created_at))
        object.__setattr__(
            self,
            _B_RECORD_STORAGE_NAME,
            _fixed_array("b_trace", b_trace, lower=0.0, upper=1.0),
        )

    def __getattribute__(self, name: str) -> Any:
        if name == _B_RECORD_STORAGE_NAME:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @property
    def _b_trace(self) -> array[float]:
        """Expose private-looking compatibility access as a copy too."""
        return array("d", _record_trace_storage(self))

    @property
    def b_trace(self) -> array[float]:
        """Return a float64 copy; callers never receive the stored mutable array."""
        return array("d", _record_trace_storage(self))

    def copy(self) -> BEligibilityRecord:
        return BEligibilityRecord(self.tick_id, self.created_at, _record_trace_storage(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BEligibilityRecord):
            return NotImplemented
        return (
            self.tick_id == other.tick_id
            and self.created_at == other.created_at
            and _record_trace_storage(self) == _record_trace_storage(other)
        )

    def __repr__(self) -> str:
        return (
            "BEligibilityRecord("
            f"tick_id={self.tick_id}, created_at={self.created_at}, b_trace=<float64[8]>)"
        )


class BrainState:
    """A B-only state snapshot with defensive float64 and ring exposure."""

    generation: int
    lineage_id: str
    clock: float
    tick_id: int
    history_epoch: int
    mutation_seq: int
    clock_regressions: int
    _sealed: bool

    __slots__ = (
        "__d_minus",
        "__d_plus",
        "__e",
        "__eligibility_ring",
        "__gain_b",
        "__theta_b",
        "_sealed",
        "clock",
        "clock_regressions",
        "generation",
        "history_epoch",
        "lineage_id",
        "mutation_seq",
        "tick_id",
    )

    def __init__(
        self,
        *,
        generation: int,
        lineage_id: str,
        e: Sequence[float] | Iterable[float],
        d_plus: Sequence[float] | Iterable[float],
        d_minus: Sequence[float] | Iterable[float],
        gain_b: Sequence[float] | Iterable[float],
        theta_b: Sequence[float] | Iterable[float],
        clock: float,
        tick_id: int,
        history_epoch: int,
        mutation_seq: int,
        eligibility_ring: Iterable[BEligibilityRecord] = (),
        eligibility_horizon: int = 8,
        clock_regressions: int = 0,
    ) -> None:
        if (
            isinstance(eligibility_horizon, bool)
            or not isinstance(eligibility_horizon, int)
            or not 1 <= eligibility_horizon <= 32
        ):
            raise BrainValidationError("eligibility_horizon must be an integer in [1, 32]")
        try:
            records = tuple(eligibility_ring)
        except TypeError as error:
            raise BrainValidationError(
                "eligibility_ring must be an iterable of BEligibilityRecord values"
            ) from error
        if len(records) > eligibility_horizon:
            raise BrainValidationError("eligibility_ring exceeds eligibility_horizon")
        if any(not isinstance(record, BEligibilityRecord) for record in records):
            raise BrainValidationError("eligibility_ring must contain BEligibilityRecord values")

        validated_tick = _validate_counter("tick_id", tick_id)
        validated_clock = _finite_float("clock", clock)
        previous_tick = -1
        previous_created_at = -math.inf
        for record in records:
            if record.tick_id <= previous_tick or record.tick_id > validated_tick:
                raise BrainValidationError(
                    "eligibility_ring tick_ids must be ordered and not future"
                )
            if record.created_at < previous_created_at or record.created_at > validated_clock:
                raise BrainValidationError(
                    "eligibility_ring created_at values must be ordered and not future"
                )
            previous_tick = record.tick_id
            previous_created_at = record.created_at

        object.__setattr__(self, "generation", _validate_counter("generation", generation))
        object.__setattr__(self, "lineage_id", _validate_lineage(lineage_id))
        object.__setattr__(
            self,
            "_BrainState__e",
            _fixed_array("e", e, lower=-1.0, upper=1.0),
        )
        object.__setattr__(
            self,
            "_BrainState__d_plus",
            _fixed_array("d_plus", d_plus, lower=0.0, upper=32.0),
        )
        object.__setattr__(
            self,
            "_BrainState__d_minus",
            _fixed_array("d_minus", d_minus, lower=0.0, upper=32.0),
        )
        object.__setattr__(
            self,
            "_BrainState__gain_b",
            _fixed_array("gain_b", gain_b, lower=0.05, upper=1.0),
        )
        object.__setattr__(
            self,
            "_BrainState__theta_b",
            _fixed_array("theta_b", theta_b, lower=0.0, upper=0.95),
        )
        object.__setattr__(self, "clock", validated_clock)
        object.__setattr__(self, "tick_id", validated_tick)
        object.__setattr__(
            self,
            "history_epoch",
            _validate_counter("history_epoch", history_epoch),
        )
        object.__setattr__(
            self,
            "mutation_seq",
            _validate_counter("mutation_seq", mutation_seq),
        )
        object.__setattr__(
            self,
            "clock_regressions",
            _validate_counter("clock_regressions", clock_regressions),
        )
        object.__setattr__(
            self,
            "_BrainState__eligibility_ring",
            deque((record.copy() for record in records), maxlen=eligibility_horizon),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        if name in _B_STATE_STORAGE_NAMES:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @classmethod
    def fresh(
        cls,
        *,
        generation: int = 0,
        lineage_id: str = DEFAULT_LINEAGE_ID,
        clock: float = 0.0,
        feedback_horizon: int = 8,
    ) -> BrainState:
        return cls(
            generation=generation,
            lineage_id=lineage_id,
            e=(0.0,) * N_B,
            d_plus=(0.0,) * N_B,
            d_minus=(0.0,) * N_B,
            gain_b=(0.5,) * N_B,
            theta_b=(0.05,) * N_B,
            clock=clock,
            tick_id=0,
            history_epoch=0,
            mutation_seq=0,
            eligibility_horizon=feedback_horizon,
        )

    @property
    def e(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__e"))

    @property
    def _e(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__e"))

    @property
    def d_plus(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__d_plus"))

    @property
    def _d_plus(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__d_plus"))

    @property
    def d_minus(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__d_minus"))

    @property
    def _d_minus(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__d_minus"))

    @property
    def gain_b(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__gain_b"))

    @property
    def _gain_b(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__gain_b"))

    @property
    def theta_b(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__theta_b"))

    @property
    def _theta_b(self) -> array[float]:
        return array("d", _state_array_storage(self, "_BrainState__theta_b"))

    @property
    def _eligibility_ring(self) -> deque[BEligibilityRecord]:
        return deque(
            (record.copy() for record in _state_ring_storage(self)),
            maxlen=self.eligibility_horizon,
        )

    @property
    def eligibility_horizon(self) -> int:
        maxlen = _state_ring_storage(self).maxlen
        if maxlen is None:  # pragma: no cover - construction always supplies maxlen
            raise AssertionError("B eligibility ring must be bounded")
        return maxlen

    @property
    def eligibility_records(self) -> tuple[BEligibilityRecord, ...]:
        return tuple(record.copy() for record in _state_ring_storage(self))

    def copy(self) -> BrainState:
        return BrainState(
            generation=self.generation,
            lineage_id=self.lineage_id,
            e=_state_array_storage(self, "_BrainState__e"),
            d_plus=_state_array_storage(self, "_BrainState__d_plus"),
            d_minus=_state_array_storage(self, "_BrainState__d_minus"),
            gain_b=_state_array_storage(self, "_BrainState__gain_b"),
            theta_b=_state_array_storage(self, "_BrainState__theta_b"),
            clock=self.clock,
            tick_id=self.tick_id,
            history_epoch=self.history_epoch,
            mutation_seq=self.mutation_seq,
            eligibility_ring=_state_ring_storage(self),
            eligibility_horizon=self.eligibility_horizon,
            clock_regressions=self.clock_regressions,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BrainState):
            return NotImplemented
        return (
            self.generation == other.generation
            and self.lineage_id == other.lineage_id
            and _state_array_storage(self, "_BrainState__e")
            == _state_array_storage(other, "_BrainState__e")
            and _state_array_storage(self, "_BrainState__d_plus")
            == _state_array_storage(other, "_BrainState__d_plus")
            and _state_array_storage(self, "_BrainState__d_minus")
            == _state_array_storage(other, "_BrainState__d_minus")
            and _state_array_storage(self, "_BrainState__gain_b")
            == _state_array_storage(other, "_BrainState__gain_b")
            and _state_array_storage(self, "_BrainState__theta_b")
            == _state_array_storage(other, "_BrainState__theta_b")
            and self.clock == other.clock
            and self.tick_id == other.tick_id
            and self.history_epoch == other.history_epoch
            and self.mutation_seq == other.mutation_seq
            and self.clock_regressions == other.clock_regressions
            and self.eligibility_horizon == other.eligibility_horizon
            and tuple(_state_ring_storage(self)) == tuple(_state_ring_storage(other))
        )

    def __repr__(self) -> str:
        return (
            "BrainState("
            f"generation={self.generation}, lineage_id={self.lineage_id!r}, "
            f"tick_id={self.tick_id}, history_epoch={self.history_epoch}, "
            f"mutation_seq={self.mutation_seq}, clock={self.clock})"
        )
