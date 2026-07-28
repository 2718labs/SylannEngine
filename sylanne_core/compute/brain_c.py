"""Deterministic zero-dependency Brain C dynamics and local feedback learning."""

from __future__ import annotations

import hashlib
import math
import struct
from array import array
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError, dataclass
from itertools import islice
from typing import Any, Literal, SupportsFloat, SupportsIndex, TypeVar, cast

from .brain_errors import BrainValidationError
from .brain_state import MAX_COUNTER

MODEL_SEED = 42
N_NEURONS = 32
N_CHANNELS = 16
N_EDGES = 128
N_AXES = 8
N_BASIS = N_NEURONS + N_CHANNELS  # plastic-readout input: filtered[32] ++ channels[16]
RELAXATION_STEPS = 4

TAU_M_SECONDS = 10.0
TAU_A_SECONDS = 60.0
TAU_F_SECONDS = 5.0
FEEDBACK_DECAY_SECONDS = 1800.0

E_MAX = 8.0
W_MAX = 1.0
ETA_C = 0.0005

# --- Plastic readout (W_out) and its third-factor learning rate --------------
# W_OUT_MAX bounds the readout weights the same way W_MAX bounds the recurrent
# weights (same order of magnitude, shipped values sit at +-0.5). ETA_OUT is
# NOT invented: a prototype (plastic readout + per-axis 8-vector third factor)
# measured this session under REALISTIC feedback (25% of events rated, 3-event
# delay, 20% wrong sign, 50-200 events/session) a treated effect of +0.056 at
# eta=0.02/200 vs a SHUFFLED control of +0.0000 -- real, sign-correct,
# control-clean learning, 2.8x the 0.02 promotion bar. The measured viable
# range is 0.02-0.05; above that the shuffled-control per-session pass rate
# balloons to 44% (per-session noise, not signal). ETA_OUT defaults to the
# conservative end of that range (0.02). Both constants still require a full
# calibration sweep before any nonzero alpha_c ships -- this default only
# guarantees day-0 (pre-feedback) bit-identical behaviour, not a tuned gain.
ETA_OUT = 0.02
W_OUT_MAX = 1.0

# Excitatory input coupling per neuron parity (even neurons more excitable than
# odd). Over RELAXATION_STEPS=4 a constant drive `current` drives the leaky
# membrane to ~0.684*current, so with a base threshold of 1.0 a neuron whose
# dominant input channel is near its maximum (|appraisal| -> 1.0) can bootstrap
# a spike from the fresh() constructor, while weak/zero appraisal stays
# sub-threshold. Values chosen so even neurons cross on channel >= ~0.585 and
# odd on channel >= ~0.73; the adaptation term self-limits the firing rate.
INPUT_GAIN_EVEN = 2.5
INPUT_GAIN_ODD = 2.0

_OFFSETS = (1, 5, 13, 17)
_C_RECORD_STORAGE_NAME = "_CEligibilityRecord__c_trace"
_C_RECORD_BASIS_STORAGE_NAME = "_CEligibilityRecord__basis_trace"
_C_STATE_ARRAY_STORAGE_NAMES = frozenset(
    {
        "_BrainCState__v",
        "_BrainCState__adaptation",
        "_BrainCState__filtered",
        "_BrainCState__weights",
        "_BrainCState__w_out",
    }
)
_C_STATE_RING_STORAGE_NAME = "_BrainCState__eligibility_ring"
_C_STATE_STORAGE_NAMES = _C_STATE_ARRAY_STORAGE_NAMES | {_C_STATE_RING_STORAGE_NAME}
_T = TypeVar("_T")


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _float32(value: float) -> float:
    return array("f", (value,))[0]


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise BrainValidationError(f"{name} must be finite")
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise BrainValidationError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise BrainValidationError(f"{name} must be finite")
    return converted


def _counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BrainValidationError(f"{name} must be a non-boolean integer")
    if not 0 <= value <= MAX_COUNTER:
        raise BrainValidationError(f"{name} is outside the persisted counter domain")
    return value


def _bounded_materialize(
    values: Iterable[_T],
    *,
    limit: int,
    error_message: str,
) -> tuple[_T, ...]:
    try:
        return tuple(islice(values, limit + 1))
    except TypeError as error:
        raise BrainValidationError(error_message) from error


def _bounded_float32_array(
    name: str,
    values: Sequence[float] | Iterable[float],
    *,
    length: int,
    lower: float,
    upper: float,
) -> array[float]:
    error_message = f"{name} must contain exactly {length} values"
    materialized = _bounded_materialize(values, limit=length, error_message=error_message)
    if len(materialized) != length:
        raise BrainValidationError(error_message)
    result = array("f")
    for index, raw in enumerate(materialized):
        value = _strict_float(f"{name}[{index}]", raw)
        if not lower <= value <= upper:
            raise BrainValidationError(f"{name}[{index}] is outside [{lower}, {upper}]")
        result.append(value)
    return result


def _validate_owned_float32_array(
    name: str,
    values: object,
    *,
    length: int,
    lower: float,
    upper: float,
) -> array[float]:
    if type(values) is not array:
        raise BrainValidationError(f"{name} must be an owned float32[{length}] array")
    validated = cast(array[float], values)
    if validated.typecode != "f" or len(validated) != length:
        raise BrainValidationError(f"{name} must be an owned float32[{length}] array")
    for index, value in enumerate(validated):
        if not math.isfinite(value):
            raise BrainValidationError(f"{name}[{index}] must be finite")
        if not lower <= value <= upper:
            raise BrainValidationError(f"{name}[{index}] is outside [{lower}, {upper}]")
    return validated


@dataclass(frozen=True, slots=True)
class BrainCTopology:
    edges: tuple[tuple[int, int], ...]
    incoming: tuple[tuple[int, ...], ...]
    initial_weights: tuple[float, ...]
    digest: str


def build_topology(seed: int = MODEL_SEED) -> BrainCTopology:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BrainValidationError("model seed must be a non-boolean integer")
    incoming = tuple(
        tuple((post + offset) % N_NEURONS for offset in _OFFSETS) for post in range(N_NEURONS)
    )
    edges = tuple((pre, post) for post, presynaptic in enumerate(incoming) for pre in presynaptic)
    encoded = b"".join(struct.pack(">BB", pre, post) for pre, post in edges)
    weights = tuple(((17 * pre + 31 * post + seed) % 2001 - 1000) / 5000.0 for pre, post in edges)
    return BrainCTopology(
        edges=edges,
        incoming=incoming,
        initial_weights=weights,
        digest=hashlib.sha256(encoded).hexdigest(),
    )


TOPOLOGY = build_topology()
TOPOLOGY_DIGEST = TOPOLOGY.digest


def _default_w_out() -> tuple[float, ...]:
    """W_out[j][a] initialised to reproduce the shipped hardcoded contrast:

    proposal[a] = tanh(0.5*(f[4a] + f[4a+1] - f[4a+2] - f[4a+3]))

    Row j (a basis index) maps to axis a = j // 4 only; every other axis
    column of that row is 0. Rows 32..47 (the appraisal-channel half of the
    basis) start at 0 -- channels only start mattering once feedback teaches
    the readout to use them. Flat layout is row-major: index = j * N_AXES + a.
    """
    flat = [0.0] * (N_BASIS * N_AXES)
    for axis in range(N_AXES):
        for offset, sign in ((0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)):
            row = 4 * axis + offset
            flat[row * N_AXES + axis] = 0.5 * sign
    return tuple(flat)


DEFAULT_W_OUT = _default_w_out()


def split_signed_input(values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    error_message = "appraisal must contain exactly eight values"
    materialized = _bounded_materialize(values, limit=N_AXES, error_message=error_message)
    if len(materialized) != N_AXES:
        raise BrainValidationError(error_message)
    channels: list[float] = []
    for raw in materialized:
        try:
            value = float(cast(str | SupportsFloat | SupportsIndex, raw))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        value = _clip(value, -1.0, 1.0)
        channels.extend((max(value, 0.0), max(-value, 0.0)))
    return tuple(channels)


def _record_trace_storage(record: CEligibilityRecord) -> array[float]:
    return cast(array[float], object.__getattribute__(record, _C_RECORD_STORAGE_NAME))


def _record_basis_storage(record: CEligibilityRecord) -> array[float]:
    return cast(array[float], object.__getattribute__(record, _C_RECORD_BASIS_STORAGE_NAME))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CEligibilityRecord:
    tick_id: int
    created_at: float
    __c_trace: array[float]
    __basis_trace: array[float]

    def __init__(
        self,
        tick_id: int,
        created_at: float,
        c_trace: Sequence[float] | Iterable[float],
        basis_trace: Sequence[float] | Iterable[float] = (0.0,) * N_BASIS,
    ) -> None:
        object.__setattr__(self, "tick_id", _counter("tick_id", tick_id))
        object.__setattr__(self, "created_at", _strict_float("created_at", created_at))
        object.__setattr__(
            self,
            _C_RECORD_STORAGE_NAME,
            _bounded_float32_array(
                "c_trace",
                c_trace,
                length=N_EDGES,
                lower=0.0,
                upper=E_MAX,
            ),
        )
        object.__setattr__(
            self,
            _C_RECORD_BASIS_STORAGE_NAME,
            _bounded_float32_array(
                "basis_trace",
                basis_trace,
                length=N_BASIS,
                lower=0.0,
                upper=1.0,
            ),
        )

    @classmethod
    def _from_owned(
        cls,
        tick_id: int,
        created_at: float,
        c_trace: object,
        basis_trace: object,
    ) -> CEligibilityRecord:
        validated_tick = _counter("tick_id", tick_id)
        validated_created_at = _strict_float("created_at", created_at)
        validated_trace = _validate_owned_float32_array(
            "c_trace",
            c_trace,
            length=N_EDGES,
            lower=0.0,
            upper=E_MAX,
        )
        validated_basis = _validate_owned_float32_array(
            "basis_trace",
            basis_trace,
            length=N_BASIS,
            lower=0.0,
            upper=1.0,
        )
        record = cls.__new__(cls)
        object.__setattr__(record, "tick_id", validated_tick)
        object.__setattr__(record, "created_at", validated_created_at)
        object.__setattr__(record, _C_RECORD_STORAGE_NAME, validated_trace)
        object.__setattr__(record, _C_RECORD_BASIS_STORAGE_NAME, validated_basis)
        return record

    def __getattribute__(self, name: str) -> Any:
        if name in (_C_RECORD_STORAGE_NAME, _C_RECORD_BASIS_STORAGE_NAME):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @property
    def c_trace(self) -> array[float]:
        return array("f", _record_trace_storage(self))

    @property
    def _c_trace(self) -> array[float]:
        return array("f", _record_trace_storage(self))

    @property
    def basis_trace(self) -> array[float]:
        return array("f", _record_basis_storage(self))

    def copy(self) -> CEligibilityRecord:
        return CEligibilityRecord(
            self.tick_id,
            self.created_at,
            _record_trace_storage(self),
            _record_basis_storage(self),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CEligibilityRecord):
            return NotImplemented
        return (
            self.tick_id == other.tick_id
            and self.created_at == other.created_at
            and _record_trace_storage(self) == _record_trace_storage(other)
            and _record_basis_storage(self) == _record_basis_storage(other)
        )

    def __repr__(self) -> str:
        return (
            "CEligibilityRecord("
            f"tick_id={self.tick_id}, created_at={self.created_at}, "
            "c_trace=<float32[128]>, basis_trace=<float32[48]>)"
        )


def _eligibility_horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32:
        raise BrainValidationError("eligibility_horizon must be an integer in [1, 32]")
    return value


def _validate_ring_records(records: Iterable[object]) -> None:
    previous_tick = -1
    previous_created_at = -math.inf
    for item in records:
        if not isinstance(item, CEligibilityRecord):
            raise BrainValidationError("eligibility_ring must contain CEligibilityRecord values")
        if item.tick_id <= previous_tick or item.created_at < previous_created_at:
            raise BrainValidationError("eligibility_ring records must be ordered")
        _validate_owned_float32_array(
            "c_trace",
            _record_trace_storage(item),
            length=N_EDGES,
            lower=0.0,
            upper=E_MAX,
        )
        _validate_owned_float32_array(
            "basis_trace",
            _record_basis_storage(item),
            length=N_BASIS,
            lower=0.0,
            upper=1.0,
        )
        previous_tick = item.tick_id
        previous_created_at = item.created_at


def _validate_owned_ring(value: object, *, horizon: int) -> deque[CEligibilityRecord]:
    if type(value) is not deque:
        raise BrainValidationError("eligibility_ring must be an owned bounded deque")
    untyped_ring = cast(deque[object], value)
    if untyped_ring.maxlen != horizon or len(untyped_ring) > horizon:
        raise BrainValidationError("eligibility_ring exceeds eligibility_horizon")
    _validate_ring_records(untyped_ring)
    return cast(deque[CEligibilityRecord], untyped_ring)


def _state_array_storage(state: BrainCState, name: str) -> array[float]:
    return cast(array[float], object.__getattribute__(state, name))


def _state_ring_storage(state: BrainCState) -> deque[CEligibilityRecord]:
    return cast(
        deque[CEligibilityRecord],
        object.__getattribute__(state, _C_STATE_RING_STORAGE_NAME),
    )


class BrainCState:
    """Immutable-by-interface float32 dynamics plus a bounded local trace ring."""

    _sealed: bool

    __slots__ = (
        "__adaptation",
        "__eligibility_ring",
        "__filtered",
        "__v",
        "__weights",
        "__w_out",
        "_sealed",
    )

    def __init__(
        self,
        *,
        v: Sequence[float] | Iterable[float],
        adaptation: Sequence[float] | Iterable[float],
        filtered: Sequence[float] | Iterable[float],
        weights: Sequence[float] | Iterable[float],
        w_out: Sequence[float] | Iterable[float] = DEFAULT_W_OUT,
        eligibility_ring: Iterable[CEligibilityRecord] = (),
        eligibility_horizon: int = 8,
    ) -> None:
        validated_horizon = _eligibility_horizon(eligibility_horizon)
        records = _bounded_materialize(
            eligibility_ring,
            limit=validated_horizon,
            error_message="eligibility_ring must be an iterable of CEligibilityRecord values",
        )
        if len(records) > validated_horizon:
            raise BrainValidationError("eligibility_ring exceeds eligibility_horizon")
        _validate_ring_records(records)

        object.__setattr__(
            self,
            "_BrainCState__v",
            _bounded_float32_array("v", v, length=N_NEURONS, lower=-2.0, upper=2.0),
        )
        object.__setattr__(
            self,
            "_BrainCState__adaptation",
            _bounded_float32_array(
                "adaptation", adaptation, length=N_NEURONS, lower=0.0, upper=1.0
            ),
        )
        object.__setattr__(
            self,
            "_BrainCState__filtered",
            _bounded_float32_array("filtered", filtered, length=N_NEURONS, lower=0.0, upper=1.0),
        )
        object.__setattr__(
            self,
            "_BrainCState__weights",
            _bounded_float32_array("weights", weights, length=N_EDGES, lower=-W_MAX, upper=W_MAX),
        )
        object.__setattr__(
            self,
            "_BrainCState__w_out",
            _bounded_float32_array(
                "w_out",
                w_out,
                length=N_BASIS * N_AXES,
                lower=-W_OUT_MAX,
                upper=W_OUT_MAX,
            ),
        )
        object.__setattr__(
            self,
            _C_STATE_RING_STORAGE_NAME,
            deque((record.copy() for record in records), maxlen=validated_horizon),
        )
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def _from_owned(
        cls,
        *,
        v: object,
        adaptation: object,
        filtered: object,
        weights: object,
        w_out: object,
        eligibility_ring: object,
        eligibility_horizon: int,
    ) -> BrainCState:
        validated_horizon = _eligibility_horizon(eligibility_horizon)
        validated_v = _validate_owned_float32_array("v", v, length=N_NEURONS, lower=-2.0, upper=2.0)
        validated_adaptation = _validate_owned_float32_array(
            "adaptation",
            adaptation,
            length=N_NEURONS,
            lower=0.0,
            upper=1.0,
        )
        validated_filtered = _validate_owned_float32_array(
            "filtered",
            filtered,
            length=N_NEURONS,
            lower=0.0,
            upper=1.0,
        )
        validated_weights = _validate_owned_float32_array(
            "weights", weights, length=N_EDGES, lower=-W_MAX, upper=W_MAX
        )
        validated_w_out = _validate_owned_float32_array(
            "w_out", w_out, length=N_BASIS * N_AXES, lower=-W_OUT_MAX, upper=W_OUT_MAX
        )
        validated_ring = _validate_owned_ring(eligibility_ring, horizon=validated_horizon)

        state = cls.__new__(cls)
        object.__setattr__(state, "_BrainCState__v", validated_v)
        object.__setattr__(state, "_BrainCState__adaptation", validated_adaptation)
        object.__setattr__(state, "_BrainCState__filtered", validated_filtered)
        object.__setattr__(state, "_BrainCState__weights", validated_weights)
        object.__setattr__(state, "_BrainCState__w_out", validated_w_out)
        object.__setattr__(state, _C_STATE_RING_STORAGE_NAME, validated_ring)
        object.__setattr__(state, "_sealed", True)
        return state

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        if name in _C_STATE_STORAGE_NAMES:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    @classmethod
    def fresh(cls, *, feedback_horizon: int = 8) -> BrainCState:
        return cls(
            v=(0.0,) * N_NEURONS,
            adaptation=(0.0,) * N_NEURONS,
            filtered=(0.0,) * N_NEURONS,
            weights=TOPOLOGY.initial_weights,
            w_out=DEFAULT_W_OUT,
            eligibility_horizon=feedback_horizon,
        )

    @property
    def v(self) -> array[float]:
        return array("f", _state_array_storage(self, "_BrainCState__v"))

    @property
    def adaptation(self) -> array[float]:
        return array("f", _state_array_storage(self, "_BrainCState__adaptation"))

    @property
    def filtered(self) -> array[float]:
        return array("f", _state_array_storage(self, "_BrainCState__filtered"))

    @property
    def weights(self) -> array[float]:
        return array("f", _state_array_storage(self, "_BrainCState__weights"))

    @property
    def w_out(self) -> array[float]:
        return array("f", _state_array_storage(self, "_BrainCState__w_out"))

    @property
    def eligibility_horizon(self) -> int:
        maxlen = _state_ring_storage(self).maxlen
        if maxlen is None:  # pragma: no cover - construction always provides maxlen
            raise AssertionError("C eligibility ring must be bounded")
        return maxlen

    @property
    def eligibility_records(self) -> tuple[CEligibilityRecord, ...]:
        return tuple(record.copy() for record in _state_ring_storage(self))

    def copy(self) -> BrainCState:
        return BrainCState(
            v=_state_array_storage(self, "_BrainCState__v"),
            adaptation=_state_array_storage(self, "_BrainCState__adaptation"),
            filtered=_state_array_storage(self, "_BrainCState__filtered"),
            weights=_state_array_storage(self, "_BrainCState__weights"),
            w_out=_state_array_storage(self, "_BrainCState__w_out"),
            eligibility_ring=_state_ring_storage(self),
            eligibility_horizon=self.eligibility_horizon,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BrainCState):
            return NotImplemented
        return (
            _state_array_storage(self, "_BrainCState__v")
            == _state_array_storage(other, "_BrainCState__v")
            and _state_array_storage(self, "_BrainCState__adaptation")
            == _state_array_storage(other, "_BrainCState__adaptation")
            and _state_array_storage(self, "_BrainCState__filtered")
            == _state_array_storage(other, "_BrainCState__filtered")
            and _state_array_storage(self, "_BrainCState__weights")
            == _state_array_storage(other, "_BrainCState__weights")
            and _state_array_storage(self, "_BrainCState__w_out")
            == _state_array_storage(other, "_BrainCState__w_out")
            and tuple(_state_ring_storage(self)) == tuple(_state_ring_storage(other))
            and self.eligibility_horizon == other.eligibility_horizon
        )

    def __repr__(self) -> str:
        return (
            "BrainCState(v=<float32[32]>, adaptation=<float32[32]>, "
            "filtered=<float32[32]>, weights=<float32[128]>, w_out=<float32[384]>, "
            f"eligibility_records={len(_state_ring_storage(self))})"
        )


Route = Literal["fast", "normal", "full"]


@dataclass(frozen=True, slots=True)
class BrainCEventCandidate:
    state: BrainCState
    tick_id: int
    route: Route
    relaxations: int
    proposal: tuple[float, ...]

    @property
    def c_trace(self) -> array[float]:
        records = _state_ring_storage(self.state)
        if not records:  # pragma: no cover - event construction always appends one
            raise AssertionError("event candidate must contain its local eligibility")
        return array("f", _record_trace_storage(records[-1]))


FeedbackStatus = Literal["applied", "missed", "no_effect"]


@dataclass(frozen=True, slots=True)
class BrainCFeedbackCandidate:
    state: BrainCState
    target_tick: int
    status: FeedbackStatus
    applied_synapses: tuple[int, ...]


def _event_delta(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if math.isnan(converted) or converted < 0.0:
        return 0.0
    return converted


def _decay(values: array[float], delta_t: float, tau: float, lower: float) -> array[float]:
    factor = 0.0 if math.isinf(delta_t) else math.exp(-delta_t / tau)
    return array(
        "f", (_clip(value * factor, lower, 2.0 if lower < 0.0 else 1.0) for value in values)
    )


def _basis(filtered: array[float], channels: Sequence[float]) -> array[float]:
    """basis = filtered[32] ++ split_signed_input(appraisal)[16], fixed order."""
    result = array("f", filtered)
    result.extend(channels)
    return result


def _proposal(basis: array[float], w_out: array[float]) -> tuple[float, ...]:
    """Plastic readout: proposal[a] = tanh(sum_j W_out[j][a] * basis[j]).

    The accumulation walks j = 0..N_BASIS-1 with a plain running `+=`
    (NOT math.fsum) so that, under DEFAULT_W_OUT -- where every row has at
    most one nonzero entry, at +-0.5 -- this reduces to EXACTLY the same
    sequence of roundings as the legacy hardcoded contrast
    tanh(0.5*(f[4a]+f[4a+1]-f[4a+2]-f[4a+3])): scaling by the exact
    power-of-two 0.5 commutes with round-to-nearest, so accumulating the
    pre-halved terms in the same left-to-right order reproduces the legacy
    bit pattern precisely (proven by test_plastic_readout_day0_...).
    """
    proposal: list[float] = []
    for axis in range(N_AXES):
        total = 0.0
        for j in range(N_BASIS):
            total += w_out[j * N_AXES + axis] * basis[j]
        proposal.append(math.tanh(total))
    return tuple(proposal)


def evolve_c_event(
    state: BrainCState,
    appraisal: Sequence[float] | Iterable[float],
    *,
    route: Route,
    tick_id: int,
    created_at: float,
    delta_t: float,
) -> BrainCEventCandidate:
    if not isinstance(state, BrainCState):
        raise BrainValidationError("state must be a BrainCState")
    if route not in ("fast", "normal", "full"):
        raise BrainValidationError("route must be fast, normal, or full")
    validated_tick = _counter("tick_id", tick_id)
    validated_created_at = _strict_float("created_at", created_at)
    current_records = _state_ring_storage(state)
    if current_records:
        latest = current_records[-1]
        if validated_tick <= latest.tick_id:
            raise BrainValidationError("tick_id must increase beyond the latest eligibility record")
        if validated_created_at < latest.created_at:
            raise BrainValidationError("created_at must not precede the latest eligibility record")
    channels = split_signed_input(appraisal)
    delta = _event_delta(delta_t)

    v = _decay(
        _state_array_storage(state, "_BrainCState__v"),
        delta,
        TAU_M_SECONDS,
        -2.0,
    )
    adaptation = _decay(
        _state_array_storage(state, "_BrainCState__adaptation"),
        delta,
        TAU_A_SECONDS,
        0.0,
    )
    filtered = _decay(
        _state_array_storage(state, "_BrainCState__filtered"),
        delta,
        TAU_F_SECONDS,
        0.0,
    )
    weights = _state_array_storage(state, "_BrainCState__weights")
    local_trace = array("f", (0.0,) * N_EDGES)
    relaxations = 0 if route == "fast" else RELAXATION_STEPS

    for _ in range(relaxations):
        pre_reset: list[float] = []
        pseudo: list[float] = []
        spikes: list[int] = []
        for post in range(N_NEURONS):
            axis = post // 4
            channel = 2 * axis + (0 if post % 4 < 2 else 1)
            input_gain = INPUT_GAIN_EVEN if post % 2 == 0 else INPUT_GAIN_ODD
            recurrent = math.fsum(
                weights[post * 4 + offset] * filtered[pre]
                for offset, pre in enumerate(TOPOLOGY.incoming[post])
            )
            current = input_gain * channels[channel] + recurrent - 0.25 * adaptation[post]
            bounded_v = _clip(v[post] + 0.25 * (-v[post] + current), -2.0, 2.0)
            threshold = _clip(1.0 + 0.25 * adaptation[post], 0.5, 1.5)
            pre_reset.append(bounded_v)
            pseudo.append(0.25 * max(0.0, 1.0 - abs(bounded_v - threshold) / 0.5))
            spikes.append(1 if bounded_v >= threshold else 0)

        next_v = array(
            "f",
            (0.0 if spikes[post] else pre_reset[post] for post in range(N_NEURONS)),
        )
        next_adaptation = array(
            "f",
            (
                _clip(0.95 * adaptation[post] + 0.1 * spikes[post], 0.0, 1.0)
                for post in range(N_NEURONS)
            ),
        )
        next_filtered = array(
            "f",
            (
                _clip(0.8 * filtered[post] + 0.2 * spikes[post], 0.0, 1.0)
                for post in range(N_NEURONS)
            ),
        )
        next_trace = array(
            "f",
            (
                _clip(
                    0.95 * local_trace[index] + spikes[pre] * pseudo[post],
                    0.0,
                    E_MAX,
                )
                for index, (pre, post) in enumerate(TOPOLOGY.edges)
            ),
        )
        v = next_v
        adaptation = next_adaptation
        filtered = next_filtered
        local_trace = next_trace

    w_out = _state_array_storage(state, "_BrainCState__w_out")
    basis = _basis(filtered, channels)

    records = deque(current_records, maxlen=state.eligibility_horizon)
    records.append(
        CEligibilityRecord._from_owned(validated_tick, validated_created_at, local_trace, basis)
    )
    next_state = BrainCState._from_owned(
        v=v,
        adaptation=adaptation,
        filtered=filtered,
        weights=weights,
        w_out=w_out,
        eligibility_ring=records,
        eligibility_horizon=state.eligibility_horizon,
    )
    return BrainCEventCandidate(
        state=next_state,
        tick_id=validated_tick,
        route=route,
        relaxations=relaxations,
        proposal=_proposal(basis, w_out),
    )


def _effective_now(trusted_now: object, state_clock: float) -> float:
    if isinstance(trusted_now, bool):
        return state_clock
    try:
        converted = float(cast(str | SupportsFloat | SupportsIndex, trusted_now))
    except (TypeError, ValueError, OverflowError):
        return state_clock
    if not math.isfinite(converted) or converted < state_clock:
        return state_clock
    return converted


def _feedback_axis_values(
    values: Sequence[float] | Iterable[float] | None,
    fallback: float,
) -> tuple[float, ...]:
    """8 per-axis signed factors. None broadcasts the scalar `value` (already
    clipped to [-1, 1]) to every axis, so every existing scalar-only caller
    keeps working unchanged."""
    if values is None:
        return (fallback,) * N_AXES
    error_message = "values must contain exactly eight values"
    materialized = _bounded_materialize(values, limit=N_AXES, error_message=error_message)
    if len(materialized) != N_AXES:
        raise BrainValidationError(error_message)
    return tuple(
        _clip(_strict_float(f"values[{index}]", raw), -1.0, 1.0)
        for index, raw in enumerate(materialized)
    )


def evolve_c_feedback(
    state: BrainCState,
    *,
    target_tick: int,
    state_tick: int,
    value: float,
    confidence: float,
    trusted_now: float,
    state_clock: float,
    feedback_ttl_seconds: float,
    values: Sequence[float] | Iterable[float] | None = None,
) -> BrainCFeedbackCandidate:
    if not isinstance(state, BrainCState):
        raise BrainValidationError("state must be a BrainCState")
    target = _counter("target_tick", target_tick)
    current_tick = _counter("state_tick", state_tick)
    if target > current_tick:
        raise BrainValidationError("target_tick must not exceed state_tick")
    signal = _clip(_strict_float("value", value), -1.0, 1.0)
    certainty = _strict_float("confidence", confidence)
    validated_clock = _strict_float("state_clock", state_clock)
    ttl = _strict_float("feedback_ttl_seconds", feedback_ttl_seconds)
    if not 0.0 <= certainty <= 1.0:
        raise BrainValidationError("confidence must be in [0, 1]")
    if ttl <= 0.0:
        raise BrainValidationError("feedback_ttl_seconds must be positive")
    axis_values = _feedback_axis_values(values, signal)

    record = next(
        (item for item in _state_ring_storage(state) if item.tick_id == target),
        None,
    )
    if record is None:
        return BrainCFeedbackCandidate(state, target, "missed", ())
    if record.created_at > validated_clock:
        raise BrainValidationError("target record created_at must not exceed state_clock")
    age = max(0.0, _effective_now(trusted_now, validated_clock) - record.created_at)
    if age > ttl:
        return BrainCFeedbackCandidate(state, target, "missed", ())

    decay = math.exp(-age / FEEDBACK_DECAY_SECONDS)
    old_weights = _state_array_storage(state, "_BrainCState__weights")
    old_w_out = _state_array_storage(state, "_BrainCState__w_out")
    trace = _record_trace_storage(record)
    basis_trace = _record_basis_storage(record)

    # PER-NEURON RECURRENT CREDIT (the missing e-prop term). L_j projects the
    # per-axis feedback through the readout weights onto post-neuron j (only
    # rows 0..N_NEURONS-1 of W_out correspond to a neuron; the channel rows
    # never drive a neuron, so axes j >= N_NEURONS are not summed here). This
    # is what makes dw sign-DIVERSE per post-neuron -- the old code applied
    # one scalar `delta` identically to every synapse, which cancelled in
    # common mode under the antisymmetric readout (measured: +1 vs -1
    # feedback were 77% bit-identical on the proposal).
    l_factor = array("f", (0.0,) * N_NEURONS)
    for post in range(N_NEURONS):
        total = 0.0
        for axis in range(N_AXES):
            total += old_w_out[post * N_AXES + axis] * axis_values[axis]
        l_factor[post] = _float32(total)

    next_weights = array("f", old_weights)
    changed_synapses: list[int] = []
    for index, (_pre, post) in enumerate(TOPOLOGY.edges):
        dw = ETA_C * certainty * l_factor[post] * trace[index] * decay
        projected = _float32(_clip(old_weights[index] + dw, -W_MAX, W_MAX))
        if projected != old_weights[index]:
            next_weights[index] = projected
            changed_synapses.append(index)

    # READOUT LEARNING. dW_out[j][a] = ETA_OUT * values[a] * confidence *
    # basis_trace[j] * decay -- the retained basis snapshot from the SAME
    # eligibility record targeted above, so credit lands on exactly the
    # basis units that drove that tick's proposal.
    next_w_out = array("f", old_w_out)
    w_out_changed = False
    for j in range(N_BASIS):
        basis_value = basis_trace[j]
        for axis in range(N_AXES):
            index = j * N_AXES + axis
            dw_out = ETA_OUT * axis_values[axis] * certainty * basis_value * decay
            projected = _float32(_clip(old_w_out[index] + dw_out, -W_OUT_MAX, W_OUT_MAX))
            if projected != old_w_out[index]:
                next_w_out[index] = projected
                w_out_changed = True

    if not changed_synapses and not w_out_changed:
        return BrainCFeedbackCandidate(state, target, "no_effect", ())

    next_state = BrainCState._from_owned(
        v=_state_array_storage(state, "_BrainCState__v"),
        adaptation=_state_array_storage(state, "_BrainCState__adaptation"),
        filtered=_state_array_storage(state, "_BrainCState__filtered"),
        weights=next_weights if changed_synapses else old_weights,
        w_out=next_w_out if w_out_changed else old_w_out,
        eligibility_ring=_state_ring_storage(state),
        eligibility_horizon=state.eligibility_horizon,
    )
    return BrainCFeedbackCandidate(next_state, target, "applied", tuple(changed_synapses))
