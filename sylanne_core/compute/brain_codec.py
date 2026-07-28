"""Deterministic, non-executable codec for authoritative public B/C state."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

from .brain_c import (
    DEFAULT_W_OUT,
    N_AXES,
    N_BASIS,
    TOPOLOGY_DIGEST,
    BrainCState,
    CEligibilityRecord,
)
from .brain_errors import BrainValidationError
from .brain_state import BEligibilityRecord, BrainState

# Schema 1 is the legacy (pre-plastic-readout) format: C has no "w_out" field
# and C eligibility records have no "basis" field. Schema 2 added those fields
# in a bare JSON blob. Schema 3 retains the schema-2 document fields but makes
# the compressed envelope mandatory for new writes, so an older store rejects
# a rewritten row by version before trying to parse compressed bytes as JSON.
BRAIN_STATE_SCHEMA_VERSION = 3
SUPPORTED_BRAIN_STATE_SCHEMA_VERSIONS = frozenset({1, 2, 3})
# New writes use a compressed envelope and remain inside the original public
# 32KiB contract. Bare schema-1/2 JSON remains readable below 64KiB solely for
# one-way migration from the brief plastic-readout format.
MAX_STATE_BLOB_BYTES = 32 * 1024
MAX_STATE_PAYLOAD_BYTES = 64 * 1024
_STATE_BLOB_MAGIC = b"SYLBCZ1\x00"
_STATE_BLOB_MAGIC_PREFIX = b"SYLBCZ"

_ROOT_FIELDS = frozenset({"schema", "topology", "b", "c"})
_B_FIELDS = frozenset(
    {
        "generation",
        "lineage",
        "e",
        "dp",
        "dm",
        "gain",
        "theta",
        "clock",
        "tick",
        "history",
        "mutation",
        "regressions",
        "horizon",
        "elig",
    }
)
_C_FIELDS_V1 = frozenset({"v", "adaptation", "filtered", "w", "horizon", "elig"})
_C_FIELDS_V2 = _C_FIELDS_V1 | {"w_out"}
_ELIGIBILITY_FIELDS = frozenset({"tick", "created_at", "trace"})
_C_ELIGIBILITY_FIELDS_V2 = _ELIGIBILITY_FIELDS | {"basis"}
_T = TypeVar("_T")


def state_sha256(blob: bytes) -> bytes:
    if not isinstance(blob, bytes):
        raise BrainValidationError("state blob must be bytes")
    return hashlib.sha256(blob).digest()


@dataclass(frozen=True, slots=True)
class BrainBundle:
    b: BrainState
    c: BrainCState

    def __post_init__(self) -> None:
        if not isinstance(self.b, BrainState) or not isinstance(self.c, BrainCState):
            raise BrainValidationError("BrainBundle requires BrainState and BrainCState")
        if self.b.eligibility_horizon != self.c.eligibility_horizon:
            raise BrainValidationError("B and C eligibility horizon must match")
        for record in self.c.eligibility_records:
            if record.tick_id > self.b.tick_id or record.created_at > self.b.clock:
                raise BrainValidationError("C eligibility tick/time must not exceed B")

    def copy(self) -> BrainBundle:
        return BrainBundle(self.b.copy(), self.c.copy())


@dataclass(frozen=True, slots=True)
class BrainEncodingMetrics:
    """Sizes of the exact candidate encoding used by the production writer."""

    payload_bytes: int
    blob_bytes: int


def _encode_array(values: object, *, length: int, code: str) -> str:
    try:
        materialized = tuple(cast(Any, values))
    except TypeError as error:
        raise BrainValidationError("state array is not iterable") from error
    if len(materialized) != length:
        raise BrainValidationError(f"state array length must be {length}")
    try:
        packed = struct.pack(f">{length}{code}", *materialized)
    except (struct.error, OverflowError) as error:
        raise BrainValidationError("state array cannot be encoded") from error
    return base64.b64encode(packed).decode("ascii")


def _b_document(state: BrainState) -> dict[str, object]:
    return {
        "generation": state.generation,
        "lineage": state.lineage_id,
        "e": _encode_array(state.e, length=8, code="d"),
        "dp": _encode_array(state.d_plus, length=8, code="d"),
        "dm": _encode_array(state.d_minus, length=8, code="d"),
        "gain": _encode_array(state.gain_b, length=8, code="d"),
        "theta": _encode_array(state.theta_b, length=8, code="d"),
        "clock": state.clock,
        "tick": state.tick_id,
        "history": state.history_epoch,
        "mutation": state.mutation_seq,
        "regressions": state.clock_regressions,
        "horizon": state.eligibility_horizon,
        "elig": [
            {
                "tick": record.tick_id,
                "created_at": record.created_at,
                "trace": _encode_array(record.b_trace, length=8, code="d"),
            }
            for record in state.eligibility_records
        ],
    }


def _c_document(state: BrainCState) -> dict[str, object]:
    return {
        "v": _encode_array(state.v, length=32, code="f"),
        "adaptation": _encode_array(state.adaptation, length=32, code="f"),
        "filtered": _encode_array(state.filtered, length=32, code="f"),
        "w": _encode_array(state.weights, length=128, code="f"),
        "w_out": _encode_array(state.w_out, length=N_BASIS * N_AXES, code="f"),
        "horizon": state.eligibility_horizon,
        "elig": [
            {
                "tick": record.tick_id,
                "created_at": record.created_at,
                "trace": _encode_array(record.c_trace, length=128, code="f"),
                "basis": _encode_array(record.basis_trace, length=N_BASIS, code="f"),
            }
            for record in state.eligibility_records
        ],
    }


def _encode_brain_bundle_parts(bundle: BrainBundle) -> tuple[bytes, bytes]:
    if not isinstance(bundle, BrainBundle):
        raise BrainValidationError("bundle must be a BrainBundle")
    validated = bundle.copy()
    document = {
        "schema": BRAIN_STATE_SCHEMA_VERSION,
        "topology": TOPOLOGY_DIGEST,
        "b": _b_document(validated.b),
        "c": _c_document(validated.c),
    }
    try:
        payload = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise BrainValidationError("bundle contains a nonfinite or non-JSON value") from error
    try:
        blob = _STATE_BLOB_MAGIC + zlib.compress(payload, level=9)
    except zlib.error as error:  # pragma: no cover - valid bytes are compressible
        raise BrainValidationError("state payload cannot be compressed") from error
    return payload, blob


def inspect_brain_bundle_encoding(bundle: BrainBundle) -> BrainEncodingMetrics:
    """Measure the production candidate without turning a budget miss into an exception."""
    payload, blob = _encode_brain_bundle_parts(bundle)
    return BrainEncodingMetrics(payload_bytes=len(payload), blob_bytes=len(blob))


def encode_brain_bundle(bundle: BrainBundle) -> tuple[bytes, bytes]:
    payload, blob = _encode_brain_bundle_parts(bundle)
    if len(payload) >= MAX_STATE_PAYLOAD_BYTES:
        raise BrainValidationError("state payload exceeds the public size limit")
    if len(blob) >= MAX_STATE_BLOB_BYTES:
        raise BrainValidationError("state blob exceeds the public size limit")
    return blob, state_sha256(blob)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BrainValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(name: str, value: object, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BrainValidationError(f"{name} must be a JSON object")
    document = cast(dict[str, object], value)
    actual = frozenset(document)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise BrainValidationError(f"{name} has missing fields: {sorted(missing)}")
    if unknown:
        raise BrainValidationError(f"{name} has unknown fields: {sorted(unknown)}")
    return document


def _list(name: str, value: object, *, maximum: int) -> list[object]:
    if not isinstance(value, list):
        raise BrainValidationError(f"{name} must be a JSON array")
    values = cast(list[object], value)
    if len(values) > maximum:
        raise BrainValidationError(f"{name} exceeds its horizon")
    return values


def _json_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BrainValidationError(f"{name} must be a finite JSON number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise BrainValidationError(f"{name} must be a finite JSON number") from error
    if not math.isfinite(converted):
        raise BrainValidationError(f"{name} must be a finite JSON number")
    return converted


def _decode_array(name: str, value: object, *, length: int, code: str) -> tuple[float, ...]:
    if not isinstance(value, str):
        raise BrainValidationError(f"{name} must be base64 text")
    byte_length = length * struct.calcsize(code)
    encoded_length = 4 * math.ceil(byte_length / 3)
    if len(value) != encoded_length:
        raise BrainValidationError(f"{name} base64 length is invalid")
    try:
        encoded = value.encode("ascii")
        raw = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise BrainValidationError(f"{name} is invalid base64") from error
    if len(raw) != byte_length:
        raise BrainValidationError(f"{name} decoded length is invalid")
    try:
        return tuple(struct.unpack(f">{length}{code}", raw))
    except struct.error as error:  # pragma: no cover - exact length already checked
        raise BrainValidationError(f"{name} decoded length is invalid") from error


def _eligibility_records(
    name: str,
    value: object,
    *,
    horizon: int,
    trace_length: int,
    trace_code: str,
    factory: Callable[[int, float, tuple[float, ...]], _T],
) -> tuple[_T, ...]:
    records: list[_T] = []
    for index, raw in enumerate(_list(name, value, maximum=horizon)):
        document = _object(f"{name}[{index}]", raw, _ELIGIBILITY_FIELDS)
        trace = _decode_array(
            f"{name}[{index}].trace",
            document["trace"],
            length=trace_length,
            code=trace_code,
        )
        created_at = _json_number(f"{name}[{index}].created_at", document["created_at"])
        records.append(factory(cast(int, document["tick"]), created_at, trace))
    return tuple(records)


def _c_eligibility_records(
    name: str,
    value: object,
    *,
    horizon: int,
    schema: int,
) -> tuple[CEligibilityRecord, ...]:
    """Schema 1 has no "basis" field; those records decode with basis_trace
    defaulted to zeros (CEligibilityRecord's own legacy-compatible default),
    i.e. no retroactive readout credit for pre-upgrade ticks."""
    fields = _C_ELIGIBILITY_FIELDS_V2 if schema >= 2 else _ELIGIBILITY_FIELDS
    records: list[CEligibilityRecord] = []
    for index, raw in enumerate(_list(name, value, maximum=horizon)):
        document = _object(f"{name}[{index}]", raw, fields)
        trace = _decode_array(
            f"{name}[{index}].trace",
            document["trace"],
            length=128,
            code="f",
        )
        created_at = _json_number(f"{name}[{index}].created_at", document["created_at"])
        tick = cast(int, document["tick"])
        if schema >= 2:
            basis = _decode_array(
                f"{name}[{index}].basis",
                document["basis"],
                length=N_BASIS,
                code="f",
            )
            records.append(CEligibilityRecord(tick, created_at, trace, basis))
        else:
            records.append(CEligibilityRecord(tick, created_at, trace))
    return tuple(records)


def _decode_state_payload(blob: bytes) -> bytes:
    if blob.startswith(_STATE_BLOB_MAGIC_PREFIX) and not blob.startswith(_STATE_BLOB_MAGIC):
        raise BrainValidationError("state compressed envelope is unsupported")
    if not blob.startswith(_STATE_BLOB_MAGIC):
        return blob

    compressed = blob[len(_STATE_BLOB_MAGIC) :]
    try:
        decompressor = zlib.decompressobj()
        payload = decompressor.decompress(
            compressed,
            max_length=MAX_STATE_PAYLOAD_BYTES,
        )
    except zlib.error as error:
        raise BrainValidationError("state blob has invalid compressed data") from error

    if decompressor.unconsumed_tail or len(payload) >= MAX_STATE_PAYLOAD_BYTES:
        raise BrainValidationError("decompressed state payload exceeds the size limit")
    if not decompressor.eof:
        raise BrainValidationError("state blob has truncated compressed data")
    if decompressor.unused_data:
        raise BrainValidationError("state blob has trailing compressed data")

    try:
        payload += decompressor.flush(MAX_STATE_PAYLOAD_BYTES - len(payload))
    except zlib.error as error:  # pragma: no cover - stream integrity checked above
        raise BrainValidationError("state blob has invalid compressed data") from error
    if len(payload) >= MAX_STATE_PAYLOAD_BYTES:
        raise BrainValidationError("decompressed state payload exceeds the size limit")
    return payload


def decode_brain_bundle(
    blob: bytes,
    expected_sha256: bytes,
    *,
    expected_schema: int | None = None,
) -> BrainBundle:
    if not isinstance(blob, bytes):
        raise BrainValidationError("state blob must be bytes")
    if expected_schema is not None and (
        type(expected_schema) is not int
        or expected_schema not in SUPPORTED_BRAIN_STATE_SCHEMA_VERSIONS
    ):
        raise BrainValidationError("expected state schema is unsupported")
    size_limit = (
        MAX_STATE_BLOB_BYTES
        if blob.startswith(_STATE_BLOB_MAGIC_PREFIX)
        else MAX_STATE_PAYLOAD_BYTES
    )
    if len(blob) >= size_limit:
        raise BrainValidationError("state blob size exceeds the public limit")
    if not isinstance(expected_sha256, bytes) or len(expected_sha256) != 32:
        raise BrainValidationError("state checksum must be raw 32 bytes")
    actual_sha256 = state_sha256(blob)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise BrainValidationError("state checksum mismatch")
    payload = _decode_state_payload(blob)
    try:
        root_raw = json.loads(payload, object_pairs_hook=_unique_object)
    except BrainValidationError:
        raise
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise BrainValidationError("state blob is invalid JSON") from error
    root = _object("root", root_raw, _ROOT_FIELDS)
    if (
        type(root["schema"]) is not int
        or root["schema"] not in SUPPORTED_BRAIN_STATE_SCHEMA_VERSIONS
    ):
        raise BrainValidationError("state schema is unsupported")
    schema = root["schema"]
    if expected_schema is not None and schema != expected_schema:
        raise BrainValidationError("state schema mismatch with stored row schema")
    has_current_envelope = blob.startswith(_STATE_BLOB_MAGIC)
    if has_current_envelope and schema != BRAIN_STATE_SCHEMA_VERSION:
        raise BrainValidationError("current compressed envelope requires state schema 3")
    if schema == BRAIN_STATE_SCHEMA_VERSION and not has_current_envelope:
        raise BrainValidationError("current state schema requires the compressed envelope")
    if root["topology"] != TOPOLOGY_DIGEST:
        raise BrainValidationError("state topology digest does not match")

    b_doc = _object("b", root["b"], _B_FIELDS)
    c_doc = _object("c", root["c"], _C_FIELDS_V2 if schema >= 2 else _C_FIELDS_V1)
    b_horizon = b_doc["horizon"]
    c_horizon = c_doc["horizon"]
    b_records = _eligibility_records(
        "b.elig",
        b_doc["elig"],
        horizon=32,
        trace_length=8,
        trace_code="d",
        factory=BEligibilityRecord,
    )
    c_records = _c_eligibility_records(
        "c.elig",
        c_doc["elig"],
        horizon=32,
        schema=schema,
    )
    b_state = BrainState(
        generation=cast(Any, b_doc["generation"]),
        lineage_id=cast(Any, b_doc["lineage"]),
        e=_decode_array("b.e", b_doc["e"], length=8, code="d"),
        d_plus=_decode_array("b.dp", b_doc["dp"], length=8, code="d"),
        d_minus=_decode_array("b.dm", b_doc["dm"], length=8, code="d"),
        gain_b=_decode_array("b.gain", b_doc["gain"], length=8, code="d"),
        theta_b=_decode_array("b.theta", b_doc["theta"], length=8, code="d"),
        clock=_json_number("b.clock", b_doc["clock"]),
        tick_id=cast(Any, b_doc["tick"]),
        history_epoch=cast(Any, b_doc["history"]),
        mutation_seq=cast(Any, b_doc["mutation"]),
        eligibility_ring=b_records,
        eligibility_horizon=cast(Any, b_horizon),
        clock_regressions=cast(Any, b_doc["regressions"]),
    )
    w_out = (
        _decode_array("c.w_out", c_doc["w_out"], length=N_BASIS * N_AXES, code="f")
        if schema >= 2
        else DEFAULT_W_OUT
    )
    c_state = BrainCState(
        v=_decode_array("c.v", c_doc["v"], length=32, code="f"),
        adaptation=_decode_array("c.adaptation", c_doc["adaptation"], length=32, code="f"),
        filtered=_decode_array("c.filtered", c_doc["filtered"], length=32, code="f"),
        weights=_decode_array("c.w", c_doc["w"], length=128, code="f"),
        w_out=w_out,
        eligibility_ring=c_records,
        eligibility_horizon=cast(Any, c_horizon),
    )
    return BrainBundle(b_state, c_state)
