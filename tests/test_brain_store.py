from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import zlib
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import pytest

import sylanne_core.compute.brain_codec as codec_module
import sylanne_core.compute.brain_store as store_module
from sylanne_core.compute.brain_c import (
    DEFAULT_W_OUT,
    N_AXES,
    N_BASIS,
    N_EDGES,
    N_NEURONS,
    TOPOLOGY,
    TOPOLOGY_DIGEST,
    BrainCState,
    CEligibilityRecord,
    evolve_c_event,
)
from sylanne_core.compute.brain_codec import (
    BRAIN_STATE_SCHEMA_VERSION,
    MAX_STATE_BLOB_BYTES,
    MAX_STATE_PAYLOAD_BYTES,
    BrainBundle,
    decode_brain_bundle,
    encode_brain_bundle,
    inspect_brain_bundle_encoding,
    state_sha256,
)
from sylanne_core.compute.brain_errors import (
    BrainAllocationError,
    BrainCounterExhaustedError,
    BrainDurabilityError,
    BrainValidationError,
)
from sylanne_core.compute.brain_state import (
    MAX_COUNTER,
    BEligibilityRecord,
    BrainState,
    EventAllocation,
    FeedbackAllocation,
)
from sylanne_core.compute.brain_store import (
    DEDUP_TTL_NS,
    AppliedFeedbackCommit,
    BackendCheckpoint,
    BrainStateStore,
    EventAllocated,
    EventCommit,
    EventCommitted,
    EventDuplicate,
    EventMiss,
    FeedbackAllocated,
    FeedbackCommitted,
    FeedbackDuplicate,
    FeedbackMiss,
    ReceiptOnlyFeedbackCommit,
    SessionLoaded,
    SessionMissing,
    StoredReceipt,
    event_id_digest,
    feedback_id_digest,
    session_digest,
)

LINEAGE = "11111111-1111-4111-8111-111111111111"
ZERO8 = (0.0,) * 8
STATE_BLOB_MAGIC = b"SYLBCZ1\x00"
T = TypeVar("T")


def _fresh_bundle(*, horizon: int = 8) -> BrainBundle:
    return BrainBundle(
        BrainState.fresh(lineage_id=LINEAGE, feedback_horizon=horizon),
        BrainCState.fresh(feedback_horizon=horizon),
    )


def _max_bundle() -> BrainBundle:
    b_records = tuple(
        BEligibilityRecord(tick, float(tick), (tick / 64.0,) * 8) for tick in range(1, 33)
    )
    c_records = tuple(
        CEligibilityRecord(tick, float(tick), (float(tick % 2),) * N_EDGES) for tick in range(1, 33)
    )
    return BrainBundle(
        BrainState(
            generation=7,
            lineage_id=LINEAGE,
            e=(-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0),
            d_plus=(1.0,) * 8,
            d_minus=(2.0,) * 8,
            gain_b=(0.5,) * 8,
            theta_b=(0.25,) * 8,
            clock=32.0,
            tick_id=32,
            history_epoch=32,
            mutation_seq=32,
            eligibility_ring=b_records,
            eligibility_horizon=32,
            clock_regressions=3,
        ),
        BrainCState(
            v=(0.25,) * N_NEURONS,
            adaptation=(0.5,) * N_NEURONS,
            filtered=(0.75,) * N_NEURONS,
            weights=TOPOLOGY.initial_weights,
            eligibility_ring=c_records,
            eligibility_horizon=32,
        ),
    )


def _high_entropy_bundle() -> BrainBundle:
    rng = random.Random(2718)
    b_records = tuple(
        BEligibilityRecord(tick, float(tick), tuple(rng.random() for _ in range(8)))
        for tick in range(1, 33)
    )
    c_records = tuple(
        CEligibilityRecord(
            tick,
            float(tick),
            tuple(8.0 * rng.random() for _ in range(N_EDGES)),
            tuple(rng.random() for _ in range(N_BASIS)),
        )
        for tick in range(1, 33)
    )
    return BrainBundle(
        BrainState(
            generation=7,
            lineage_id=LINEAGE,
            e=tuple(2.0 * rng.random() - 1.0 for _ in range(8)),
            d_plus=tuple(32.0 * rng.random() for _ in range(8)),
            d_minus=tuple(32.0 * rng.random() for _ in range(8)),
            gain_b=tuple(0.05 + 0.95 * rng.random() for _ in range(8)),
            theta_b=tuple(0.95 * rng.random() for _ in range(8)),
            clock=32.0,
            tick_id=32,
            history_epoch=32,
            mutation_seq=32,
            eligibility_ring=b_records,
            eligibility_horizon=32,
            clock_regressions=3,
        ),
        BrainCState(
            v=tuple(4.0 * rng.random() - 2.0 for _ in range(N_NEURONS)),
            adaptation=tuple(rng.random() for _ in range(N_NEURONS)),
            filtered=tuple(rng.random() for _ in range(N_NEURONS)),
            weights=tuple(2.0 * rng.random() - 1.0 for _ in range(N_EDGES)),
            w_out=tuple(2.0 * rng.random() - 1.0 for _ in range(N_BASIS * N_AXES)),
            eligibility_ring=c_records,
            eligibility_horizon=32,
        ),
    )


def _canonical(document: dict[str, Any], *, allow_nan: bool = False) -> bytes:
    return json.dumps(
        document,
        allow_nan=allow_nan,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _json_payload(blob: bytes) -> bytes:
    assert blob.startswith(STATE_BLOB_MAGIC)
    return zlib.decompress(blob[len(STATE_BLOB_MAGIC) :])


def _current_blob(document: dict[str, Any], *, allow_nan: bool = False) -> bytes:
    payload = _canonical(document, allow_nan=allow_nan)
    return STATE_BLOB_MAGIC + zlib.compress(payload, level=9)


def _mutate(
    blob: bytes,
    change: Callable[[dict[str, Any]], None],
    *,
    allow_nan: bool = False,
) -> tuple[bytes, bytes]:
    document = cast(dict[str, Any], json.loads(_json_payload(blob)))
    change(document)
    changed = _current_blob(document, allow_nan=allow_nan)
    return changed, state_sha256(changed)


def _event_material(allocated: EventAllocated) -> tuple[BrainBundle, StoredReceipt]:
    old = allocated.bundle
    allocation = allocated.allocation
    clock = old.b.clock + 1.0
    records = old.b.eligibility_records + (
        BEligibilityRecord(allocation.tick_id, clock, (0.5,) * 8),
    )
    next_b = BrainState(
        generation=allocation.generation,
        lineage_id=allocation.lineage_id,
        e=(0.125,) + tuple(old.b.e[1:]),
        d_plus=(old.b.d_plus[0] + 0.25,) + tuple(old.b.d_plus[1:]),
        d_minus=old.b.d_minus,
        gain_b=old.b.gain_b,
        theta_b=old.b.theta_b,
        clock=clock,
        tick_id=allocation.tick_id,
        history_epoch=allocation.history_epoch,
        mutation_seq=allocation.mutation_seq,
        eligibility_ring=records[-old.b.eligibility_horizon :],
        eligibility_horizon=old.b.eligibility_horizon,
        clock_regressions=old.b.clock_regressions,
    )
    next_c = evolve_c_event(
        old.c,
        (1.0,) + ZERO8[1:],
        route="normal",
        tick_id=allocation.tick_id,
        created_at=clock,
        delta_t=1.0,
    ).state
    bundle = BrainBundle(next_b, next_c)
    return bundle, StoredReceipt(
        kind="event",
        status="applied",
        generation=next_b.generation,
        tick_id=next_b.tick_id,
        history_epoch=next_b.history_epoch,
        mutation_seq=next_b.mutation_seq,
    )


def _feedback_material(allocated: FeedbackAllocated) -> tuple[BrainBundle, StoredReceipt]:
    old = allocated.bundle
    allocation = allocated.allocation
    gain = old.b.gain_b
    gain[0] = min(1.0, gain[0] + 0.01)
    next_b = BrainState(
        generation=old.b.generation,
        lineage_id=old.b.lineage_id,
        e=old.b.e,
        d_plus=old.b.d_plus,
        d_minus=old.b.d_minus,
        gain_b=gain,
        theta_b=old.b.theta_b,
        clock=old.b.clock,
        tick_id=old.b.tick_id,
        history_epoch=old.b.history_epoch,
        mutation_seq=allocation.next_mutation_seq,
        eligibility_ring=old.b.eligibility_records,
        eligibility_horizon=old.b.eligibility_horizon,
        clock_regressions=old.b.clock_regressions,
    )
    weights = old.c.weights
    weights[0] = min(1.0, weights[0] + 0.01)
    next_c = BrainCState(
        v=old.c.v,
        adaptation=old.c.adaptation,
        filtered=old.c.filtered,
        weights=weights,
        eligibility_ring=old.c.eligibility_records,
        eligibility_horizon=old.c.eligibility_horizon,
    )
    bundle = BrainBundle(next_b, next_c)
    return bundle, StoredReceipt(
        kind="feedback",
        status="applied",
        generation=next_b.generation,
        tick_id=next_b.tick_id,
        history_epoch=next_b.history_epoch,
        mutation_seq=next_b.mutation_seq,
        target_tick=allocation.target_tick,
        applied_dimensions=(0,),
        applied_synapses=1,
    )


def _checkpoint(bundle: BrainBundle, token: bytes = b"checkpoint") -> BackendCheckpoint:
    return BackendCheckpoint(
        generation=bundle.b.generation,
        backend_name="private-test",
        backend_state_version=3,
        acknowledged_mutation_seq=bundle.b.mutation_seq,
        token=token,
        token_sha256=hashlib.sha256(token).digest(),
    )


def _assert_event_committed(
    result: object,
    receipt: StoredReceipt,
    session_key: bytes,
    event_key: bytes,
    checkpoint: BackendCheckpoint | None = None,
) -> EventCommitted:
    assert isinstance(result, EventCommitted)
    assert result.receipt == receipt
    assert result.session_digest == session_key
    assert result.id_digest == event_key
    assert result.checkpoint_token_sha256 == (
        None if checkpoint is None else checkpoint.token_sha256
    )
    return result


def _assert_feedback_committed(
    result: object,
    receipt: StoredReceipt,
    session_key: bytes,
    feedback_key: bytes,
    checkpoint: BackendCheckpoint | None = None,
) -> FeedbackCommitted:
    assert isinstance(result, FeedbackCommitted)
    assert result.receipt == receipt
    assert result.session_digest == session_key
    assert result.id_digest == feedback_key
    assert result.checkpoint_token_sha256 == (
        None if checkpoint is None else checkpoint.token_sha256
    )
    return result


def _event_commit(
    store: BrainStateStore,
    session_id: str,
    event_id: str,
    *,
    checkpoint: BackendCheckpoint | None = None,
) -> tuple[BrainBundle, StoredReceipt]:
    session_key = session_digest(session_id)
    event_key = event_id_digest(event_id)
    outcome = store.preflight_allocate(session_key, event_key)
    assert isinstance(outcome, EventAllocated)
    bundle, receipt = _event_material(outcome)
    result = store.commit_event(
        session_key,
        event_key,
        EventCommit(outcome, bundle, receipt, checkpoint),
    )
    _assert_event_committed(result, receipt, session_key, event_key, checkpoint)
    return bundle, receipt


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition timed out")


def _thread_call(
    function: Callable[[], T],
) -> tuple[threading.Thread, list[T], list[BaseException]]:
    results: list[T] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(function())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results, errors


def _table_snapshot(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in (
                "brain_session",
                "brain_control",
                "brain_dedup",
                "brain_backend_checkpoint",
            )
        }


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    return environment


def test_codec_is_deterministic_big_endian_and_round_trips_exact_precision() -> None:
    bundle = _max_bundle()
    blob, digest = encode_brain_bundle(bundle)
    assert encode_brain_bundle(bundle.copy()) == (blob, digest)
    assert digest == hashlib.sha256(blob).digest() and len(digest) == 32
    assert decode_brain_bundle(blob, digest) == bundle
    assert blob.startswith(STATE_BLOB_MAGIC)
    payload = _json_payload(blob)
    assert b" " not in payload and b"\n" not in payload
    document = cast(dict[str, Any], json.loads(payload))
    b_doc = cast(dict[str, Any], document["b"])
    c_doc = cast(dict[str, Any], document["c"])
    assert base64.b64decode(b_doc["e"], validate=True) == struct.pack(">8d", *bundle.b.e)
    assert base64.b64decode(b_doc["elig"][0]["trace"], validate=True) == struct.pack(
        ">8d", *bundle.b.eligibility_records[0].b_trace
    )
    assert base64.b64decode(c_doc["v"], validate=True) == struct.pack(">32f", *bundle.c.v)
    assert base64.b64decode(c_doc["elig"][0]["trace"], validate=True) == struct.pack(
        ">128f", *bundle.c.eligibility_records[0].c_trace
    )
    assert document["schema"] == BRAIN_STATE_SCHEMA_VERSION == 3
    assert document["topology"] == TOPOLOGY_DIGEST
    assert len(blob) < MAX_STATE_BLOB_BYTES == 32768
    assert base64.b64decode(c_doc["w_out"], validate=True) == struct.pack(">384f", *bundle.c.w_out)
    assert base64.b64decode(c_doc["elig"][0]["basis"], validate=True) == struct.pack(
        ">48f", *bundle.c.eligibility_records[0].basis_trace
    )


def test_codec_size_inspector_uses_the_production_candidate_encoding() -> None:
    representative = _max_bundle()
    adversarial = _high_entropy_bundle()
    representative_blob, _ = encode_brain_bundle(representative)
    adversarial_blob, _ = encode_brain_bundle(adversarial)

    representative_metrics = inspect_brain_bundle_encoding(representative)
    adversarial_metrics = inspect_brain_bundle_encoding(adversarial)

    assert representative_metrics == inspect_brain_bundle_encoding(representative.copy())
    assert representative_metrics.payload_bytes == len(_json_payload(representative_blob))
    assert representative_metrics.blob_bytes == len(representative_blob)
    assert adversarial_metrics.payload_bytes == len(_json_payload(adversarial_blob))
    assert adversarial_metrics.blob_bytes == len(adversarial_blob)
    assert 0 < adversarial_metrics.payload_bytes < MAX_STATE_PAYLOAD_BYTES == 64 * 1024
    assert representative_metrics.blob_bytes < adversarial_metrics.blob_bytes


def test_codec_high_entropy_max_horizon_stays_below_32k_and_round_trips() -> None:
    bundle = _high_entropy_bundle()
    blob, digest = encode_brain_bundle(bundle)

    assert len(blob) < MAX_STATE_BLOB_BYTES == 32 * 1024
    assert decode_brain_bundle(blob, digest) == bundle


def test_codec_reads_legacy_schema_2_json_larger_than_the_new_32k_write_limit() -> None:
    bundle = _max_bundle()
    current_blob, _ = encode_brain_bundle(bundle)
    legacy_document = cast(dict[str, Any], json.loads(_json_payload(current_blob)))
    legacy_document["schema"] = 2
    legacy_blob = _canonical(legacy_document)

    assert MAX_STATE_BLOB_BYTES <= len(legacy_blob) < 64 * 1024
    assert decode_brain_bundle(legacy_blob, state_sha256(legacy_blob)) == bundle


@pytest.mark.parametrize(
    ("legacy_schema", "expected_length", "expected_sha256"),
    (
        (1, 2_000, "d7a4f701996a21a0537faeec8904fa43849f0b99b7a7fd9c191e4d9302884b37"),
        (2, 4_059, "cefb3949b4b56d6ff80e25ee091b1255a9edb17d6049e7d779a6fbd618f6fd0e"),
    ),
)
def test_legacy_bare_json_wire_goldens_remain_decodable(
    legacy_schema: int,
    expected_length: int,
    expected_sha256: str,
) -> None:
    current_blob, _ = encode_brain_bundle(_fresh_bundle())
    document = cast(dict[str, Any], json.loads(_json_payload(current_blob)))
    document["schema"] = legacy_schema
    if legacy_schema == 1:
        c_document = cast(dict[str, Any], document["c"])
        c_document.pop("w_out")
        for record in cast(list[dict[str, Any]], c_document["elig"]):
            record.pop("basis")
    legacy_blob = _canonical(document)

    assert len(legacy_blob) == expected_length
    assert hashlib.sha256(legacy_blob).hexdigest() == expected_sha256
    decode_brain_bundle(
        legacy_blob,
        state_sha256(legacy_blob),
        expected_schema=legacy_schema,
    )


def test_codec_rejects_bare_current_schema_outside_legacy_migration() -> None:
    bundle = _fresh_bundle()
    current_blob, _ = encode_brain_bundle(bundle)
    bare_current = _json_payload(current_blob)

    with pytest.raises(BrainValidationError, match="compressed envelope"):
        decode_brain_bundle(bare_current, state_sha256(bare_current))


def test_codec_rejects_legacy_schema_inside_current_compressed_envelope() -> None:
    current_blob, _ = encode_brain_bundle(_fresh_bundle())
    legacy_document = cast(dict[str, Any], json.loads(_json_payload(current_blob)))
    legacy_document["schema"] = 2
    compressed_legacy = _current_blob(legacy_document)

    with pytest.raises(BrainValidationError, match="compressed envelope.*schema 3"):
        decode_brain_bundle(compressed_legacy, state_sha256(compressed_legacy))


def test_codec_rejects_unknown_compressed_envelope_before_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_blob, _ = encode_brain_bundle(_fresh_bundle())
    unknown_envelope = b"SYLBCZ2\x00" + current_blob[len(STATE_BLOB_MAGIC) :]
    monkeypatch.setattr(codec_module.json, "loads", lambda _blob: pytest.fail("JSON reached"))

    with pytest.raises(BrainValidationError, match="compressed envelope is unsupported"):
        decode_brain_bundle(unknown_envelope, state_sha256(unknown_envelope))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda _blob: STATE_BLOB_MAGIC + b"not-zlib", "invalid compressed data"),
        (lambda blob: blob[:-1], "truncated compressed data"),
        (lambda blob: blob + b"trailing", "trailing compressed data"),
    ),
)
def test_codec_rejects_invalid_truncated_and_trailing_compressed_streams(
    mutate: Callable[[bytes], bytes],
    message: str,
) -> None:
    current_blob, _ = encode_brain_bundle(_fresh_bundle())
    damaged = mutate(current_blob)

    with pytest.raises(BrainValidationError, match=message):
        decode_brain_bundle(damaged, state_sha256(damaged))


def test_codec_rejects_row_and_inner_schema_mismatch() -> None:
    blob, digest = encode_brain_bundle(_fresh_bundle())

    with pytest.raises(BrainValidationError, match="schema.*mismatch"):
        decode_brain_bundle(blob, digest, expected_schema=2)


def test_codec_schema_1_blob_one_way_migrates_to_legacy_contrast_w_out_and_zero_basis() -> None:
    """Schema 1 (pre-plastic-readout) blobs have no "w_out" field on c and no
    "basis" field on c eligibility records. They must decode to the
    legacy-contrast init (bit-identical to the old hardcoded readout) and
    zeroed basis traces -- a one-way migration, never re-encoded as schema 1."""
    bundle = _max_bundle()
    blob, _digest = encode_brain_bundle(bundle)
    document = cast(dict[str, Any], json.loads(_json_payload(blob)))
    c_doc = cast(dict[str, Any], document["c"])
    legacy_c = {key: value for key, value in c_doc.items() if key != "w_out"}
    legacy_c["elig"] = [
        {key: value for key, value in record.items() if key != "basis"}
        for record in cast(list[dict[str, Any]], c_doc["elig"])
    ]
    legacy_document = dict(document)
    legacy_document["c"] = legacy_c
    legacy_document["schema"] = 1
    legacy_blob = json.dumps(
        legacy_document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    legacy_digest = state_sha256(legacy_blob)

    decoded = decode_brain_bundle(legacy_blob, legacy_digest)

    assert decoded.c.w_out == array("f", DEFAULT_W_OUT)
    assert all(
        record.basis_trace == array("f", (0.0,) * N_BASIS)
        for record in decoded.c.eligibility_records
    )
    # Everything else round-trips identically to the current-schema bundle.
    assert decoded.b == bundle.b
    assert decoded.c.v == bundle.c.v
    assert decoded.c.adaptation == bundle.c.adaptation
    assert decoded.c.filtered == bundle.c.filtered
    assert decoded.c.weights == bundle.c.weights
    assert tuple(record.c_trace for record in decoded.c.eligibility_records) == tuple(
        record.c_trace for record in bundle.c.eligibility_records
    )
    assert len(decoded.c.w_out) == N_BASIS * N_AXES


@pytest.mark.parametrize(
    ("blob", "digest", "message"),
    (
        (b"not-json", hashlib.sha256(b"not-json").digest(), "JSON"),
        (b'{"schema":1', hashlib.sha256(b'{"schema":1').digest(), "JSON"),
        (b"{}", b"0" * 32, "checksum"),
    ),
)
def test_codec_rejects_invalid_json_truncation_and_checksum(
    blob: bytes, digest: bytes, message: str
) -> None:
    with pytest.raises(BrainValidationError, match=message):
        decode_brain_bundle(blob, digest)


def test_codec_hard_size_gate_runs_before_json_and_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_STATE_BLOB_BYTES == 32 * 1024
    oversized = STATE_BLOB_MAGIC + b"x" * (MAX_STATE_BLOB_BYTES - len(STATE_BLOB_MAGIC))
    monkeypatch.setattr(
        codec_module.zlib, "decompressobj", lambda: pytest.fail("decompression reached")
    )
    monkeypatch.setattr(codec_module.json, "loads", lambda _blob: pytest.fail("JSON reached"))
    monkeypatch.setattr(
        codec_module.base64, "b64decode", lambda *_args, **_kwargs: pytest.fail("base64 reached")
    )
    with pytest.raises(BrainValidationError, match="size"):
        decode_brain_bundle(oversized, state_sha256(oversized))


def test_codec_bounds_decompressed_json_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    oversized_json = b"{" + b" " * (64 * 1024 - 1)
    compressed = STATE_BLOB_MAGIC + zlib.compress(oversized_json, level=9)
    monkeypatch.setattr(codec_module.json, "loads", lambda _blob: pytest.fail("JSON reached"))

    with pytest.raises(BrainValidationError, match="decompressed.*size"):
        decode_brain_bundle(compressed, state_sha256(compressed))


def test_codec_rejects_base64_length_bomb_invalid_alphabet_and_fixed_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob, _ = encode_brain_bundle(_fresh_bundle())
    bomb, bomb_hash = _mutate(
        blob, lambda doc: cast(dict[str, Any], doc["b"]).__setitem__("e", "A" * 20_000)
    )
    real_decode = base64.b64decode
    monkeypatch.setattr(
        codec_module.base64,
        "b64decode",
        lambda *_args, **_kwargs: pytest.fail("bomb was decoded"),
    )
    with pytest.raises(BrainValidationError, match="base64.*length"):
        decode_brain_bundle(bomb, bomb_hash)
    monkeypatch.setattr(codec_module.base64, "b64decode", real_decode)

    invalid, invalid_hash = _mutate(
        blob, lambda doc: cast(dict[str, Any], doc["b"]).__setitem__("e", "*" * 88)
    )
    with pytest.raises(BrainValidationError, match="base64"):
        decode_brain_bundle(invalid, invalid_hash)
    short, short_hash = _mutate(
        blob,
        lambda doc: cast(dict[str, Any], doc["c"]).__setitem__(
            "w", base64.b64encode(b"short").decode("ascii")
        ),
    )
    with pytest.raises(BrainValidationError, match="length"):
        decode_brain_bundle(short, short_hash)


def test_codec_rejects_duplicate_unknown_missing_bool_nonfinite_and_domain_fields() -> None:
    blob, _ = encode_brain_bundle(_fresh_bundle())
    duplicate_payload = _json_payload(blob).replace(b'{"b":', b'{"schema":1,"b":', 1)
    duplicate = STATE_BLOB_MAGIC + zlib.compress(duplicate_payload, level=9)
    with pytest.raises(BrainValidationError, match="duplicate"):
        decode_brain_bundle(duplicate, state_sha256(duplicate))

    for change, message, allow_nan in (
        (lambda doc: doc.__setitem__("unknown", 1), "unknown", False),
        (lambda doc: doc.pop("topology"), "missing", False),
        (
            lambda doc: cast(dict[str, Any], doc["b"]).__setitem__("tick", True),
            "non-boolean",
            False,
        ),
        (
            lambda doc: cast(dict[str, Any], doc["b"]).__setitem__("clock", float("nan")),
            "finite",
            True,
        ),
        (lambda doc: doc.__setitem__("topology", "0" * 64), "topology", False),
        (lambda doc: doc.__setitem__("schema", 999), "schema", False),
    ):
        changed, digest = _mutate(blob, change, allow_nan=allow_nan)
        with pytest.raises(BrainValidationError, match=message):
            decode_brain_bundle(changed, digest)

    domain, domain_hash = _mutate(
        blob,
        lambda doc: cast(dict[str, Any], doc["b"]).__setitem__(
            "e", base64.b64encode(struct.pack(">8d", 2.0, *ZERO8[1:])).decode()
        ),
    )
    with pytest.raises(BrainValidationError, match="outside"):
        decode_brain_bundle(domain, domain_hash)


def test_codec_normalizes_deep_json_recursion() -> None:
    depth = 5_000
    blob = b"[" * depth + b"0" + b"]" * depth
    with pytest.raises(BrainValidationError, match="JSON"):
        decode_brain_bundle(blob, state_sha256(blob))


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        ("b", "_BrainState__e"),
        ("c", "_BrainCState__v"),
    ),
)
def test_codec_rejects_corrupted_nonfinite_internal_arrays(
    owner: str,
    attribute: str,
) -> None:
    bundle = _fresh_bundle()
    state = bundle.b if owner == "b" else bundle.c
    object.__getattribute__(state, attribute)[0] = float("nan")
    with pytest.raises(BrainValidationError, match="finite"):
        encode_brain_bundle(bundle)


@pytest.mark.parametrize(
    "change",
    (
        lambda doc: cast(dict[str, Any], doc["b"]).__setitem__("clock", "0.0"),
        lambda doc: cast(list[dict[str, Any]], cast(dict[str, Any], doc["b"])["elig"])[
            0
        ].__setitem__("created_at", "1.0"),
        lambda doc: cast(list[dict[str, Any]], cast(dict[str, Any], doc["c"])["elig"])[
            0
        ].__setitem__("created_at", "1.0"),
    ),
)
def test_codec_rejects_numeric_strings_in_json_fields(
    change: Callable[[dict[str, Any]], None],
) -> None:
    blob, _ = encode_brain_bundle(_max_bundle())
    changed, digest = _mutate(blob, change)
    with pytest.raises(BrainValidationError, match="JSON number"):
        decode_brain_bundle(changed, digest)


def test_bundle_rejects_cross_b_c_tick_time_and_horizon_invariants() -> None:
    b = BrainState.fresh(lineage_id=LINEAGE, feedback_horizon=8)
    future_c = BrainCState(
        v=(0.0,) * N_NEURONS,
        adaptation=(0.0,) * N_NEURONS,
        filtered=(0.0,) * N_NEURONS,
        weights=TOPOLOGY.initial_weights,
        eligibility_ring=(CEligibilityRecord(1, 1.0, (0.0,) * N_EDGES),),
        eligibility_horizon=8,
    )
    with pytest.raises(BrainValidationError, match="C.*B"):
        BrainBundle(b, future_c)
    with pytest.raises(BrainValidationError, match="horizon"):
        BrainBundle(b, BrainCState.fresh(feedback_horizon=4))


def test_raw_digest_contract_uses_full_utf8_and_domain_separation() -> None:
    assert session_digest("") != session_digest("default")
    assert session_digest("x" * 1000 + "a") != session_digest("x" * 1000 + "b")
    assert session_digest("澜") == hashlib.sha256("澜".encode()).digest()
    assert event_id_digest("same") != feedback_id_digest("same")
    assert all(
        len(value) == 32
        for value in (session_digest("s"), event_id_digest("e"), feedback_id_digest("f"))
    )


@pytest.mark.parametrize(
    "digest_function",
    (session_digest, event_id_digest, feedback_id_digest),
)
def test_identifier_digests_normalize_illegal_unicode(
    digest_function: Callable[[str], bytes],
) -> None:
    with pytest.raises(BrainValidationError, match="UTF-8"):
        digest_function("\ud800")


def test_store_boundary_requires_fixed_raw_digests(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    event_key = event_id_digest("e")
    feedback_key = feedback_id_digest("f")
    try:
        assert store.load(session_key) == SessionMissing()
        assert store.lookup_event_receipt(session_key, event_key) == EventMiss()
        assert store.lookup_feedback_receipt(session_key, feedback_key) == FeedbackMiss()
        for invalid in ("raw-id", b"", b"x" * 31, b"x" * 33):
            with pytest.raises(BrainValidationError, match="raw 32 bytes"):
                store.load(cast(Any, invalid))
            with pytest.raises(BrainValidationError, match="raw 32 bytes"):
                store.lookup_event_receipt(session_key, cast(Any, invalid))
            with pytest.raises(BrainValidationError, match="raw 32 bytes"):
                store.lookup_feedback_receipt(session_key, cast(Any, invalid))
    finally:
        store.close()


def test_store_owner_connection_exact_pragmas_tables_and_receipt_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, dict[str, object]]] = []
    real_connect = sqlite3.connect

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        calls.append((threading.get_ident(), dict(kwargs)))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(store_module.sqlite3, "connect", recording_connect)
    store = BrainStateStore.start(tmp_path)
    try:
        assert calls == [
            (store._thread.ident, {"isolation_level": None, "check_same_thread": True})
        ]
        assert store._queue.maxsize == 1024
        assert store._debug_pragmas() == {
            "journal_mode": "wal",
            "synchronous": 2,
            "foreign_keys": 1,
            "busy_timeout": 5000,
        }
        with sqlite3.connect(store.database_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'brain_%'"
                )
            }
            columns = {
                table: [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
                for table in tables
            }
        assert tables == {
            "brain_session",
            "brain_control",
            "brain_dedup",
            "brain_backend_checkpoint",
        }
        assert columns["brain_session"] == [
            "session_digest",
            "schema_version",
            "generation",
            "lineage_id",
            "history_epoch",
            "mutation_seq",
            "state_blob",
            "state_sha256",
            "updated_ns",
        ]
        assert columns["brain_dedup"][-2:] == ["receipt_sha256", "created_ns"]
        assert "state_version" not in columns["brain_session"]
    finally:
        store.close()


def test_startup_failure_closes_partial_connection_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenConnection:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, _statement: str) -> object:
            raise sqlite3.OperationalError("setup failed")

        def close(self) -> None:
            self.closed = True

    connection = BrokenConnection()
    real_connect = sqlite3.connect
    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *_args, **_kwargs: connection)
    with pytest.raises(BrainDurabilityError, match="OperationalError"):
        BrainStateStore.start(tmp_path)
    assert connection.closed

    monkeypatch.setattr(store_module.sqlite3, "connect", real_connect)
    BrainStateStore.start(tmp_path).close()


def test_startup_rejects_unapplied_required_pragmas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IgnoreWalConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection
            self.closed = False

        def execute(self, statement: str, *args: object) -> sqlite3.Cursor:
            if statement == "PRAGMA journal_mode=WAL":
                return self._connection.execute("PRAGMA journal_mode")
            return self._connection.execute(statement, *args)

        def executescript(self, script: str) -> sqlite3.Cursor:
            return self._connection.executescript(script)

        @property
        def in_transaction(self) -> bool:
            return self._connection.in_transaction

        def close(self) -> None:
            self.closed = True
            self._connection.close()

    real_connect = sqlite3.connect
    wrappers: list[IgnoreWalConnection] = []

    def ignoring_connect(*args: object, **kwargs: object) -> IgnoreWalConnection:
        wrapper = IgnoreWalConnection(real_connect(*args, **kwargs))
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(store_module.sqlite3, "connect", ignoring_connect)
    started: BrainStateStore | None = None
    try:
        with pytest.raises(BrainDurabilityError, match="PRAGMA"):
            started = BrainStateStore.start(tmp_path)
    finally:
        if started is not None:
            started.close()
    assert wrappers and wrappers[0].closed


def test_startup_wait_timeout_cancels_and_reaps_late_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores: list[BrainStateStore] = []
    open_entered = threading.Event()
    release_open = threading.Event()
    original_init = BrainStateStore.__init__
    original_open = BrainStateStore._open_database
    real_event_wait = threading.Event.wait

    def recording_init(self: BrainStateStore, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        stores.append(self)

    def slow_open(self: BrainStateStore) -> sqlite3.Connection:
        open_entered.set()
        release_open.wait(5)
        return original_open(self)

    def force_start_timeout(event: threading.Event, timeout: float | None = None) -> bool:
        if stores and event is stores[0]._ready:
            return False
        return real_event_wait(event, timeout)

    monkeypatch.setattr(BrainStateStore, "__init__", recording_init)
    monkeypatch.setattr(BrainStateStore, "_open_database", slow_open)
    monkeypatch.setattr(threading.Event, "wait", force_start_timeout)

    def release_later() -> None:
        assert open_entered.wait(5)
        time.sleep(0.05)
        release_open.set()

    releaser = threading.Thread(target=release_later)
    releaser.start()
    with pytest.raises(BrainDurabilityError, match="did not start"):
        BrainStateStore.start(tmp_path)
    releaser.join(5)
    assert stores
    store = stores[0]
    orphaned = store._thread.is_alive() or store._lease_registered
    if orphaned:
        store.close()
    assert not orphaned
    assert store._closed_event.is_set()


def test_typed_phase_one_miss_does_not_create_state_and_formal_preflight_allocates(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("raw-session")
        event_key = event_id_digest("raw-event")
        assert store.lookup_event_receipt(session_key, event_key) == EventMiss()
        assert _table_snapshot(store.database_path)["brain_session"] == []
        outcome = store.preflight_allocate(session_key, event_key)
        assert isinstance(outcome, EventAllocated)
        assert outcome.allocation == EventAllocation(
            generation=outcome.bundle.b.generation,
            lineage_id=outcome.bundle.b.lineage_id,
            tick_id=1,
            history_epoch=1,
            mutation_seq=1,
        )
    finally:
        store.close()


def test_atomic_event_checkpoint_receipt_allowlist_and_no_raw_ids_or_source(tmp_path: Path) -> None:
    session_id = "secret-session-原文"
    event_id = "secret-event-原文"
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest(session_id)
        event_key = event_id_digest(event_id)
        allocated = store.preflight_allocate(session_key, event_key)
        assert isinstance(allocated, EventAllocated)
        bundle, receipt = _event_material(allocated)
        checkpoint = _checkpoint(bundle, b"\x00raw-token\xff")
        result = store.commit_event(
            session_key, event_key, EventCommit(allocated, bundle, receipt, checkpoint)
        )
        _assert_event_committed(result, receipt, session_key, event_key, checkpoint)
        assert store.load(session_key) == SessionLoaded(bundle, checkpoint)
        with sqlite3.connect(store.database_path) as connection:
            session_row = connection.execute(
                "SELECT schema_version,generation,lineage_id,history_epoch,mutation_seq,"
                "state_blob,state_sha256 "
                "FROM brain_session"
            ).fetchone()
            dedup = connection.execute(
                "SELECT kind,id_digest,mutation_seq,receipt_blob,receipt_sha256 FROM brain_dedup"
            ).fetchone()
            backend = connection.execute(
                "SELECT generation,backend_state_version,acknowledged_mutation_seq,"
                "token_blob,token_sha256 FROM brain_backend_checkpoint"
            ).fetchone()
        assert session_row[0] == BRAIN_STATE_SCHEMA_VERSION == 3
        state_blob = session_row[5]
        assert session_row[1:5] == (
            bundle.b.generation,
            bundle.b.lineage_id,
            bundle.b.history_epoch,
            bundle.b.mutation_seq,
        )
        assert session_row[6] == state_sha256(state_blob)
        assert dedup[:3] == ("event", event_id_digest(event_id), 1)
        assert dedup[4] == hashlib.sha256(dedup[3]).digest()
        receipt_doc = json.loads(dedup[3])
        assert set(receipt_doc) == {
            "applied_dimensions",
            "applied_synapses",
            "generation",
            "history_epoch",
            "kind",
            "mutation_seq",
            "status",
            "target_tick",
            "tick_id",
        }
        assert receipt_doc["status"] != "duplicate" and "source" not in receipt_doc
        assert backend == (0, 3, 1, checkpoint.token, checkpoint.token_sha256)
        raw = store.database_path.read_bytes()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{store.database_path}{suffix}")
            if sidecar.exists():
                raw += sidecar.read_bytes()
        assert session_id.encode() not in raw and event_id.encode() not in raw
        assert b"private-test" in raw
        assert checkpoint.token not in state_blob
    finally:
        store.close()


@pytest.mark.parametrize("legacy_schema", (1, 2))
def test_store_loads_legacy_json_and_rewrites_schema_3(
    tmp_path: Path,
    legacy_schema: int,
) -> None:
    session_id = f"legacy-schema-{legacy_schema}"
    session_key = session_digest(session_id)
    store = BrainStateStore.start(tmp_path)
    try:
        _event_commit(store, session_id, "event-1")
        database_path = store.database_path
    finally:
        store.close()

    with sqlite3.connect(database_path) as connection:
        current_blob = connection.execute(
            "SELECT state_blob FROM brain_session WHERE session_digest=?",
            (session_key,),
        ).fetchone()[0]
        document = cast(dict[str, Any], json.loads(_json_payload(current_blob)))
        document["schema"] = legacy_schema
        if legacy_schema == 1:
            c_document = cast(dict[str, Any], document["c"])
            c_document.pop("w_out")
            for record in cast(list[dict[str, Any]], c_document["elig"]):
                record.pop("basis")
        legacy_blob = _canonical(document)
        legacy_bundle = decode_brain_bundle(legacy_blob, state_sha256(legacy_blob))
        connection.execute(
            "UPDATE brain_session SET schema_version=?,state_blob=?,state_sha256=? "
            "WHERE session_digest=?",
            (legacy_schema, legacy_blob, state_sha256(legacy_blob), session_key),
        )

    reopened = BrainStateStore.start(tmp_path)
    try:
        assert reopened.load(session_key) == SessionLoaded(legacy_bundle, None)
        with sqlite3.connect(database_path) as connection:
            unchanged = connection.execute(
                "SELECT schema_version,state_blob FROM brain_session WHERE session_digest=?",
                (session_key,),
            ).fetchone()
        assert unchanged == (legacy_schema, legacy_blob)
        _event_commit(reopened, session_id, "event-2")
    finally:
        reopened.close()

    with sqlite3.connect(database_path) as connection:
        schema, rewritten_blob = connection.execute(
            "SELECT schema_version,state_blob FROM brain_session WHERE session_digest=?",
            (session_key,),
        ).fetchone()
    assert schema == BRAIN_STATE_SCHEMA_VERSION == 3
    assert rewritten_blob.startswith(STATE_BLOB_MAGIC)


def test_store_rejects_row_and_blob_schema_mismatch(tmp_path: Path) -> None:
    session_id = "row-blob-schema-mismatch"
    session_key = session_digest(session_id)
    store = BrainStateStore.start(tmp_path)
    try:
        _event_commit(store, session_id, "event-1")
        database_path = store.database_path
    finally:
        store.close()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE brain_session SET schema_version=2 WHERE session_digest=?",
            (session_key,),
        )

    reopened = BrainStateStore.start(tmp_path)
    try:
        with pytest.raises(BrainDurabilityError, match="stored state blob validation failed"):
            reopened.load(session_key)
    finally:
        reopened.close()


@pytest.mark.parametrize("status", ("missed", "no_effect", "disabled"))
def test_event_commit_rejects_non_event_terminal_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        event_key = event_id_digest("e")
        allocated = store.preflight_allocate(session_key, event_key)
        assert isinstance(allocated, EventAllocated)
        bundle, receipt = _event_material(allocated)
        invalid = replace(receipt, status=cast(Any, status))
        with pytest.raises(BrainValidationError, match="event receipt status"):
            store.commit_event(session_key, event_key, EventCommit(allocated, bundle, invalid))
        assert store.load(session_key) == SessionLoaded(allocated.bundle, None)
    finally:
        store.close()


def test_event_commit_persists_degraded_terminal_status(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        event_key = event_id_digest("e")
        allocated = store.preflight_allocate(session_key, event_key)
        assert isinstance(allocated, EventAllocated)
        bundle, receipt = _event_material(allocated)
        degraded = replace(receipt, status="degraded")
        result = store.commit_event(
            session_key, event_key, EventCommit(allocated, bundle, degraded)
        )
        _assert_event_committed(result, degraded, session_key, event_key)
    finally:
        store.close()


def test_checkpoint_accepts_65536_and_rejects_hash_binding_or_65537_before_mutation(
    tmp_path: Path,
) -> None:
    accepted_dir = tmp_path / "accepted"
    store = BrainStateStore.start(accepted_dir)
    session_key = session_digest("s")
    event_key = event_id_digest("e")
    allocated = store.preflight_allocate(session_key, event_key)
    assert isinstance(allocated, EventAllocated)
    bundle, receipt = _event_material(allocated)
    token = b"x" * (64 * 1024)
    checkpoint = _checkpoint(bundle, token)
    store.commit_event(session_key, event_key, EventCommit(allocated, bundle, receipt, checkpoint))
    assert store.load(session_key) == SessionLoaded(bundle, checkpoint)
    store.close()

    for name, bad in (
        ("size", _checkpoint(bundle, b"x" * (64 * 1024 + 1))),
        ("hash", replace(_checkpoint(bundle), token_sha256=b"0" * 32)),
        ("generation", replace(_checkpoint(bundle), generation=1)),
        ("acknowledged", replace(_checkpoint(bundle), acknowledged_mutation_seq=0)),
        ("name", replace(_checkpoint(bundle), backend_name="\ud800")),
        ("name_size", replace(_checkpoint(bundle), backend_name="x" * 257)),
    ):
        case_store = BrainStateStore.start(tmp_path / name)
        try:
            outcome = case_store.preflight_allocate(session_key, event_key)
            assert isinstance(outcome, EventAllocated)
            candidate, candidate_receipt = _event_material(outcome)
            with pytest.raises(
                BrainValidationError,
                match="token|generation|acknowledged|64KB|UTF-8|backend_name",
            ):
                case_store.commit_event(
                    session_key,
                    event_key,
                    EventCommit(outcome, candidate, candidate_receipt, bad),
                )
            assert case_store.load(session_key) == SessionLoaded(outcome.bundle, None)
        finally:
            case_store.close()


def test_same_id_race_becomes_committed_plus_duplicate_and_restart_is_durable(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    event_key = event_id_digest("same")
    first = store.preflight_allocate(session_key, event_key)
    second = store.preflight_allocate(session_key, event_key)
    assert isinstance(first, EventAllocated) and isinstance(second, EventAllocated)
    bundle, receipt = _event_material(first)
    result = store.commit_event(session_key, event_key, EventCommit(first, bundle, receipt))
    _assert_event_committed(result, receipt, session_key, event_key)
    other_bundle, other_receipt = _event_material(second)
    assert store.commit_event(
        session_key, event_key, EventCommit(second, other_bundle, other_receipt)
    ) == EventDuplicate(receipt, bundle)
    store.close()
    reopened = BrainStateStore.start(tmp_path)
    try:
        assert reopened.lookup_event_receipt(session_key, event_key) == EventDuplicate(
            receipt, bundle
        )
        assert reopened.preflight_allocate(session_key, event_key) == EventDuplicate(
            receipt, bundle
        )
    finally:
        reopened.close()


def test_different_ids_use_latest_allocation_and_inverse_commit_fails_without_retry(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        event1_key = event_id_digest("e1")
        event2_key = event_id_digest("e2")
        first = store.preflight_allocate(session_key, event1_key)
        inverse = store.preflight_allocate(session_key, event2_key)
        assert isinstance(first, EventAllocated) and isinstance(inverse, EventAllocated)
        bundle, receipt = _event_material(first)
        store.commit_event(session_key, event1_key, EventCommit(first, bundle, receipt))
        latest = store.preflight_allocate(session_key, event_id_digest("e3"))
        assert isinstance(latest, EventAllocated)
        assert latest.allocation.mutation_seq == 2
        stale_bundle, stale_receipt = _event_material(inverse)
        with pytest.raises(BrainAllocationError, match="stale|CAS|mutation"):
            store.commit_event(
                session_key, event2_key, EventCommit(inverse, stale_bundle, stale_receipt)
            )
        assert store.load(session_key) == SessionLoaded(bundle, None)
    finally:
        store.close()


def test_feedback_typed_preflight_applied_cas_and_receipt_only_no_mutation(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        event_bundle, _ = _event_commit(store, "s", "e1")
        feedback1_key = feedback_id_digest("f1")
        allocated = store.preflight_feedback(session_key, feedback1_key, target_tick=1)
        assert isinstance(allocated, FeedbackAllocated)
        assert allocated.allocation == FeedbackAllocation(
            generation=event_bundle.b.generation,
            lineage_id=event_bundle.b.lineage_id,
            target_tick=1,
            expected_mutation_seq=1,
            next_mutation_seq=2,
        )
        bundle, receipt = _feedback_material(allocated)
        forged = replace(allocated, allocation=replace(allocated.allocation, next_mutation_seq=3))
        with pytest.raises(BrainAllocationError, match="allocation"):
            store.commit_feedback(
                session_key,
                feedback_id_digest("forged"),
                AppliedFeedbackCommit(forged, bundle, receipt),
            )
        result = store.commit_feedback(
            session_key, feedback1_key, AppliedFeedbackCommit(allocated, bundle, receipt)
        )
        _assert_feedback_committed(result, receipt, session_key, feedback1_key)
        assert store.load(session_key) == SessionLoaded(bundle, None)
        assert store.preflight_feedback(
            session_key, feedback1_key, target_tick=1
        ) == FeedbackDuplicate(receipt, bundle)

        no_effect = StoredReceipt(
            kind="feedback",
            status="no_effect",
            generation=bundle.b.generation,
            tick_id=bundle.b.tick_id,
            history_epoch=bundle.b.history_epoch,
            mutation_seq=bundle.b.mutation_seq,
            target_tick=1,
        )
        feedback2_key = feedback_id_digest("f2")
        result = store.commit_feedback(
            session_key,
            feedback2_key,
            ReceiptOnlyFeedbackCommit(no_effect),
        )
        _assert_feedback_committed(result, no_effect, session_key, feedback2_key)
        assert store.load(session_key) == SessionLoaded(bundle, None)
        assert (
            store.lookup_feedback_receipt(session_key, feedback_id_digest("missing"))
            == FeedbackMiss()
        )
        with pytest.raises(TypeError):
            ReceiptOnlyFeedbackCommit(no_effect, _checkpoint(bundle))
    finally:
        store.close()


def test_feedback_applied_commit_persists_degraded_terminal_status(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        feedback_key = feedback_id_digest("f")
        _event_commit(store, "s", "e")
        allocated = store.preflight_feedback(session_key, feedback_key, target_tick=1)
        assert isinstance(allocated, FeedbackAllocated)
        bundle, receipt = _feedback_material(allocated)
        degraded = replace(receipt, status="degraded")
        result = store.commit_feedback(
            session_key,
            feedback_key,
            AppliedFeedbackCommit(allocated, bundle, degraded),
        )
        _assert_feedback_committed(result, degraded, session_key, feedback_key)
    finally:
        store.close()


def test_feedback_receipt_only_persists_degraded_terminal_status(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        bundle, _ = _event_commit(store, "s", "e")
        degraded = StoredReceipt(
            kind="feedback",
            status="degraded",
            generation=bundle.b.generation,
            tick_id=bundle.b.tick_id,
            history_epoch=bundle.b.history_epoch,
            mutation_seq=bundle.b.mutation_seq,
            target_tick=1,
        )
        feedback_key = feedback_id_digest("f")
        result = store.commit_feedback(
            session_key,
            feedback_key,
            ReceiptOnlyFeedbackCommit(degraded),
        )
        _assert_feedback_committed(result, degraded, session_key, feedback_key)
    finally:
        store.close()


@pytest.mark.parametrize("status", ("no_effect", "missed", "degraded"))
def test_receipt_only_feedback_rejects_claimed_applied_dimensions(
    tmp_path: Path,
    status: str,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        bundle, _ = _event_commit(store, "s", "e")
        receipt = StoredReceipt(
            kind="feedback",
            status=cast(Any, status),
            generation=bundle.b.generation,
            tick_id=bundle.b.tick_id,
            history_epoch=bundle.b.history_epoch,
            mutation_seq=bundle.b.mutation_seq,
            target_tick=1,
            applied_dimensions=(0,),
        )
        feedback_key = feedback_id_digest("f")
        with pytest.raises(BrainAllocationError, match="dimensions"):
            store.commit_feedback(
                session_key,
                feedback_key,
                ReceiptOnlyFeedbackCommit(receipt),
            )
        assert store.load(session_key) == SessionLoaded(bundle, None)
        assert store.lookup_feedback_receipt(session_key, feedback_key) == FeedbackMiss()
    finally:
        store.close()


def test_feedback_stale_cas_and_counter_exhaustion_are_strict(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        feedback_key = feedback_id_digest("f1")
        _event_commit(store, "s", "e1")
        allocated = store.preflight_feedback(session_key, feedback_key, target_tick=1)
        assert isinstance(allocated, FeedbackAllocated)
        feedback_bundle, receipt = _feedback_material(allocated)
        latest, _ = _event_commit(store, "s", "e2")
        with pytest.raises(BrainAllocationError, match="stale|CAS|mutation"):
            store.commit_feedback(
                session_key,
                feedback_key,
                AppliedFeedbackCommit(allocated, feedback_bundle, receipt),
            )
        assert store.load(session_key) == SessionLoaded(latest, None)
        with pytest.raises(BrainAllocationError, match="future"):
            store.preflight_feedback(session_key, feedback_id_digest("future"), target_tick=99)
    finally:
        store.close()


def test_event_and_feedback_counter_exhaustion_precedes_allocation(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    outcome = store.preflight_allocate(session_key, event_id_digest("seed"))
    assert isinstance(outcome, EventAllocated)
    store.close()
    exhausted_b = BrainState(
        generation=outcome.bundle.b.generation,
        lineage_id=outcome.bundle.b.lineage_id,
        e=outcome.bundle.b.e,
        d_plus=outcome.bundle.b.d_plus,
        d_minus=outcome.bundle.b.d_minus,
        gain_b=outcome.bundle.b.gain_b,
        theta_b=outcome.bundle.b.theta_b,
        clock=outcome.bundle.b.clock,
        tick_id=outcome.bundle.b.tick_id,
        history_epoch=outcome.bundle.b.history_epoch,
        mutation_seq=MAX_COUNTER,
        eligibility_horizon=outcome.bundle.b.eligibility_horizon,
    )
    exhausted = BrainBundle(exhausted_b, outcome.bundle.c)
    blob, digest = encode_brain_bundle(exhausted)
    path = tmp_path / "brain" / "brain_state.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE brain_session SET mutation_seq=?,state_blob=?,state_sha256=? "
            "WHERE session_digest=?",
            (MAX_COUNTER, blob, digest, session_digest("s")),
        )
    before = _table_snapshot(path)
    reopened = BrainStateStore.start(tmp_path)
    try:
        with pytest.raises(BrainCounterExhaustedError):
            reopened.preflight_allocate(session_key, event_id_digest("event"))
        with pytest.raises(BrainCounterExhaustedError):
            reopened.preflight_feedback(session_key, feedback_id_digest("feedback"), target_tick=0)
    finally:
        reopened.close()
    assert _table_snapshot(path) == before


@pytest.mark.parametrize(
    "target",
    (
        "receipt",
        "receipt_duplicate_key",
        "receipt_recursion",
        "receipt_mutation",
        "checkpoint",
        "version",
    ),
)
def test_load_rejects_tampered_receipt_or_checkpoint_hash(tmp_path: Path, target: str) -> None:
    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    event_key = event_id_digest("e")
    allocated = store.preflight_allocate(session_key, event_key)
    assert isinstance(allocated, EventAllocated)
    bundle, receipt = _event_material(allocated)
    checkpoint = _checkpoint(bundle)
    store.commit_event(session_key, event_key, EventCommit(allocated, bundle, receipt, checkpoint))
    store.close()
    path = tmp_path / "brain" / "brain_state.sqlite3"
    with sqlite3.connect(path) as connection:
        if target == "receipt":
            connection.execute("UPDATE brain_dedup SET receipt_sha256=?", (b"0" * 32,))
        elif target == "receipt_duplicate_key":
            blob = connection.execute("SELECT receipt_blob FROM brain_dedup").fetchone()[0]
            changed = blob.replace(b"{", b'{"kind":"event",', 1)
            connection.execute(
                "UPDATE brain_dedup SET receipt_blob=?,receipt_sha256=?",
                (changed, hashlib.sha256(changed).digest()),
            )
        elif target == "receipt_recursion":
            changed = b"[" * 5_000 + b"0" + b"]" * 5_000
            connection.execute(
                "UPDATE brain_dedup SET receipt_blob=?,receipt_sha256=?",
                (changed, hashlib.sha256(changed).digest()),
            )
        elif target == "receipt_mutation":
            connection.execute("UPDATE brain_dedup SET mutation_seq=mutation_seq+1")
        elif target == "checkpoint":
            connection.execute("UPDATE brain_backend_checkpoint SET token_sha256=?", (b"0" * 32,))
        else:
            connection.execute("UPDATE brain_backend_checkpoint SET backend_state_version=-1")
    reopened = BrainStateStore.start(tmp_path)
    try:
        if target.startswith("receipt"):
            with pytest.raises(BrainDurabilityError, match="receipt"):
                reopened.lookup_event_receipt(session_key, event_key)
        else:
            with pytest.raises(BrainDurabilityError, match="checkpoint invariant"):
                reopened.load(session_key)
    finally:
        reopened.close()


def test_receipt_never_accepts_duplicate_as_a_stored_status() -> None:
    with pytest.raises(BrainValidationError, match="duplicate"):
        StoredReceipt(
            kind="event",
            status=cast(Any, "duplicate"),
            generation=0,
            tick_id=0,
            history_epoch=0,
            mutation_seq=0,
        )


def test_lookup_filters_two_hour_ttl_and_prunes_per_session_and_kind_deterministically(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path, dedup_horizon=2)
    try:
        for session in ("s1", "s2"):
            for index in range(4):
                _event_commit(store, session, f"e{index}")
            session_key = session_digest(session)
            current = store.load(session_key)
            assert isinstance(current, SessionLoaded)
            for index in range(4):
                receipt = StoredReceipt(
                    kind="feedback",
                    status="missed",
                    generation=current.bundle.b.generation,
                    tick_id=current.bundle.b.tick_id,
                    history_epoch=current.bundle.b.history_epoch,
                    mutation_seq=current.bundle.b.mutation_seq,
                    target_tick=0,
                )
                store.commit_feedback(
                    session_key,
                    feedback_id_digest(f"f{index}"),
                    ReceiptOnlyFeedbackCommit(receipt),
                )
        with sqlite3.connect(store.database_path) as connection:
            counts = connection.execute(
                "SELECT session_digest,kind,COUNT(*) FROM brain_dedup "
                "GROUP BY session_digest,kind ORDER BY session_digest,kind"
            ).fetchall()
            assert all(count == 2 for _, _, count in counts) and len(counts) == 4
            key = session_digest("s1")
            connection.execute(
                "UPDATE brain_dedup SET created_ns=? WHERE session_digest=? AND kind='event'",
                (time.time_ns() - DEDUP_TTL_NS - 1, key),
            )
        assert (
            store.lookup_event_receipt(session_digest("s1"), event_id_digest("e3")) == EventMiss()
        )
        assert isinstance(
            store.lookup_event_receipt(session_digest("s2"), event_id_digest("e3")),
            EventDuplicate,
        )
        with sqlite3.connect(store.database_path) as connection:
            remaining_expired = connection.execute(
                "SELECT COUNT(*) FROM brain_dedup WHERE session_digest=? AND kind='event'",
                (session_digest("s1"),),
            ).fetchone()[0]
        assert remaining_expired == 0

        # Equal timestamps retain the lexicographically larger raw digest first.
        with sqlite3.connect(store.database_path) as connection:
            key = session_digest("s2")
            old_digests = [
                row[0]
                for row in connection.execute(
                    "SELECT id_digest FROM brain_dedup WHERE session_digest=? AND kind='event'",
                    (key,),
                )
            ]
            connection.execute(
                "UPDATE brain_dedup SET created_ns=? WHERE session_digest=? AND kind='event'",
                (time.time_ns(), key),
            )
        _event_commit(store, "s2", "tie-trigger")
        with sqlite3.connect(store.database_path) as connection:
            survivors = [
                row[0]
                for row in connection.execute(
                    "SELECT id_digest FROM brain_dedup WHERE session_digest=? AND kind='event' "
                    "ORDER BY created_ns DESC,id_digest DESC",
                    (session_digest("s2"),),
                )
            ]
        assert survivors == [event_id_digest("tie-trigger"), max(old_digests)]
    finally:
        store.close()


def test_lookup_honors_a_smaller_horizon_after_restart(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path, dedup_horizon=4)
    try:
        for index in range(4):
            _event_commit(store, "s", f"e{index}")
    finally:
        store.close()

    now = time.time_ns()
    with sqlite3.connect(tmp_path / "brain" / "brain_state.sqlite3") as connection:
        for index in range(4):
            connection.execute(
                "UPDATE brain_dedup SET created_ns=? WHERE session_digest=? AND kind='event' "
                "AND id_digest=?",
                (now + index, session_digest("s"), event_id_digest(f"e{index}")),
            )

    reopened = BrainStateStore.start(tmp_path, dedup_horizon=2)
    try:
        session_key = session_digest("s")
        assert isinstance(
            reopened.lookup_event_receipt(session_key, event_id_digest("e3")),
            EventDuplicate,
        )
        assert isinstance(
            reopened.lookup_event_receipt(session_key, event_id_digest("e2")),
            EventDuplicate,
        )
        assert reopened.lookup_event_receipt(session_key, event_id_digest("e1")) == EventMiss()
        assert reopened.lookup_event_receipt(session_key, event_id_digest("e0")) == EventMiss()
        with sqlite3.connect(reopened.database_path) as connection:
            retained = connection.execute(
                "SELECT COUNT(*) FROM brain_dedup WHERE session_digest=? AND kind='event'",
                (session_key,),
            ).fetchone()[0]
        assert retained == 2
    finally:
        reopened.close()


def test_expired_identifier_can_be_reused_without_old_primary_key_collision(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        _event_commit(store, "s", "same")
        with sqlite3.connect(store.database_path) as connection:
            connection.execute(
                "UPDATE brain_dedup SET created_ns=? WHERE session_digest=? AND kind='event'",
                (time.time_ns() - DEDUP_TTL_NS - 1, session_digest("s")),
            )
        session_key = session_digest("s")
        event_key = event_id_digest("same")
        allocated = store.preflight_allocate(session_key, event_key)
        assert isinstance(allocated, EventAllocated)
        bundle, receipt = _event_material(allocated)
        result = store.commit_event(session_key, event_key, EventCommit(allocated, bundle, receipt))
        _assert_event_committed(result, receipt, session_key, event_key)
    finally:
        store.close()


def test_load_rejects_column_blob_control_cross_invariant_mismatch(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    _event_commit(store, "s", "e")
    store.close()
    path = tmp_path / "brain" / "brain_state.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE brain_session SET mutation_seq=mutation_seq+1 WHERE session_digest=?",
            (session_digest("s"),),
        )
    reopened = BrainStateStore.start(tmp_path)
    try:
        with pytest.raises(BrainDurabilityError, match="invariant|mutation"):
            reopened.load(session_digest("s"))
    finally:
        reopened.close()


def test_reset_destroy_expected_generation_idempotency_and_tombstone(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        bundle, _ = _event_commit(store, "s", "e", checkpoint=None)
        assert store.runtime_reference_matches(
            session_key,
            generation=bundle.b.generation,
            lineage_id=bundle.b.lineage_id,
            mutation_seq=bundle.b.mutation_seq,
        )
        assert not store.runtime_reference_matches(
            session_key,
            generation=bundle.b.generation,
            lineage_id=bundle.b.lineage_id,
            mutation_seq=0,
        )
        with pytest.raises(BrainAllocationError, match="generation"):
            store.reset(session_key, expected_generation=9)
        reset = store.reset(session_key, expected_generation=bundle.b.generation)
        assert reset.b.generation == bundle.b.generation + 1
        assert not store.runtime_reference_matches(
            session_key,
            generation=bundle.b.generation,
            lineage_id=bundle.b.lineage_id,
            mutation_seq=bundle.b.mutation_seq,
        )
        assert store.runtime_reference_matches(
            session_key,
            generation=reset.b.generation,
            lineage_id=reset.b.lineage_id,
            mutation_seq=0,
        )
        with pytest.raises(BrainAllocationError, match="generation"):
            store.destroy(session_key, expected_generation=0)
        destroyed = store.destroy(session_key, expected_generation=reset.b.generation)
        assert store.destroy(session_key, expected_generation=destroyed) == destroyed
        assert store.load(session_key) == SessionMissing()
        assert not store.runtime_reference_matches(
            session_key,
            generation=reset.b.generation,
            lineage_id=reset.b.lineage_id,
            mutation_seq=0,
        )
        with pytest.raises(BrainDurabilityError, match="destroyed"):
            store.preflight_allocate(session_key, event_id_digest("lazy"))
        assert _table_snapshot(store.database_path)["brain_session"] == []
    finally:
        store.close()


def test_idempotent_destroy_cleans_any_residual_auxiliary_rows(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        session_key = session_digest("s")
        bundle, _ = _event_commit(store, "s", "e")
        destroyed = store.destroy(session_key, expected_generation=bundle.b.generation)
        with sqlite3.connect(store.database_path) as connection:
            key = session_digest("s")
            connection.execute(
                "INSERT INTO brain_dedup(session_digest,kind,id_digest,mutation_seq,receipt_blob,"
                "receipt_sha256,created_ns) VALUES(?,?,?,?,?,?,?)",
                (
                    key,
                    "event",
                    event_id_digest("residual"),
                    0,
                    b"{}",
                    hashlib.sha256(b"{}").digest(),
                    0,
                ),
            )
            connection.execute(
                "INSERT INTO brain_backend_checkpoint(session_digest,generation,backend_name,"
                "backend_state_version,acknowledged_mutation_seq,token_blob,token_sha256) "
                "VALUES(?,?,?,?,?,?,?)",
                (key, destroyed, "residual", 0, 0, b"", hashlib.sha256(b"").digest()),
            )
        assert store.destroy(session_key, expected_generation=destroyed) == destroyed
        snapshot = _table_snapshot(store.database_path)
        assert snapshot["brain_dedup"] == []
        assert snapshot["brain_backend_checkpoint"] == []
    finally:
        store.close()


def test_generation_exhaustion_leaves_all_four_tables_byte_for_byte_unchanged(
    tmp_path: Path,
) -> None:
    store = BrainStateStore.start(tmp_path)
    _event_commit(store, "s", "e")
    store.close()
    path = tmp_path / "brain" / "brain_state.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE brain_control SET generation=? WHERE session_digest=?",
            (MAX_COUNTER, session_digest("s")),
        )
    before = _table_snapshot(path)
    reopened = BrainStateStore.start(tmp_path)
    try:
        with pytest.raises(BrainCounterExhaustedError):
            reopened.reset(session_digest("s"), expected_generation=MAX_COUNTER)
        with pytest.raises(BrainCounterExhaustedError):
            reopened.destroy(session_digest("s"), expected_generation=MAX_COUNTER)
    finally:
        reopened.close()
    assert _table_snapshot(path) == before


def test_candidate_encoding_runs_outside_write_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    event_key = event_id_digest("e")
    allocated = store.preflight_allocate(session_key, event_key)
    assert isinstance(allocated, EventAllocated)
    bundle, receipt = _event_material(allocated)
    real_encode = encode_brain_bundle
    observations: list[bool] = []

    def observing_encode(candidate: BrainBundle) -> tuple[bytes, bytes]:
        observations.append(store._debug_owner_in_transaction())
        return real_encode(candidate)

    monkeypatch.setattr(store_module, "encode_brain_bundle", observing_encode)
    try:
        store.commit_event(session_key, event_key, EventCommit(allocated, bundle, receipt))
        assert observations == [False]
    finally:
        store.close()


def test_process_cross_process_and_canonical_alias_leases_release(tmp_path: Path) -> None:
    store = BrainStateStore.start(tmp_path)
    try:
        with pytest.raises(BrainDurabilityError, match="lease|owner"):
            BrainStateStore.start(tmp_path / ".")
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "lease", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_child_environment(),
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert "LEASE_REFUSED" in child.stdout
    finally:
        store.close()
    BrainStateStore.start(tmp_path).close()


def test_cancelled_command_future_does_not_kill_worker_and_close_is_concurrent_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = BrainStateStore._execute_command

    def slow(self: BrainStateStore, connection: sqlite3.Connection, command: object) -> object:
        if cast(Any, command).operation == "load":
            entered.set()
            release.wait(5)
        return original(self, connection, command)

    monkeypatch.setattr(BrainStateStore, "_execute_command", slow)
    store = BrainStateStore.start(tmp_path)
    try:
        blocker, _, blocker_errors = _thread_call(lambda: store.load(session_digest("blocker")))
        assert entered.wait(5)
        command = store._create_command("load", (session_digest("cancelled"),), {})
        store._queue.put_nowait(command)
        assert command.future.cancel()
        release.set()
        blocker.join(10)
        assert not blocker.is_alive() and not blocker_errors
        _wait(lambda: command.future.done())
        assert store._thread.is_alive()
        assert store.load(session_digest("still-alive")) == SessionMissing()
        closers = [_thread_call(store.close) for _ in range(4)]
        for thread, results, errors in closers:
            thread.join(10)
            assert not thread.is_alive() and results == [None] and not errors
        assert not store._thread.is_alive()
    finally:
        store.close()


def test_close_waits_for_admission_registered_before_queue_put(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = threading.Event()
    release = threading.Event()
    original = BrainStateStore._create_command

    def pause_after_create(
        self: BrainStateStore,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        closing: bool = False,
        admission: bool = False,
    ) -> object:
        command = original(
            self,
            operation,
            args,
            kwargs,
            closing=closing,
            admission=admission,
        )
        if operation == "load":
            created.set()
            release.wait(5)
        return command

    monkeypatch.setattr(BrainStateStore, "_create_command", pause_after_create)
    store = BrainStateStore.start(tmp_path)
    caller, caller_results, caller_errors = _thread_call(
        lambda: store.load(session_digest("admission"))
    )
    assert created.wait(5)
    closer, close_results, close_errors = _thread_call(store.close)
    closer.join(0.5)
    close_returned_early = not closer.is_alive()
    release.set()
    if close_returned_early:
        _wait(lambda: store._queue.qsize() >= 1)
        store._mark_failed(BrainDurabilityError("test cleanup after premature close"))
    caller.join(5)
    closer.join(5)
    assert not close_returned_early
    assert caller_results == [SessionMissing()] and not caller_errors
    assert close_results == [None] and not close_errors
    assert not caller.is_alive() and not closer.is_alive()


def test_close_join_timeout_does_not_claim_closed_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BrainStateStore.start(tmp_path)
    join_calls = 0
    alive_checks = 0

    def fake_join(_timeout: float | None = None) -> None:
        nonlocal join_calls
        join_calls += 1

    def fake_is_alive() -> bool:
        nonlocal alive_checks
        alive_checks += 1
        return alive_checks <= 2

    monkeypatch.setattr(store._thread, "join", fake_join)
    monkeypatch.setattr(store._thread, "is_alive", fake_is_alive)

    with pytest.raises(BrainDurabilityError, match="did not terminate"):
        store.close()
    assert not store._closed and not store._closing

    store.close()
    assert store._closed and not store._closing
    assert join_calls == 2


def test_real_cleanup_stall_reuses_close_request_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    original_release = BrainStateStore._release_lease
    real_join = threading.Thread.join
    join_calls = 0

    store = BrainStateStore.start(tmp_path)

    def blocked_release(self: BrainStateStore) -> None:
        if self is store:
            cleanup_entered.set()
            release_cleanup.wait(5)
        original_release(self)

    def controlled_join(_timeout: float | None = None) -> None:
        nonlocal join_calls
        join_calls += 1
        real_join(store._thread, 0.05 if join_calls == 1 else 2.0)

    monkeypatch.setattr(BrainStateStore, "_release_lease", blocked_release)
    monkeypatch.setattr(store._thread, "join", controlled_join)
    with pytest.raises(BrainDurabilityError, match="did not terminate"):
        store.close()
    assert cleanup_entered.is_set() and store._thread.is_alive()

    retry, retry_results, retry_errors = _thread_call(store.close)
    time.sleep(0.1)
    release_cleanup.set()
    retry.join(1)
    retry_hung = retry.is_alive()
    if retry_hung:
        store._mark_failed(BrainDurabilityError("test cleanup after duplicate close request"))
        retry.join(1)
    assert not retry_hung
    assert retry_results == [None] and not retry_errors
    assert not store._thread.is_alive()


def test_normal_close_drains_already_accepted_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = BrainStateStore._execute_command

    def slow(self: BrainStateStore, connection: sqlite3.Connection, command: object) -> object:
        if cast(Any, command).operation == "load":
            entered.set()
            release.wait(5)
        return original(self, connection, command)

    monkeypatch.setattr(BrainStateStore, "_execute_command", slow)
    store = BrainStateStore.start(tmp_path)
    first, first_results, first_errors = _thread_call(lambda: store.load(session_digest("first")))
    assert entered.wait(5)
    second, second_results, second_errors = _thread_call(
        lambda: store.load(session_digest("second"))
    )
    _wait(lambda: store._queue.qsize() >= 1)
    closer, close_results, close_errors = _thread_call(store.close)
    release.set()
    for thread in (first, second, closer):
        thread.join(10)
        assert not thread.is_alive()
    assert first_results == [SessionMissing()] and not first_errors
    assert second_results == [SessionMissing()] and not second_errors
    assert close_results == [None] and not close_errors
    assert not store._thread.is_alive()


def test_baseexception_inside_transaction_rolls_back_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FatalCommit(BaseException):
        pass

    store = BrainStateStore.start(tmp_path)
    session_key = session_digest("s")
    event_key = event_id_digest("e")
    allocated = store.preflight_allocate(session_key, event_key)
    assert isinstance(allocated, EventAllocated)
    bundle, receipt = _event_material(allocated)
    original = BrainStateStore._commit_transaction

    def fatal_commit(_self: BrainStateStore, _connection: sqlite3.Connection) -> None:
        raise FatalCommit("transaction owner died")

    monkeypatch.setattr(BrainStateStore, "_commit_transaction", fatal_commit)
    with pytest.raises(BrainDurabilityError, match="FatalCommit"):
        store.commit_event(session_key, event_key, EventCommit(allocated, bundle, receipt))
    store.close()
    assert not store._thread.is_alive()
    monkeypatch.setattr(BrainStateStore, "_commit_transaction", original)
    reopened = BrainStateStore.start(tmp_path)
    try:
        assert reopened.load(session_key) == SessionLoaded(allocated.bundle, None)
    finally:
        reopened.close()


def test_sqlite_command_error_is_normalized_and_stops_the_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BrainStateStore._execute_command

    def corrupt(
        _self: BrainStateStore,
        _connection: sqlite3.Connection,
        _command: object,
    ) -> object:
        raise sqlite3.DatabaseError("database image is malformed")

    store = BrainStateStore.start(tmp_path)
    monkeypatch.setattr(BrainStateStore, "_execute_command", corrupt)
    try:
        with pytest.raises(BrainDurabilityError, match="DatabaseError"):
            store.load(session_digest("corrupt"))
        assert store._closed_event.wait(5)
        assert not store._thread.is_alive()
        with pytest.raises(BrainDurabilityError, match="DatabaseError"):
            store.load(session_digest("after-failure"))
    finally:
        store.close()

    monkeypatch.setattr(BrainStateStore, "_execute_command", original)
    BrainStateStore.start(tmp_path).close()


def test_abnormal_owner_fails_current_queued_and_capacity_waiters_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FatalExit(BaseException):
        pass

    entered = threading.Event()
    release = threading.Event()

    def fatal(_self: BrainStateStore, _connection: sqlite3.Connection, _command: object) -> object:
        entered.set()
        release.wait(5)
        raise FatalExit("owner died")

    monkeypatch.setattr(BrainStateStore, "_execute_command", fatal)
    store = BrainStateStore.start(tmp_path)
    current, _, current_errors = _thread_call(lambda: store.load(session_digest("current")))
    assert entered.wait(5)
    queued = [store._create_command("load", (f"q{i}",), {}) for i in range(1024)]
    for command in queued:
        store._queue.put_nowait(command)
    waiter, _, waiter_errors = _thread_call(lambda: store.load(session_digest("capacity")))
    time.sleep(0.1)
    release.set()
    current.join(10)
    waiter.join(10)
    assert isinstance(current_errors[0], BrainDurabilityError)
    assert isinstance(waiter_errors[0], BrainDurabilityError)
    for command in queued:
        with pytest.raises(BrainDurabilityError):
            command.future.result(timeout=1)
    store.close()
    assert not store._thread.is_alive()
    monkeypatch.setattr(BrainStateStore, "_execute_command", store_module._ORIGINAL_EXECUTE)
    BrainStateStore.start(tmp_path).close()


@pytest.mark.parametrize(
    ("phase", "expected_mutation"),
    (
        ("after_state", 0),
        ("after_dedup", 0),
        ("after_checkpoint", 0),
        ("before_commit", 0),
        ("after_commit", 1),
    ),
)
def test_real_os_exit_crash_windows_are_only_old_or_new_complete_state(
    tmp_path: Path, phase: str, expected_mutation: int
) -> None:
    data_dir = tmp_path / phase
    store = BrainStateStore.start(data_dir)
    session_key = session_digest("crash")
    initial = store.preflight_allocate(session_key, event_id_digest("probe"))
    assert isinstance(initial, EventAllocated)
    store.close()
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "crash", str(data_dir), phase],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env=_child_environment(),
    )
    assert child.returncode in (91, 92), child.stdout + child.stderr
    reopened = BrainStateStore.start(data_dir)
    try:
        loaded = reopened.load(session_key)
        assert isinstance(loaded, SessionLoaded)
        assert loaded.bundle.b.mutation_seq == expected_mutation
        if expected_mutation == 0:
            assert loaded == SessionLoaded(initial.bundle, None)
            assert reopened.lookup_event_receipt(session_key, event_id_digest("e1")) == EventMiss()
        else:
            assert loaded.bundle.b.tick_id == loaded.bundle.b.history_epoch == 1
            assert loaded.bundle.c.eligibility_records[-1].tick_id == 1
            assert loaded.checkpoint is not None
            assert loaded.checkpoint.acknowledged_mutation_seq == 1
            assert isinstance(
                reopened.lookup_event_receipt(session_key, event_id_digest("e1")),
                EventDuplicate,
            )
    finally:
        reopened.close()


def _child_main(arguments: list[str]) -> int:
    mode = arguments[1]
    data_dir = Path(arguments[2])
    if mode == "lease":
        try:
            store = BrainStateStore.start(data_dir)
        except BrainDurabilityError:
            print("LEASE_REFUSED", flush=True)
            return 0
        store.close()
        return 2
    if mode == "crash":
        phase = arguments[3]
        store = BrainStateStore.start(data_dir)
        session_key = session_digest("crash")
        event_key = event_id_digest("e1")
        outcome = store.preflight_allocate(session_key, event_key)
        assert isinstance(outcome, EventAllocated)
        bundle, receipt = _event_material(outcome)
        commit = EventCommit(outcome, bundle, receipt, _checkpoint(bundle, b"crash-token"))
        if phase == "after_state":
            original = BrainStateStore._write_session

            def crash(self: BrainStateStore, connection: sqlite3.Connection, *args: object) -> None:
                original(self, connection, *args)
                os._exit(91)

            BrainStateStore._write_session = crash
        elif phase == "after_dedup":
            original = BrainStateStore._write_dedup

            def crash(self: BrainStateStore, connection: sqlite3.Connection, *args: object) -> None:
                original(self, connection, *args)
                os._exit(91)

            BrainStateStore._write_dedup = crash
        elif phase == "after_checkpoint":
            original = BrainStateStore._write_checkpoint

            def crash(self: BrainStateStore, connection: sqlite3.Connection, *args: object) -> None:
                original(self, connection, *args)
                os._exit(91)

            BrainStateStore._write_checkpoint = crash
        else:
            original = BrainStateStore._commit_transaction

            def crash(self: BrainStateStore, connection: sqlite3.Connection) -> None:
                if phase == "before_commit":
                    os._exit(91)
                original(self, connection)
                os._exit(92)

            BrainStateStore._commit_transaction = crash
        store.commit_event(session_key, event_key, commit)
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv))
