"""Single-owner SQLite persistence for authoritative brain state."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future, InvalidStateError
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, cast
from uuid import uuid4

from .brain_c import BrainCState
from .brain_codec import (
    BRAIN_STATE_SCHEMA_VERSION,
    SUPPORTED_BRAIN_STATE_SCHEMA_VERSIONS,
    BrainBundle,
    decode_brain_bundle,
    encode_brain_bundle,
)
from .brain_errors import (
    BrainAllocationError,
    BrainCounterExhaustedError,
    BrainDurabilityError,
    BrainValidationError,
)
from .brain_state import MAX_COUNTER, BrainState, EventAllocation, FeedbackAllocation

DEDUP_TTL_NS = 2 * 60 * 60 * 1_000_000_000
MAX_CHECKPOINT_BYTES = 64 * 1024
MAX_BACKEND_NAME_BYTES = 256
QUEUE_CAPACITY = 1024
_QUEUE_POLL_SECONDS = 0.05

_REQUIRED_PRAGMAS: dict[str, object] = {
    "journal_mode": "wal",
    "synchronous": 2,
    "foreign_keys": 1,
    "busy_timeout": 5000,
}

_PROCESS_OWNERS: set[Path] = set()
_PROCESS_OWNERS_LOCK = threading.Lock()

_RECEIPT_FIELDS = frozenset(
    {
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
)


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


class _MsvcrtModule(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, file_descriptor: int, mode: int, nbytes: int) -> None: ...


def _fcntl_module() -> _FcntlModule:
    return cast(_FcntlModule, importlib.import_module("fcntl"))


def _msvcrt_module() -> _MsvcrtModule:
    return cast(_MsvcrtModule, importlib.import_module("msvcrt"))


def _utf8(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise BrainValidationError(f"{name} must be a string")
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise BrainValidationError(f"{name} must be valid UTF-8 text") from error


def session_digest(session_id: str) -> bytes:
    return hashlib.sha256(_utf8(session_id, "session_id")).digest()


def event_id_digest(event_id: str) -> bytes:
    return hashlib.sha256(b"event\x00" + _utf8(event_id, "event_id")).digest()


def feedback_id_digest(feedback_id: str) -> bytes:
    return hashlib.sha256(b"feedback\x00" + _utf8(feedback_id, "feedback_id")).digest()


def _raw_digest(name: str, value: object) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise BrainValidationError(f"{name} must be raw 32 bytes")
    return value


def _counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BrainValidationError(f"{name} must be a non-boolean counter")
    if not 0 <= value <= MAX_COUNTER:
        raise BrainValidationError(f"{name} is outside the persisted counter domain")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BrainValidationError(f"{name} must be a positive non-boolean integer")
    return value


@dataclass(frozen=True, slots=True)
class StoredReceipt:
    kind: Literal["event", "feedback"]
    status: Literal["applied", "missed", "no_effect", "degraded", "disabled"]
    generation: int
    tick_id: int
    history_epoch: int
    mutation_seq: int
    target_tick: int | None = None
    applied_dimensions: tuple[int, ...] = ()
    applied_synapses: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("event", "feedback"):
            raise BrainValidationError("receipt kind is invalid")
        if self.status not in ("applied", "missed", "no_effect", "degraded", "disabled"):
            raise BrainValidationError("receipt status is invalid; duplicate is not stored")
        _counter("receipt generation", self.generation)
        _counter("receipt tick_id", self.tick_id)
        _counter("receipt history_epoch", self.history_epoch)
        _counter("receipt mutation_seq", self.mutation_seq)
        if self.target_tick is not None:
            _counter("receipt target_tick", self.target_tick)
        if self.kind == "event" and self.target_tick is not None:
            raise BrainValidationError("event receipt must not contain target_tick")
        if self.kind == "feedback" and self.target_tick is None:
            raise BrainValidationError("feedback receipt requires target_tick")
        if not isinstance(self.applied_dimensions, tuple):
            raise BrainValidationError("applied_dimensions must be a tuple")
        previous = -1
        for dimension in self.applied_dimensions:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or not 0 <= dimension < 8
            ):
                raise BrainValidationError(
                    "applied_dimensions must contain ordered indices in [0, 7]"
                )
            if dimension <= previous:
                raise BrainValidationError("applied_dimensions must be strictly ordered")
            previous = dimension
        if (
            isinstance(self.applied_synapses, bool)
            or not isinstance(self.applied_synapses, int)
            or not 0 <= self.applied_synapses <= 128
        ):
            raise BrainValidationError("applied_synapses must be an integer in [0, 128]")
        if self.kind == "event" and self.applied_synapses != 0:
            raise BrainValidationError("event receipt must not contain applied synapses")


@dataclass(frozen=True, slots=True)
class BackendCheckpoint:
    generation: int
    backend_name: str
    backend_state_version: int
    acknowledged_mutation_seq: int
    token: bytes
    token_sha256: bytes


@dataclass(frozen=True, slots=True)
class SessionMissing:
    pass


@dataclass(frozen=True, slots=True)
class SessionLoaded:
    bundle: BrainBundle
    checkpoint: BackendCheckpoint | None


@dataclass(frozen=True, slots=True)
class SessionControl:
    generation: int
    status: Literal["active", "destroyed"]

    def __post_init__(self) -> None:
        _counter("control generation", self.generation)
        if self.status not in ("active", "destroyed"):
            raise BrainValidationError("control status is invalid")


@dataclass(frozen=True, slots=True)
class EventMiss:
    pass


@dataclass(frozen=True, slots=True)
class EventDuplicate:
    receipt: StoredReceipt
    bundle: BrainBundle


@dataclass(frozen=True, slots=True)
class EventAllocated:
    bundle: BrainBundle
    allocation: EventAllocation


@dataclass(frozen=True, slots=True)
class EventCommit:
    allocated: EventAllocated
    bundle: BrainBundle
    receipt: StoredReceipt
    checkpoint: BackendCheckpoint | None = None


@dataclass(frozen=True, slots=True, init=False)
class EventCommitted:
    receipt: StoredReceipt
    session_digest: bytes
    id_digest: bytes
    checkpoint_token_sha256: bytes | None
    __seal: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("EventCommitted values are issued only by BrainStateStore")


@dataclass(frozen=True, slots=True)
class FeedbackMiss:
    pass


@dataclass(frozen=True, slots=True)
class FeedbackDuplicate:
    receipt: StoredReceipt
    bundle: BrainBundle


@dataclass(frozen=True, slots=True)
class FeedbackAllocated:
    bundle: BrainBundle
    allocation: FeedbackAllocation


@dataclass(frozen=True, slots=True)
class AppliedFeedbackCommit:
    allocated: FeedbackAllocated
    bundle: BrainBundle
    receipt: StoredReceipt
    checkpoint: BackendCheckpoint | None = None


@dataclass(frozen=True, slots=True)
class ReceiptOnlyFeedbackCommit:
    receipt: StoredReceipt


@dataclass(frozen=True, slots=True, init=False)
class FeedbackCommitted:
    receipt: StoredReceipt
    session_digest: bytes
    id_digest: bytes
    checkpoint_token_sha256: bytes | None
    __seal: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("FeedbackCommitted values are issued only by BrainStateStore")


class _AcknowledgementAuthority:
    __slots__ = ("__key",)

    def __init__(self) -> None:
        self.__key = os.urandom(32)

    def __payload_seal(
        self,
        acknowledgement_type: type[EventCommitted] | type[FeedbackCommitted],
        receipt: StoredReceipt,
        session_key: bytes,
        identifier: bytes,
        checkpoint_token_sha256: bytes | None,
    ) -> bytes:
        if acknowledgement_type is EventCommitted:
            type_tag = b"event"
        elif acknowledgement_type is FeedbackCommitted:
            type_tag = b"feedback"
        else:  # pragma: no cover - internal callers use the closed pair above
            raise BrainValidationError("store acknowledgement type is invalid")
        session = _raw_digest("session_digest", session_key)
        id_digest = _raw_digest("id_digest", identifier)
        if checkpoint_token_sha256 is None:
            checkpoint = b"\x00"
        else:
            checkpoint = b"\x01" + _raw_digest("checkpoint_token_sha256", checkpoint_token_sha256)
        receipt_blob, _ = _receipt_bytes(receipt)
        digest = hashlib.blake2b(
            digest_size=32,
            key=self.__key,
            person=b"SylannAck.v1",
        )
        digest.update(type_tag)
        digest.update(len(receipt_blob).to_bytes(8, "big"))
        digest.update(receipt_blob)
        digest.update(session)
        digest.update(id_digest)
        digest.update(checkpoint)
        return digest.digest()

    def issue(
        self,
        store: BrainStateStore,
        acknowledgement_type: type[EventCommitted] | type[FeedbackCommitted],
        receipt: StoredReceipt,
        session_key: bytes,
        identifier: bytes,
        checkpoint: BackendCheckpoint | None,
    ) -> EventCommitted | FeedbackCommitted:
        if (
            type(store) is not BrainStateStore
            or threading.current_thread() is not store._thread
            or store._owner_connection is None
        ):
            raise BrainDurabilityError(
                "store acknowledgements can only be issued by the active owner thread"
            )
        session = _raw_digest("session_digest", session_key)
        id_digest = _raw_digest("id_digest", identifier)
        checkpoint_digest = None if checkpoint is None else checkpoint.token_sha256
        acknowledgement = acknowledgement_type.__new__(acknowledgement_type)
        object.__setattr__(acknowledgement, "receipt", receipt)
        object.__setattr__(acknowledgement, "session_digest", session)
        object.__setattr__(acknowledgement, "id_digest", id_digest)
        object.__setattr__(
            acknowledgement,
            "checkpoint_token_sha256",
            checkpoint_digest,
        )
        storage_name = f"_{acknowledgement_type.__name__}__seal"
        object.__setattr__(
            acknowledgement,
            storage_name,
            self.__payload_seal(
                acknowledgement_type,
                receipt,
                session,
                id_digest,
                checkpoint_digest,
            ),
        )
        return acknowledgement

    def is_authentic(self, value: object) -> bool:
        acknowledgement_type: type[EventCommitted] | type[FeedbackCommitted]
        if type(value) is EventCommitted:
            acknowledgement_type = EventCommitted
            storage_name = "_EventCommitted__seal"
        elif type(value) is FeedbackCommitted:
            acknowledgement_type = FeedbackCommitted
            storage_name = "_FeedbackCommitted__seal"
        else:
            return False
        try:
            actual = object.__getattribute__(value, storage_name)
            expected = self.__payload_seal(
                acknowledgement_type,
                object.__getattribute__(value, "receipt"),
                object.__getattribute__(value, "session_digest"),
                object.__getattribute__(value, "id_digest"),
                object.__getattribute__(value, "checkpoint_token_sha256"),
            )
            return (
                type(actual) is bytes
                and len(actual) == 32
                and hmac.compare_digest(actual, expected)
            )
        except Exception:
            return False


_ACKNOWLEDGEMENT_AUTHORITY = _AcknowledgementAuthority()


def _is_authentic_store_acknowledgement(value: object) -> bool:
    return _ACKNOWLEDGEMENT_AUTHORITY.is_authentic(value)


@dataclass(frozen=True, slots=True)
class _PreparedEvent:
    allocated: EventAllocated
    bundle: BrainBundle
    state_blob: bytes
    state_digest: bytes
    receipt: StoredReceipt
    receipt_blob: bytes
    receipt_digest: bytes
    checkpoint: BackendCheckpoint | None


@dataclass(frozen=True, slots=True)
class _PreparedAppliedFeedback:
    allocated: FeedbackAllocated
    bundle: BrainBundle
    state_blob: bytes
    state_digest: bytes
    receipt: StoredReceipt
    receipt_blob: bytes
    receipt_digest: bytes
    checkpoint: BackendCheckpoint | None


@dataclass(frozen=True, slots=True)
class _PreparedReceiptOnly:
    receipt: StoredReceipt
    receipt_blob: bytes
    receipt_digest: bytes


@dataclass(frozen=True, slots=True)
class _Command:
    operation: str
    args: tuple[object, ...]
    kwargs: dict[str, object]
    future: Future[Any]


def _receipt_bytes(receipt: StoredReceipt) -> tuple[bytes, bytes]:
    if not isinstance(receipt, StoredReceipt):
        raise BrainValidationError("receipt must be a StoredReceipt")
    document = {
        "applied_dimensions": list(receipt.applied_dimensions),
        "applied_synapses": receipt.applied_synapses,
        "generation": receipt.generation,
        "history_epoch": receipt.history_epoch,
        "kind": receipt.kind,
        "mutation_seq": receipt.mutation_seq,
        "status": receipt.status,
        "target_tick": receipt.target_tick,
        "tick_id": receipt.tick_id,
    }
    blob = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return blob, hashlib.sha256(blob).digest()


def _unique_receipt_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise BrainDurabilityError(f"receipt contains duplicate key: {key}")
        document[key] = value
    return document


def _decode_receipt(blob: object, digest: object, expected_kind: str) -> StoredReceipt:
    if not isinstance(blob, bytes) or not isinstance(digest, bytes) or len(digest) != 32:
        raise BrainDurabilityError("receipt blob/checksum invariant failed")
    if not hmac.compare_digest(hashlib.sha256(blob).digest(), digest):
        raise BrainDurabilityError("receipt checksum mismatch")
    try:
        raw = json.loads(blob, object_pairs_hook=_unique_receipt_object)
    except BrainDurabilityError:
        raise
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise BrainDurabilityError("receipt JSON is invalid") from error
    if not isinstance(raw, dict) or frozenset(raw) != _RECEIPT_FIELDS:
        raise BrainDurabilityError("receipt allowlist invariant failed")
    document = cast(dict[str, object], raw)
    try:
        receipt = StoredReceipt(
            kind=cast(Any, document["kind"]),
            status=cast(Any, document["status"]),
            generation=cast(Any, document["generation"]),
            tick_id=cast(Any, document["tick_id"]),
            history_epoch=cast(Any, document["history_epoch"]),
            mutation_seq=cast(Any, document["mutation_seq"]),
            target_tick=cast(Any, document["target_tick"]),
            applied_dimensions=tuple(cast(Any, document["applied_dimensions"])),
            applied_synapses=cast(Any, document["applied_synapses"]),
        )
    except (BrainValidationError, TypeError) as error:
        raise BrainDurabilityError("receipt domain invariant failed") from error
    if receipt.kind != expected_kind:
        raise BrainDurabilityError("receipt kind invariant failed")
    return receipt


def _validate_checkpoint(
    checkpoint: BackendCheckpoint | None,
    bundle: BrainBundle,
) -> BackendCheckpoint | None:
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, BackendCheckpoint):
        raise BrainValidationError("checkpoint must be a BackendCheckpoint")
    _validate_checkpoint_shape(checkpoint)
    if checkpoint.generation != bundle.b.generation:
        raise BrainValidationError("checkpoint generation does not match bundle")
    if checkpoint.acknowledged_mutation_seq != bundle.b.mutation_seq:
        raise BrainValidationError("checkpoint acknowledged mutation does not match bundle")
    return checkpoint


def _validate_checkpoint_shape(checkpoint: BackendCheckpoint) -> None:
    _counter("checkpoint generation", checkpoint.generation)
    _counter("backend_state_version", checkpoint.backend_state_version)
    _counter("acknowledged_mutation_seq", checkpoint.acknowledged_mutation_seq)
    backend_name = _utf8(checkpoint.backend_name, "checkpoint backend_name")
    if not backend_name:
        raise BrainValidationError("checkpoint backend_name must be nonempty")
    if len(backend_name) > MAX_BACKEND_NAME_BYTES:
        raise BrainValidationError("checkpoint backend_name exceeds 256 UTF-8 bytes")
    if not isinstance(checkpoint.token, bytes):
        raise BrainValidationError("checkpoint token must be bytes")
    if len(checkpoint.token) > MAX_CHECKPOINT_BYTES:
        raise BrainValidationError("checkpoint token exceeds 64KB")
    if not isinstance(checkpoint.token_sha256, bytes) or len(checkpoint.token_sha256) != 32:
        raise BrainValidationError("checkpoint token_sha256 must be raw 32 bytes")
    if not hmac.compare_digest(hashlib.sha256(checkpoint.token).digest(), checkpoint.token_sha256):
        raise BrainValidationError("checkpoint token_sha256 mismatch")


def _read_pragmas(connection: sqlite3.Connection) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in _REQUIRED_PRAGMAS:
        row = connection.execute(f"PRAGMA {name}").fetchone()
        if row is None or len(row) != 1:
            raise BrainDurabilityError(f"SQLite PRAGMA {name} has no readable value")
        values[name] = row[0]
    return values


class BrainStateStore:
    """Synchronous facade over one bounded owner-thread command queue."""

    def __init__(
        self,
        data_dir: Path,
        *,
        dedup_horizon: int,
        feedback_horizon: int,
    ) -> None:
        self._data_dir = Path(data_dir).resolve(strict=False)
        self._brain_dir = (self._data_dir / "brain").resolve(strict=False)
        self.database_path = self._brain_dir / "brain_state.sqlite3"
        self._dedup_horizon = _positive_int("dedup_horizon", dedup_horizon)
        if (
            isinstance(feedback_horizon, bool)
            or not isinstance(feedback_horizon, int)
            or not 1 <= feedback_horizon <= 32
        ):
            raise BrainValidationError("feedback_horizon must be an integer in [1, 32]")
        self._feedback_horizon = feedback_horizon
        self._queue: queue.Queue[_Command] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._status = threading.Condition(threading.Lock())
        self._accepted: set[Future[Any]] = set()
        self._pending_admission = 0
        self._accepting = True
        self._failure: BrainDurabilityError | None = None
        self._ready = threading.Event()
        self._startup_cancelled = threading.Event()
        self._startup_error: BrainDurabilityError | None = None
        self._closed_event = threading.Event()
        self._close_condition = threading.Condition(threading.Lock())
        self._closing = False
        self._closed = False
        self._close_future: Future[Any] | None = None
        self._owner_connection: sqlite3.Connection | None = None
        self._lease_file: BinaryIO | None = None
        self._lease_stack: ExitStack | None = None
        self._lease_registered = False
        self._thread = threading.Thread(
            target=self._owner_main,
            name=f"SylanneBrainStore-{self._brain_dir.name}",
            daemon=False,
        )

    @classmethod
    def start(
        cls,
        data_dir: Path,
        *,
        dedup_horizon: int = 256,
        feedback_horizon: int = 8,
    ) -> BrainStateStore:
        store = cls(
            data_dir,
            dedup_horizon=dedup_horizon,
            feedback_horizon=feedback_horizon,
        )
        store._thread.start()
        if not store._ready.wait(10):
            with store._status:
                timed_out = not store._ready.is_set()
                if timed_out:
                    store._startup_cancelled.set()
            if timed_out:
                store._thread.join()
                raise BrainDurabilityError("brain store owner did not start")
        if store._startup_error is not None:
            store._thread.join(10)
            raise store._startup_error
        return store

    def _acquire_lease(self) -> None:
        self._brain_dir.mkdir(parents=True, exist_ok=True)
        canonical = self._brain_dir.resolve(strict=True)
        with _PROCESS_OWNERS_LOCK:
            if canonical in _PROCESS_OWNERS:
                raise BrainDurabilityError("brain writer process owner lease is already held")
            _PROCESS_OWNERS.add(canonical)
            self._lease_registered = True
        try:
            lease_path = canonical / ".writer.lock"
            stack = ExitStack()
            self._lease_stack = stack
            lease = stack.enter_context(lease_path.open("a+b"))
            if lease.seek(0, os.SEEK_END) == 0:
                lease.write(b"\x00")
                lease.flush()
            lease.seek(0)
            if os.name == "nt":
                msvcrt = _msvcrt_module()
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on Unix CI
                fcntl = _fcntl_module()
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lease_file = lease
        except BaseException:
            self._release_lease()
            raise

    def _release_lease(self) -> None:
        lease = self._lease_file
        self._lease_file = None
        stack = self._lease_stack
        self._lease_stack = None
        if lease is not None:
            try:
                lease.seek(0)
                if os.name == "nt":
                    msvcrt = _msvcrt_module()
                    msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised on Unix CI
                    fcntl = _fcntl_module()
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        if stack is not None:
            stack.close()
        if self._lease_registered:
            canonical = self._brain_dir.resolve(strict=False)
            with _PROCESS_OWNERS_LOCK:
                _PROCESS_OWNERS.discard(canonical)
            self._lease_registered = False

    def _open_database(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            check_same_thread=True,
        )
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            actual_pragmas = _read_pragmas(connection)
            if actual_pragmas != _REQUIRED_PRAGMAS:
                raise BrainDurabilityError(
                    f"required SQLite PRAGMA values were not applied: {actual_pragmas}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS brain_session(
                  session_digest BLOB PRIMARY KEY,
                  schema_version INTEGER NOT NULL,
                  generation INTEGER NOT NULL,
                  lineage_id TEXT NOT NULL,
                  history_epoch INTEGER NOT NULL,
                  mutation_seq INTEGER NOT NULL,
                  state_blob BLOB NOT NULL,
                  state_sha256 BLOB NOT NULL,
                  updated_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS brain_control(
                  session_digest BLOB PRIMARY KEY,
                  generation INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  updated_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS brain_dedup(
                  session_digest BLOB NOT NULL,
                  kind TEXT NOT NULL,
                  id_digest BLOB NOT NULL,
                  mutation_seq INTEGER NOT NULL,
                  receipt_blob BLOB NOT NULL,
                  receipt_sha256 BLOB NOT NULL,
                  created_ns INTEGER NOT NULL,
                  PRIMARY KEY(session_digest, kind, id_digest)
                );
                CREATE TABLE IF NOT EXISTS brain_backend_checkpoint(
                  session_digest BLOB PRIMARY KEY,
                  generation INTEGER NOT NULL,
                  backend_name TEXT NOT NULL,
                  backend_state_version INTEGER NOT NULL,
                  acknowledged_mutation_seq INTEGER NOT NULL,
                  token_blob BLOB NOT NULL,
                  token_sha256 BLOB NOT NULL
                );
                """
            )
            return connection
        except BaseException:
            try:
                connection.close()
            except sqlite3.Error:
                pass
            raise

    def _owner_main(self) -> None:
        connection: sqlite3.Connection | None = None
        normal_close = False
        try:
            self._acquire_lease()
            connection = self._open_database()
            with self._status:
                if self._startup_cancelled.is_set():
                    normal_close = True
                    return
                self._owner_connection = connection
                self._ready.set()
            while True:
                command = self._queue.get()
                try:
                    if command.operation == "__close__":
                        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        self._resolve(command.future, result=None)
                        normal_close = True
                        break
                    if not command.future.set_running_or_notify_cancel():
                        self._forget(command.future)
                        continue
                    result = self._execute_command(connection, command)
                    self._resolve(command.future, result=result)
                except sqlite3.Error as error:
                    if connection.in_transaction:
                        try:
                            connection.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                    raise BrainDurabilityError(
                        f"brain store SQLite command failed: {type(error).__name__}: {error}"
                    ) from error
                except Exception as error:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    self._resolve(command.future, error=error)
                finally:
                    self._queue.task_done()
        except BaseException as error:
            durability = (
                error
                if isinstance(error, BrainDurabilityError)
                else BrainDurabilityError(
                    f"brain store owner failed: {type(error).__name__}: {error}"
                )
            )
            if not self._ready.is_set():
                self._startup_error = durability
                self._ready.set()
            self._mark_failed(durability)
            if connection is not None and connection.in_transaction:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
        finally:
            self._owner_connection = None
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            self._release_lease()
            if not normal_close and self._failure is None:
                self._mark_failed(BrainDurabilityError("brain store owner stopped abnormally"))
            self._closed_event.set()

    def _resolve(
        self,
        future: Future[Any],
        *,
        result: object = None,
        error: BaseException | None = None,
    ) -> None:
        try:
            if not future.cancelled():
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)
        except InvalidStateError:
            pass
        finally:
            self._forget(future)

    def _forget(self, future: Future[Any]) -> None:
        with self._status:
            self._accepted.discard(future)
            self._status.notify_all()

    def _mark_failed(self, error: BrainDurabilityError) -> None:
        with self._status:
            if self._failure is None:
                self._failure = error
            self._accepting = False
            futures = tuple(self._accepted)
            self._status.notify_all()
        for future in futures:
            self._resolve(future, error=self._failure)

    def _create_command(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        closing: bool = False,
        admission: bool = False,
    ) -> _Command:
        future: Future[Any] = Future()
        with self._status:
            if self._failure is not None:
                raise self._failure
            if not self._accepting and not closing:
                raise BrainDurabilityError("brain store is closed")
            self._accepted.add(future)
            if admission:
                self._pending_admission += 1
        return _Command(operation, args, kwargs, future)

    def _submit(self, operation: str, *args: object, **kwargs: object) -> Any:
        command = self._create_command(operation, args, dict(kwargs), admission=True)
        try:
            while True:
                with self._status:
                    if self._failure is not None:
                        raise self._failure
                    if command.future.done():
                        return command.future.result()
                try:
                    self._queue.put(command, timeout=_QUEUE_POLL_SECONDS)
                    break
                except queue.Full:
                    continue
        finally:
            with self._status:
                self._pending_admission -= 1
                self._status.notify_all()
        return command.future.result()

    def close(self) -> None:
        with self._close_condition:
            while self._closing and not self._closed:
                self._close_condition.wait()
            if self._closed:
                return
            self._closing = True
        closed = False
        try:
            with self._status:
                self._accepting = False
                while self._pending_admission and self._failure is None:
                    self._status.wait(_QUEUE_POLL_SECONDS)
                failed = self._failure is not None
            if not failed and self._thread.is_alive():
                if self._close_future is None:
                    command = self._create_command("__close__", (), {}, closing=True)
                    self._close_future = command.future
                    while True:
                        try:
                            self._queue.put(command, timeout=_QUEUE_POLL_SECONDS)
                            break
                        except queue.Full:
                            with self._status:
                                if self._failure is not None:
                                    break
                if self._failure is None:
                    self._close_future.result()
            self._thread.join(10)
            if self._thread.is_alive():
                raise BrainDurabilityError("brain store owner did not terminate")
            closed = True
        finally:
            with self._close_condition:
                self._closed = closed
                self._closing = False
                self._close_condition.notify_all()

    def load(self, session_key: bytes) -> SessionLoaded | SessionMissing:
        key = _raw_digest("session_digest", session_key)
        return cast(SessionLoaded | SessionMissing, self._submit("load", key))

    def control(self, session_key: bytes) -> SessionControl | None:
        key = _raw_digest("session_digest", session_key)
        return cast(SessionControl | None, self._submit("control", key))

    def lookup_event_receipt(
        self,
        session_key: bytes,
        event_key: bytes,
    ) -> EventDuplicate | EventMiss:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("event_digest", event_key)
        return cast(
            EventDuplicate | EventMiss,
            self._submit("lookup_event", key, identifier),
        )

    def lookup_feedback_receipt(
        self,
        session_key: bytes,
        feedback_key: bytes,
    ) -> FeedbackDuplicate | FeedbackMiss:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("feedback_digest", feedback_key)
        return cast(
            FeedbackDuplicate | FeedbackMiss,
            self._submit("lookup_feedback", key, identifier),
        )

    def preflight_allocate(
        self,
        session_key: bytes,
        event_key: bytes,
    ) -> EventAllocated | EventDuplicate:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("event_digest", event_key)
        return cast(
            EventAllocated | EventDuplicate,
            self._submit("preflight_event", key, identifier),
        )

    def preflight_feedback(
        self,
        session_key: bytes,
        feedback_key: bytes,
        *,
        target_tick: int,
    ) -> FeedbackAllocated | FeedbackDuplicate:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("feedback_digest", feedback_key)
        target = _counter("target_tick", target_tick)
        return cast(
            FeedbackAllocated | FeedbackDuplicate,
            self._submit("preflight_feedback", key, identifier, target),
        )

    def commit_event(
        self,
        session_key: bytes,
        event_key: bytes,
        commit: EventCommit,
    ) -> EventCommitted | EventDuplicate:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("event_digest", event_key)
        if not isinstance(commit, EventCommit):
            raise BrainValidationError("event commit must be an EventCommit")
        state_blob, state_digest = encode_brain_bundle(commit.bundle)
        receipt_blob, receipt_digest = _receipt_bytes(commit.receipt)
        checkpoint = _validate_checkpoint(commit.checkpoint, commit.bundle)
        prepared = _PreparedEvent(
            commit.allocated,
            commit.bundle,
            state_blob,
            state_digest,
            commit.receipt,
            receipt_blob,
            receipt_digest,
            checkpoint,
        )
        return cast(
            EventCommitted | EventDuplicate,
            self._submit("commit_event", key, identifier, prepared),
        )

    def commit_feedback(
        self,
        session_key: bytes,
        feedback_key: bytes,
        commit: AppliedFeedbackCommit | ReceiptOnlyFeedbackCommit,
    ) -> FeedbackCommitted | FeedbackDuplicate:
        key = _raw_digest("session_digest", session_key)
        identifier = _raw_digest("feedback_digest", feedback_key)
        if isinstance(commit, AppliedFeedbackCommit):
            state_blob, state_digest = encode_brain_bundle(commit.bundle)
            receipt_blob, receipt_digest = _receipt_bytes(commit.receipt)
            checkpoint = _validate_checkpoint(commit.checkpoint, commit.bundle)
            prepared: _PreparedAppliedFeedback | _PreparedReceiptOnly = _PreparedAppliedFeedback(
                commit.allocated,
                commit.bundle,
                state_blob,
                state_digest,
                commit.receipt,
                receipt_blob,
                receipt_digest,
                checkpoint,
            )
        elif isinstance(commit, ReceiptOnlyFeedbackCommit):
            receipt_blob, receipt_digest = _receipt_bytes(commit.receipt)
            prepared = _PreparedReceiptOnly(commit.receipt, receipt_blob, receipt_digest)
        else:
            raise BrainValidationError("feedback commit type is invalid")
        return cast(
            FeedbackCommitted | FeedbackDuplicate,
            self._submit("commit_feedback", key, identifier, prepared),
        )

    def reset(self, session_key: bytes, *, expected_generation: int) -> BrainBundle:
        key = _raw_digest("session_digest", session_key)
        expected = _counter("expected_generation", expected_generation)
        return cast(BrainBundle, self._submit("reset", key, expected))

    def destroy(self, session_key: bytes, *, expected_generation: int) -> int:
        key = _raw_digest("session_digest", session_key)
        expected = _counter("expected_generation", expected_generation)
        return cast(int, self._submit("destroy", key, expected))

    def runtime_reference_matches(
        self,
        session_key: bytes,
        *,
        generation: int,
        lineage_id: str,
        mutation_seq: int,
    ) -> bool:
        key = _raw_digest("session_digest", session_key)
        return cast(
            bool,
            self._submit(
                "runtime_reference",
                key,
                _counter("generation", generation),
                lineage_id,
                _counter("mutation_seq", mutation_seq),
            ),
        )

    def _debug_pragmas(self) -> dict[str, object]:
        return cast(dict[str, object], self._submit("debug_pragmas"))

    def _debug_owner_in_transaction(self) -> bool:
        return cast(bool, self._submit("debug_in_transaction"))

    def _execute_command(
        self,
        connection: sqlite3.Connection,
        command: _Command,
    ) -> object:
        operation = command.operation
        if operation == "load":
            return self._load(connection, cast(bytes, command.args[0]))
        if operation == "control":
            control = self._control(connection, cast(bytes, command.args[0]))
            return None if control is None else SessionControl(control[0], cast(Any, control[1]))
        if operation == "lookup_event":
            return self._lookup(
                connection,
                cast(bytes, command.args[0]),
                "event",
                cast(bytes, command.args[1]),
            )
        if operation == "lookup_feedback":
            return self._lookup(
                connection,
                cast(bytes, command.args[0]),
                "feedback",
                cast(bytes, command.args[1]),
            )
        if operation == "preflight_event":
            return self._preflight_event(
                connection, cast(bytes, command.args[0]), cast(bytes, command.args[1])
            )
        if operation == "preflight_feedback":
            return self._preflight_feedback(
                connection,
                cast(bytes, command.args[0]),
                cast(bytes, command.args[1]),
                cast(int, command.args[2]),
            )
        if operation == "commit_event":
            return self._commit_event(
                connection,
                cast(bytes, command.args[0]),
                cast(bytes, command.args[1]),
                cast(_PreparedEvent, command.args[2]),
            )
        if operation == "commit_feedback":
            return self._commit_feedback(
                connection,
                cast(bytes, command.args[0]),
                cast(bytes, command.args[1]),
                cast(_PreparedAppliedFeedback | _PreparedReceiptOnly, command.args[2]),
            )
        if operation == "reset":
            return self._reset(connection, cast(bytes, command.args[0]), cast(int, command.args[1]))
        if operation == "destroy":
            return self._destroy(
                connection, cast(bytes, command.args[0]), cast(int, command.args[1])
            )
        if operation == "runtime_reference":
            return self._runtime_reference(
                connection,
                cast(bytes, command.args[0]),
                cast(int, command.args[1]),
                cast(str, command.args[2]),
                cast(int, command.args[3]),
            )
        if operation == "debug_pragmas":
            return _read_pragmas(connection)
        if operation == "debug_in_transaction":
            return connection.in_transaction
        raise BrainDurabilityError(f"unknown brain store operation: {operation}")

    def _control(self, connection: sqlite3.Connection, key: bytes) -> tuple[int, str] | None:
        row = connection.execute(
            "SELECT generation,status FROM brain_control WHERE session_digest=?", (key,)
        ).fetchone()
        if row is None:
            return None
        generation, status = row
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or not 0 <= generation <= MAX_COUNTER
            or status not in ("active", "destroyed")
        ):
            raise BrainDurabilityError("brain control invariant failed")
        return generation, cast(str, status)

    def _load(self, connection: sqlite3.Connection, key: bytes) -> SessionLoaded | SessionMissing:
        control = self._control(connection, key)
        row = connection.execute(
            "SELECT schema_version,generation,lineage_id,history_epoch,mutation_seq,"
            "state_blob,state_sha256 FROM brain_session WHERE session_digest=?",
            (key,),
        ).fetchone()
        if control is None:
            if row is not None:
                raise BrainDurabilityError("session exists without brain control")
            return SessionMissing()
        generation, status = control
        if status == "destroyed":
            if row is not None:
                raise BrainDurabilityError("destroyed control retains session state")
            residual = connection.execute(
                "SELECT (SELECT COUNT(*) FROM brain_dedup WHERE session_digest=?),"
                "(SELECT COUNT(*) FROM brain_backend_checkpoint WHERE session_digest=?)",
                (key, key),
            ).fetchone()
            if residual != (0, 0):
                raise BrainDurabilityError("destroyed control retains auxiliary state")
            return SessionMissing()
        if row is None:
            raise BrainDurabilityError("active control is missing session state")
        schema, row_generation, lineage, history, mutation, blob, digest = row
        if schema not in SUPPORTED_BRAIN_STATE_SCHEMA_VERSIONS:
            raise BrainDurabilityError("session schema invariant failed")
        try:
            bundle = decode_brain_bundle(
                cast(bytes, blob),
                cast(bytes, digest),
                expected_schema=cast(int, schema),
            )
        except BrainValidationError as error:
            raise BrainDurabilityError("stored state blob validation failed") from error
        if (
            row_generation != bundle.b.generation
            or generation != bundle.b.generation
            or lineage != bundle.b.lineage_id
            or history != bundle.b.history_epoch
            or mutation != bundle.b.mutation_seq
        ):
            raise BrainDurabilityError("session column/blob/control mutation invariant failed")
        checkpoint = self._load_checkpoint(connection, key, bundle)
        return SessionLoaded(bundle, checkpoint)

    def _load_checkpoint(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        bundle: BrainBundle,
    ) -> BackendCheckpoint | None:
        row = connection.execute(
            "SELECT generation,backend_name,backend_state_version,acknowledged_mutation_seq,"
            "token_blob,token_sha256 FROM brain_backend_checkpoint WHERE session_digest=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        checkpoint = BackendCheckpoint(
            generation=cast(int, row[0]),
            backend_name=cast(str, row[1]),
            backend_state_version=cast(int, row[2]),
            acknowledged_mutation_seq=cast(int, row[3]),
            token=cast(bytes, row[4]),
            token_sha256=cast(bytes, row[5]),
        )
        try:
            _validate_checkpoint_shape(checkpoint)
            if (
                checkpoint.generation != bundle.b.generation
                or checkpoint.acknowledged_mutation_seq > bundle.b.mutation_seq
            ):
                raise BrainValidationError("checkpoint version binding is invalid")
        except BrainValidationError as error:
            raise BrainDurabilityError("backend checkpoint invariant failed") from error
        return checkpoint

    def _ensure_session(self, connection: sqlite3.Connection, key: bytes) -> BrainBundle:
        loaded = self._load(connection, key)
        if isinstance(loaded, SessionLoaded):
            return loaded.bundle
        control = self._control(connection, key)
        if control is not None and control[1] == "destroyed":
            raise BrainDurabilityError("destroyed session cannot be lazily recreated")
        lineage = str(uuid4())
        bundle = BrainBundle(
            BrainState.fresh(
                generation=0,
                lineage_id=lineage,
                feedback_horizon=self._feedback_horizon,
            ),
            BrainCState.fresh(feedback_horizon=self._feedback_horizon),
        )
        blob, digest = encode_brain_bundle(bundle)
        now = time.time_ns()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO brain_control(session_digest,generation,status,updated_ns) "
                "VALUES(?,?,?,?)",
                (key, 0, "active", now),
            )
            connection.execute(
                "INSERT INTO brain_session(session_digest,schema_version,generation,lineage_id,"
                "history_epoch,mutation_seq,state_blob,state_sha256,updated_ns) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (key, BRAIN_STATE_SCHEMA_VERSION, 0, lineage, 0, 0, blob, digest, now),
            )
            self._commit_transaction(connection)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return bundle

    def _dedup_receipt(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: Literal["event", "feedback"],
        identifier: bytes,
        *,
        now: int | None = None,
    ) -> StoredReceipt | None:
        cutoff = (time.time_ns() if now is None else now) - DEDUP_TTL_NS
        row = connection.execute(
            "SELECT mutation_seq,receipt_blob,receipt_sha256,created_ns FROM ("
            "SELECT id_digest,mutation_seq,receipt_blob,receipt_sha256,created_ns "
            "FROM brain_dedup WHERE session_digest=? AND kind=? AND created_ns>=? "
            "ORDER BY created_ns DESC,id_digest DESC LIMIT ?"
            ") WHERE id_digest=?",
            (key, kind, cutoff, self._dedup_horizon, identifier),
        ).fetchone()
        if row is None:
            return None
        if isinstance(row[3], bool) or not isinstance(row[3], int):
            raise BrainDurabilityError("receipt timestamp invariant failed")
        receipt = _decode_receipt(row[1], row[2], kind)
        if row[0] != receipt.mutation_seq:
            raise BrainDurabilityError("receipt mutation column invariant failed")
        return receipt

    def _maintain_dedup_window(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: Literal["event", "feedback"],
        now: int,
    ) -> None:
        cutoff = now - DEDUP_TTL_NS
        needs_cleanup = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM brain_dedup WHERE session_digest=? AND kind=? "
            "AND created_ns<?),EXISTS(SELECT 1 FROM brain_dedup WHERE session_digest=? "
            "AND kind=? ORDER BY created_ns DESC,id_digest DESC LIMIT 1 OFFSET ?)",
            (key, kind, cutoff, key, kind, self._dedup_horizon),
        ).fetchone()
        if needs_cleanup == (0, 0):
            return
        if needs_cleanup is None or len(needs_cleanup) != 2:
            raise BrainDurabilityError("dedup maintenance query invariant failed")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "DELETE FROM brain_dedup WHERE session_digest=? AND kind=? AND created_ns<?",
                (key, kind, cutoff),
            )
            self._prune_dedup(connection, key, kind)
            self._commit_transaction(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _lookup(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: Literal["event", "feedback"],
        identifier: bytes,
    ) -> EventDuplicate | EventMiss | FeedbackDuplicate | FeedbackMiss:
        now = time.time_ns()
        self._maintain_dedup_window(connection, key, kind, now)
        receipt = self._dedup_receipt(connection, key, kind, identifier, now=now)
        if receipt is None:
            return EventMiss() if kind == "event" else FeedbackMiss()
        loaded = self._load(connection, key)
        if not isinstance(loaded, SessionLoaded):
            raise BrainDurabilityError("dedup receipt exists without active session")
        if kind == "event":
            return EventDuplicate(receipt, loaded.bundle)
        return FeedbackDuplicate(receipt, loaded.bundle)

    def _preflight_event(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        identifier: bytes,
    ) -> EventAllocated | EventDuplicate:
        duplicate = self._lookup(connection, key, "event", identifier)
        if isinstance(duplicate, EventDuplicate):
            return duplicate
        bundle = self._ensure_session(connection, key)
        if (
            bundle.b.tick_id == MAX_COUNTER
            or bundle.b.history_epoch == MAX_COUNTER
            or bundle.b.mutation_seq == MAX_COUNTER
        ):
            raise BrainCounterExhaustedError("event allocation counter is exhausted")
        return EventAllocated(
            bundle,
            EventAllocation(
                generation=bundle.b.generation,
                lineage_id=bundle.b.lineage_id,
                tick_id=bundle.b.tick_id + 1,
                history_epoch=bundle.b.history_epoch + 1,
                mutation_seq=bundle.b.mutation_seq + 1,
            ),
        )

    def _preflight_feedback(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        identifier: bytes,
        target_tick: int,
    ) -> FeedbackAllocated | FeedbackDuplicate:
        duplicate = self._lookup(connection, key, "feedback", identifier)
        if isinstance(duplicate, FeedbackDuplicate):
            return duplicate
        loaded = self._load(connection, key)
        if not isinstance(loaded, SessionLoaded):
            raise BrainAllocationError("feedback session is missing")
        bundle = loaded.bundle
        if target_tick > bundle.b.tick_id:
            raise BrainAllocationError("future feedback target is invalid")
        if bundle.b.mutation_seq == MAX_COUNTER:
            raise BrainCounterExhaustedError("feedback mutation counter is exhausted")
        return FeedbackAllocated(
            bundle,
            FeedbackAllocation(
                generation=bundle.b.generation,
                lineage_id=bundle.b.lineage_id,
                target_tick=target_tick,
                expected_mutation_seq=bundle.b.mutation_seq,
                next_mutation_seq=bundle.b.mutation_seq + 1,
            ),
        )

    def _validate_event_prepared(self, current: BrainBundle, prepared: _PreparedEvent) -> None:
        allocated = prepared.allocated
        if allocated.bundle != current:
            raise BrainAllocationError("event allocation is stale")
        expected = EventAllocation(
            generation=current.b.generation,
            lineage_id=current.b.lineage_id,
            tick_id=current.b.tick_id + 1,
            history_epoch=current.b.history_epoch + 1,
            mutation_seq=current.b.mutation_seq + 1,
        )
        if allocated.allocation != expected:
            raise BrainAllocationError("event allocation does not match latest state")
        candidate = prepared.bundle.b
        if (
            candidate.generation,
            candidate.lineage_id,
            candidate.tick_id,
            candidate.history_epoch,
            candidate.mutation_seq,
        ) != (
            expected.generation,
            expected.lineage_id,
            expected.tick_id,
            expected.history_epoch,
            expected.mutation_seq,
        ):
            raise BrainAllocationError("event candidate counters do not match allocation")
        receipt = prepared.receipt
        if receipt.status not in ("applied", "degraded"):
            raise BrainValidationError("event receipt status must be applied or degraded")
        if (
            receipt.kind != "event"
            or receipt.generation != candidate.generation
            or receipt.tick_id != candidate.tick_id
            or receipt.history_epoch != candidate.history_epoch
            or receipt.mutation_seq != candidate.mutation_seq
        ):
            raise BrainValidationError("event receipt does not match candidate")

    def _commit_event(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        identifier: bytes,
        prepared: _PreparedEvent,
    ) -> EventCommitted | EventDuplicate:
        duplicate = self._lookup(connection, key, "event", identifier)
        if isinstance(duplicate, EventDuplicate):
            return duplicate
        loaded = self._load(connection, key)
        if not isinstance(loaded, SessionLoaded):
            raise BrainAllocationError("event session is unavailable")
        current = loaded.bundle
        if current.b.mutation_seq == MAX_COUNTER:
            raise BrainCounterExhaustedError("event mutation counter is exhausted")
        self._validate_event_prepared(current, prepared)
        now = time.time_ns()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._delete_expired_identifier(connection, key, "event", identifier, now)
            duplicate_receipt = self._dedup_receipt(connection, key, "event", identifier)
            if duplicate_receipt is not None:
                connection.execute("ROLLBACK")
                reloaded = self._load(connection, key)
                if not isinstance(reloaded, SessionLoaded):
                    raise BrainDurabilityError("duplicate event lost active session")
                return EventDuplicate(duplicate_receipt, reloaded.bundle)
            self._recheck_current(connection, key, current)
            self._write_session(
                connection,
                key,
                current.b.mutation_seq,
                prepared.bundle,
                prepared.state_blob,
                prepared.state_digest,
                now,
            )
            self._write_dedup(
                connection,
                key,
                "event",
                identifier,
                prepared.receipt.mutation_seq,
                prepared.receipt_blob,
                prepared.receipt_digest,
                now,
            )
            if prepared.checkpoint is not None:
                self._write_checkpoint(connection, key, prepared.checkpoint)
            self._prune_dedup(connection, key, "event")
            self._commit_transaction(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return cast(
            EventCommitted,
            _ACKNOWLEDGEMENT_AUTHORITY.issue(
                self,
                EventCommitted,
                prepared.receipt,
                key,
                identifier,
                prepared.checkpoint,
            ),
        )

    def _validate_feedback_prepared(
        self,
        current: BrainBundle,
        prepared: _PreparedAppliedFeedback,
    ) -> None:
        allocated = prepared.allocated
        if allocated.bundle != current:
            raise BrainAllocationError("feedback allocation is stale")
        allocation = allocated.allocation
        expected = FeedbackAllocation(
            generation=current.b.generation,
            lineage_id=current.b.lineage_id,
            target_tick=allocation.target_tick,
            expected_mutation_seq=current.b.mutation_seq,
            next_mutation_seq=current.b.mutation_seq + 1,
        )
        if allocation != expected:
            raise BrainAllocationError("feedback allocation does not match latest mutation")
        candidate = prepared.bundle.b
        if candidate.mutation_seq != allocation.next_mutation_seq:
            raise BrainAllocationError("feedback candidate mutation does not match allocation")
        receipt = prepared.receipt
        if (
            receipt.kind != "feedback"
            or receipt.status not in ("applied", "degraded")
            or receipt.target_tick != allocation.target_tick
            or receipt.generation != candidate.generation
            or receipt.tick_id != candidate.tick_id
            or receipt.history_epoch != candidate.history_epoch
            or receipt.mutation_seq != candidate.mutation_seq
        ):
            raise BrainValidationError("feedback receipt does not match candidate")

    def _commit_feedback(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        identifier: bytes,
        prepared: _PreparedAppliedFeedback | _PreparedReceiptOnly,
    ) -> FeedbackCommitted | FeedbackDuplicate:
        duplicate = self._lookup(connection, key, "feedback", identifier)
        if isinstance(duplicate, FeedbackDuplicate):
            return duplicate
        loaded = self._load(connection, key)
        if not isinstance(loaded, SessionLoaded):
            raise BrainAllocationError("feedback session is unavailable")
        current = loaded.bundle
        now = time.time_ns()
        if isinstance(prepared, _PreparedReceiptOnly):
            receipt = prepared.receipt
            if receipt.status not in ("no_effect", "missed", "degraded"):
                raise BrainAllocationError("receipt-only feedback status is invalid")
            if receipt.applied_dimensions:
                raise BrainAllocationError("receipt-only feedback applied_dimensions must be empty")
            if receipt.applied_synapses:
                raise BrainAllocationError("receipt-only feedback applied_synapses must be zero")
            if (
                receipt.kind != "feedback"
                or receipt.generation != current.b.generation
                or receipt.tick_id != current.b.tick_id
                or receipt.history_epoch != current.b.history_epoch
                or receipt.mutation_seq != current.b.mutation_seq
            ):
                raise BrainAllocationError("receipt-only feedback must keep current mutation")
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._delete_expired_identifier(connection, key, "feedback", identifier, now)
                duplicate_receipt = self._dedup_receipt(connection, key, "feedback", identifier)
                if duplicate_receipt is not None:
                    connection.execute("ROLLBACK")
                    reloaded = self._load(connection, key)
                    if not isinstance(reloaded, SessionLoaded):
                        raise BrainDurabilityError("duplicate feedback lost active session")
                    return FeedbackDuplicate(duplicate_receipt, reloaded.bundle)
                self._recheck_current(connection, key, current)
                self._write_dedup(
                    connection,
                    key,
                    "feedback",
                    identifier,
                    receipt.mutation_seq,
                    prepared.receipt_blob,
                    prepared.receipt_digest,
                    now,
                )
                self._prune_dedup(connection, key, "feedback")
                self._commit_transaction(connection)
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            return cast(
                FeedbackCommitted,
                _ACKNOWLEDGEMENT_AUTHORITY.issue(
                    self,
                    FeedbackCommitted,
                    receipt,
                    key,
                    identifier,
                    None,
                ),
            )

        if current.b.mutation_seq == MAX_COUNTER:
            raise BrainCounterExhaustedError("feedback mutation counter is exhausted")
        self._validate_feedback_prepared(current, prepared)
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._delete_expired_identifier(connection, key, "feedback", identifier, now)
            duplicate_receipt = self._dedup_receipt(connection, key, "feedback", identifier)
            if duplicate_receipt is not None:
                connection.execute("ROLLBACK")
                reloaded = self._load(connection, key)
                if not isinstance(reloaded, SessionLoaded):
                    raise BrainDurabilityError("duplicate feedback lost active session")
                return FeedbackDuplicate(duplicate_receipt, reloaded.bundle)
            self._recheck_current(connection, key, current)
            self._write_session(
                connection,
                key,
                current.b.mutation_seq,
                prepared.bundle,
                prepared.state_blob,
                prepared.state_digest,
                now,
            )
            self._write_dedup(
                connection,
                key,
                "feedback",
                identifier,
                prepared.receipt.mutation_seq,
                prepared.receipt_blob,
                prepared.receipt_digest,
                now,
            )
            if prepared.checkpoint is not None:
                self._write_checkpoint(connection, key, prepared.checkpoint)
            self._prune_dedup(connection, key, "feedback")
            self._commit_transaction(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return cast(
            FeedbackCommitted,
            _ACKNOWLEDGEMENT_AUTHORITY.issue(
                self,
                FeedbackCommitted,
                prepared.receipt,
                key,
                identifier,
                prepared.checkpoint,
            ),
        )

    def _recheck_current(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        current: BrainBundle,
    ) -> None:
        row = connection.execute(
            "SELECT generation,lineage_id,mutation_seq FROM brain_session WHERE session_digest=?",
            (key,),
        ).fetchone()
        control = self._control(connection, key)
        if row != (current.b.generation, current.b.lineage_id, current.b.mutation_seq):
            raise BrainAllocationError("state CAS mutation is stale")
        if control != (current.b.generation, "active"):
            raise BrainAllocationError("control generation is stale")

    def _write_session(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        expected_mutation: int,
        bundle: BrainBundle,
        blob: bytes,
        digest: bytes,
        now: int,
    ) -> None:
        cursor = connection.execute(
            "UPDATE brain_session SET schema_version=?,generation=?,lineage_id=?,history_epoch=?,"
            "mutation_seq=?,state_blob=?,state_sha256=?,updated_ns=? "
            "WHERE session_digest=? AND mutation_seq=?",
            (
                BRAIN_STATE_SCHEMA_VERSION,
                bundle.b.generation,
                bundle.b.lineage_id,
                bundle.b.history_epoch,
                bundle.b.mutation_seq,
                blob,
                digest,
                now,
                key,
                expected_mutation,
            ),
        )
        if cursor.rowcount != 1:
            raise BrainAllocationError("state CAS update failed")
        connection.execute(
            "UPDATE brain_control SET generation=?,status='active',updated_ns=? "
            "WHERE session_digest=?",
            (bundle.b.generation, now, key),
        )

    def _write_dedup(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: str,
        identifier: bytes,
        mutation_seq: int,
        receipt_blob: bytes,
        receipt_digest: bytes,
        now: int,
    ) -> None:
        connection.execute(
            "INSERT INTO brain_dedup(session_digest,kind,id_digest,mutation_seq,receipt_blob,"
            "receipt_sha256,created_ns) VALUES(?,?,?,?,?,?,?)",
            (key, kind, identifier, mutation_seq, receipt_blob, receipt_digest, now),
        )

    def _delete_expired_identifier(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: str,
        identifier: bytes,
        now: int,
    ) -> None:
        connection.execute(
            "DELETE FROM brain_dedup WHERE session_digest=? AND kind=? AND id_digest=? "
            "AND created_ns<?",
            (key, kind, identifier, now - DEDUP_TTL_NS),
        )

    def _write_checkpoint(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        checkpoint: BackendCheckpoint,
    ) -> None:
        connection.execute(
            "INSERT INTO brain_backend_checkpoint(session_digest,generation,backend_name,"
            "backend_state_version,acknowledged_mutation_seq,token_blob,token_sha256) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_digest) DO UPDATE SET "
            "generation=excluded.generation,backend_name=excluded.backend_name,"
            "backend_state_version=excluded.backend_state_version,"
            "acknowledged_mutation_seq=excluded.acknowledged_mutation_seq,"
            "token_blob=excluded.token_blob,token_sha256=excluded.token_sha256",
            (
                key,
                checkpoint.generation,
                checkpoint.backend_name,
                checkpoint.backend_state_version,
                checkpoint.acknowledged_mutation_seq,
                checkpoint.token,
                checkpoint.token_sha256,
            ),
        )

    def _prune_dedup(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        kind: str,
    ) -> None:
        connection.execute(
            "DELETE FROM brain_dedup WHERE rowid IN ("
            "SELECT rowid FROM brain_dedup WHERE session_digest=? AND kind=? "
            "ORDER BY created_ns DESC,id_digest DESC LIMIT -1 OFFSET ?)",
            (key, kind, self._dedup_horizon),
        )

    def _commit_transaction(self, connection: sqlite3.Connection) -> None:
        connection.execute("COMMIT")

    def _reset(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        expected_generation: int,
    ) -> BrainBundle:
        control = self._control(connection, key)
        current_generation = 0 if control is None else control[0]
        if current_generation != expected_generation:
            raise BrainAllocationError("reset expected generation does not match")
        if current_generation == MAX_COUNTER:
            raise BrainCounterExhaustedError("generation is exhausted")
        generation = current_generation + 1
        lineage = str(uuid4())
        bundle = BrainBundle(
            BrainState.fresh(
                generation=generation,
                lineage_id=lineage,
                feedback_horizon=self._feedback_horizon,
            ),
            BrainCState.fresh(feedback_horizon=self._feedback_horizon),
        )
        blob, digest = encode_brain_bundle(bundle)
        now = time.time_ns()
        connection.execute("BEGIN IMMEDIATE")
        try:
            if self._control(connection, key) != control:
                raise BrainAllocationError("reset generation changed before commit")
            connection.execute(
                "INSERT INTO brain_control(session_digest,generation,status,updated_ns) "
                "VALUES(?,?,?,?) ON CONFLICT(session_digest) DO UPDATE SET "
                "generation=excluded.generation,status=excluded.status,updated_ns=excluded.updated_ns",
                (key, generation, "active", now),
            )
            connection.execute(
                "INSERT INTO brain_session(session_digest,schema_version,generation,lineage_id,"
                "history_epoch,mutation_seq,state_blob,state_sha256,updated_ns) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(session_digest) DO UPDATE SET schema_version=excluded.schema_version,"
                "generation=excluded.generation,lineage_id=excluded.lineage_id,history_epoch=0,"
                "mutation_seq=0,state_blob=excluded.state_blob,state_sha256=excluded.state_sha256,"
                "updated_ns=excluded.updated_ns",
                (
                    key,
                    BRAIN_STATE_SCHEMA_VERSION,
                    generation,
                    lineage,
                    0,
                    0,
                    blob,
                    digest,
                    now,
                ),
            )
            connection.execute("DELETE FROM brain_dedup WHERE session_digest=?", (key,))
            connection.execute(
                "DELETE FROM brain_backend_checkpoint WHERE session_digest=?", (key,)
            )
            self._commit_transaction(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return bundle

    def _destroy(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        expected_generation: int,
    ) -> int:
        control = self._control(connection, key)
        current_generation = 0 if control is None else control[0]
        if current_generation != expected_generation:
            raise BrainAllocationError("destroy expected generation does not match")
        if control is not None and control[1] == "destroyed":
            connection.execute("BEGIN IMMEDIATE")
            try:
                if self._control(connection, key) != control:
                    raise BrainAllocationError("destroy generation changed before cleanup")
                connection.execute("DELETE FROM brain_session WHERE session_digest=?", (key,))
                connection.execute("DELETE FROM brain_dedup WHERE session_digest=?", (key,))
                connection.execute(
                    "DELETE FROM brain_backend_checkpoint WHERE session_digest=?", (key,)
                )
                self._commit_transaction(connection)
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            return current_generation
        if current_generation == MAX_COUNTER:
            raise BrainCounterExhaustedError("generation is exhausted")
        generation = current_generation + 1
        now = time.time_ns()
        connection.execute("BEGIN IMMEDIATE")
        try:
            if self._control(connection, key) != control:
                raise BrainAllocationError("destroy generation changed before commit")
            connection.execute(
                "INSERT INTO brain_control(session_digest,generation,status,updated_ns) "
                "VALUES(?,?,?,?) ON CONFLICT(session_digest) DO UPDATE SET "
                "generation=excluded.generation,status=excluded.status,updated_ns=excluded.updated_ns",
                (key, generation, "destroyed", now),
            )
            connection.execute("DELETE FROM brain_session WHERE session_digest=?", (key,))
            connection.execute("DELETE FROM brain_dedup WHERE session_digest=?", (key,))
            connection.execute(
                "DELETE FROM brain_backend_checkpoint WHERE session_digest=?", (key,)
            )
            self._commit_transaction(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return generation

    def _runtime_reference(
        self,
        connection: sqlite3.Connection,
        key: bytes,
        generation: int,
        lineage_id: str,
        mutation_seq: int,
    ) -> bool:
        loaded = self._load(connection, key)
        return isinstance(loaded, SessionLoaded) and (
            loaded.bundle.b.generation,
            loaded.bundle.b.lineage_id,
            loaded.bundle.b.mutation_seq,
        ) == (generation, lineage_id, mutation_seq)


_ORIGINAL_EXECUTE = BrainStateStore._execute_command
