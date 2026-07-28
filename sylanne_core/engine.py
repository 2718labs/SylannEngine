"""SylanneEngine — the public entry point for Sylanne-Core SDK.

Provides the async API for integrating affective computation into chatbots.
Instantiate it directly with your own LLM callback.

Typical usage::

    from sylanne_core import SylanneEngine, SylanneConfig

    engine = SylanneEngine(data_dir="./data", llm=my_llm_fn)
    await engine.start()
    surface = await engine.process("session_1", "你好")
    print(surface["decision"]["action"])  # e.g. "express"
    await engine.shutdown()
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import logging
import math
import re
import struct
import time
import unicodedata
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .compute import SylanneHost
    from .compute.brain_store import BrainStateStore, StoredReceipt
    from .telemetry import DistillationSink

from .compute.utils import safe_filename
from .config import SylanneConfig
from .types import EngineStatus, FeedbackReceipt, HealthStatus, Surface

logger = logging.getLogger("sylanne_core")

_IDENTIFIER_MAX_BYTES = 256
_NOTIFICATION_CAPACITY = 64
_LISTENER_TIMEOUT_SECONDS = 30.0
_FEEDBACK_SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
_FEEDBACK_STATUSES = {
    "applied",
    "duplicate",
    "missed",
    "no_effect",
    "disabled",
    "degraded",
}


def _validated_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error
    if len(encoded) > _IDENTIFIER_MAX_BYTES:
        raise ValueError(f"{name} UTF-8 encoding must be at most 256 bytes")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value):
        raise ValueError(f"{name} must not contain Unicode control characters")
    return value


def _finite_feedback_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _generated_feedback_id(
    *, source: str, target_tick: int, value: float, confidence: float
) -> str:
    source_bytes = source.encode("utf-8")
    canonical_value = 0.0 if value == 0.0 else value
    canonical_confidence = 0.0 if confidence == 0.0 else confidence
    identity = (
        struct.pack(">I", len(source_bytes))
        + source_bytes
        + struct.pack(">Qdd", target_tick, canonical_value, canonical_confidence)
    )
    return hashlib.sha256(identity).hexdigest()


class _NotificationSlots:
    """Loop-affine reservation counter with a genuine nonblocking listener path."""

    __slots__ = ("_accepting", "_available", "_capacity", "_waiters")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        self._accepting = True
        self._waiters: deque[asyncio.Future[None]] = deque()

    async def acquire(self, *, nonblocking: bool) -> None:
        from .compute.brain_errors import BrainNotificationBackpressureError

        if not self._accepting:
            raise BrainNotificationBackpressureError("notification admission is closed")
        if self._available:
            self._available -= 1
            return
        if nonblocking:
            raise BrainNotificationBackpressureError(
                "listener-context notification capacity is full"
            )
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except BaseException:
            if not waiter.done():
                waiter.cancel()
            try:
                self._waiters.remove(waiter)
            except ValueError:
                pass
            raise

    def release(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return
        self._available = min(self._capacity, self._available + 1)

    def close(self, error: BaseException) -> None:
        self._accepting = False
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_exception(error)


@dataclass(slots=True)
class _NotificationRecord:
    session_id: str
    generation: int
    tick_id: int
    mutation_seq: int
    surface: Surface
    listeners: tuple[Callable[[str, Surface], Any], ...]
    completion: asyncio.Future[None]
    stop_after_current_callback: bool = False
    released: bool = False


@dataclass(slots=True)
class _NotificationState:
    slots: _NotificationSlots = field(
        default_factory=lambda: _NotificationSlots(_NOTIFICATION_CAPACITY)
    )
    queue: deque[_NotificationRecord] = field(default_factory=deque)
    worker: asyncio.Task[None] | None = None
    current: _NotificationRecord | None = None
    blocked_through_generation: int = -1


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    surface: Surface
    delivery: asyncio.Future[None] | None


_LISTENER_RECORD: contextvars.ContextVar[_NotificationRecord | None] = contextvars.ContextVar(
    "sylanne_listener_record",
    default=None,
)


@dataclass(slots=True)
class _SessionEntry:
    session_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    host: SylanneHost | None = None
    epoch: int = 0
    references: int = 0
    waiters: int = 0
    owners: int = 0
    eviction_reserved: bool = False


class _SessionLease:
    __slots__ = ("_engine", "_entry", "_session_id")

    def __init__(self, engine: SylanneEngine, session_id: str) -> None:
        self._engine = engine
        self._session_id = session_id
        self._entry: _SessionEntry | None = None

    async def __aenter__(self) -> _SessionEntry:
        entry = await self._engine._retain_session_entry(self._session_id)
        self._entry = entry
        try:
            await entry.lock.acquire()
        except BaseException:
            await self._engine._release_session_waiter(entry)
            self._entry = None
            raise
        await self._engine._promote_session_owner(entry)
        return entry

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> None:
        entry = self._entry
        if entry is None:
            return
        if exc_type is not None:
            from .compute.brain_errors import BrainDurabilityError, BrainOwnershipError

            if issubclass(exc_type, (BrainDurabilityError, BrainOwnershipError)):
                try:
                    await self._engine._detach_entry_host(entry, flush=False)
                except BaseException:
                    logger.warning(
                        "failed to close strict-failure Host for session %r",
                        entry.session_id,
                        exc_info=True,
                    )
        entry.lock.release()
        await self._engine._release_session_owner(entry)
        self._entry = None


@dataclass
class _Submission:
    """One in-flight-or-recently-completed ``submit()`` computation.

    Stored under one or two keys in ``engine._submissions`` — the text-hash
    key always, and the ``msg_id`` key too once a caller has supplied one —
    with BOTH keys pointing at the same instance, so a join through either
    index reaches the same task and eviction (via ``keys``) removes it once.
    """

    task: asyncio.Task[Surface]
    created: float
    text_hash: str
    msg_id: str | None
    ctx_fp: tuple[Any, ...]
    keys: tuple[Any, ...]
    done_at: float | None = None
    warned_text_divergence: bool = False
    warned_ctx_divergence: bool = False


class SylanneEngine:
    """Affective computation engine with session management and persistence.

    Each session maintains independent emotional state that evolves through
    a 7-layer computation pipeline on every process() call.

    Args:
        data_dir: Directory for session state persistence (created if missing).
        llm: Async function(system_prompt, user_prompt) -> str for semantic assessment.
        embedding: Optional async function(text) -> list[float] for memory retrieval.
        config: Engine configuration. Defaults to SylanneConfig().

    Example::

        async def my_llm(system: str, user: str) -> str:
            return await call_openai(system, user)

        engine = SylanneEngine(data_dir="./data", llm=my_llm)
        await engine.start()
        surface = await engine.process("user_123", "hello")
        # surface["decision"]["action"] in {"express", "listen", "hold", ...}
        await engine.shutdown()
    """

    def __init__(
        self,
        data_dir: str | Path,
        llm: Callable[[str, str], Awaitable[str]],
        embedding: Callable[[str], Awaitable[list[float]]] | None = None,
        config: SylanneConfig | None = None,
        *,
        assessor_llm: Callable[[str, str], Awaitable[str]] | None = None,
        _shared: bool = False,
    ) -> None:
        # _shared is set only by SylanneEngine.shared() when it builds the one
        # canonical instance for a data_dir. Direct construction (the default)
        # warns if that data_dir already has a live shared engine, to surface
        # the "many plugins, many redundant engines on one directory" waste.
        if not _shared:
            from ._sharing import warn_if_shared_exists

            warn_if_shared_exists(data_dir)
        self._data_dir = Path(data_dir)
        self._llm = llm
        self._embedding = embedding
        # Track whether config came from the file (vs passed in code) so start()
        # can drop a starter template only for the file-driven path.
        self._config_from_file = config is None
        # No config passed -> self-read it from the shared config file in data_dir
        # (one stable place users edit, independent of which copy owns the engine).
        # A missing/invalid file falls back to defaults. An ``assessor_model`` block
        # becomes a small dedicated assessor llm; without one, assessment falls back
        # to the main llm.
        if config is None:
            from ._config_store import load_config

            config, assessor_block = load_config(data_dir)
            if assessor_llm is None and assessor_block:
                from ._assessor_llm import build_from_config

                assessor_llm = build_from_config(assessor_block)
        from .compute.brain_backend import get_brain_backend_factory

        try:
            get_brain_backend_factory(config.brain_compute.c_backend)
        except KeyError:
            raise ValueError(f"unknown brain backend {config.brain_compute.c_backend!r}") from None
        self._config = config
        self._assessor_llm = assessor_llm
        self._shared = _shared
        self._status: EngineStatus = "init"
        self._hosts: OrderedDict[str, SylanneHost] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_entries: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._host_map_lock = asyncio.Lock()
        self._host_condition = asyncio.Condition(self._host_map_lock)
        self._host_build_reservations = 0
        self._host_trim_task: asyncio.Task[None] | None = None
        self._brain_store: BrainStateStore | None = None
        self._brain_inflight: dict[tuple[bytes, bytes], asyncio.Task[_ProcessResult]] = {}
        self._listeners: list[Callable[[str, Surface], Any]] = []
        self._listener_tasks: set[asyncio.Future[Any]] = set()
        self._notifications: dict[str, _NotificationState] = {}
        self._notification_pending: set[asyncio.Future[None]] = set()
        self._notifications_accepting = True
        self._telemetry_sink: DistillationSink | None = self._build_telemetry_sink()
        # --- submit() dedup table (engine-instance, engine-loop-affine: no locks) ---
        self._submissions: dict[Any, _Submission] = {}
        # Bounded FIFO of recently-evicted keys, used only to classify a later miss
        # on the same key as "recomputed_after_window" vs a genuinely new key for
        # submit_stats(). Capped like the dedup table itself so long-running
        # processes never grow this without bound.
        self._recent_evicted: OrderedDict[Any, None] = OrderedDict()
        self._stat_computed = 0
        self._stat_joined = 0
        self._stat_recomputed_after_window = 0
        # Diagnostics-only identity registry; see ``participants()``. NEVER read
        # by any branch that affects behavior — see module docs / submit().
        self._participants: dict[str, dict[str, Any]] = {}
        self._process_nudge_logged = False
        # --- tick() absolute-minimum-interval coalescer state ---
        self._last_tick: dict[str, tuple[float, Surface]] = {}

    @property
    def status(self) -> EngineStatus:
        return self._status

    async def start(self) -> None:
        """Initialize the engine. Must be called before process/tick."""
        if self._status == "running":
            return
        if self._status == "closed":
            self._host_map_lock = asyncio.Lock()
            self._host_condition = asyncio.Condition(self._host_map_lock)
            self._host_build_reservations = 0
            self._host_trim_task = None
        self._notifications_accepting = True
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if self._config_from_file:
            from ._config_store import write_default_config

            write_default_config(self._data_dir)
        if self._config.brain_compute.enabled:
            from .compute.brain_store import BrainStateStore

            self._brain_store = await asyncio.to_thread(
                BrainStateStore.start,
                self._data_dir,
                dedup_horizon=self._config.brain_compute.dedup_horizon,
                feedback_horizon=self._config.brain_compute.feedback_horizon,
            )
        self._status = "running"

    async def _ensure_started(self) -> None:
        """Start a not-yet-started engine; restart a closed DIRECT engine.

        A closed SHARED engine must NOT silently self-resurrect: it was released
        and removed from the registry, so reviving it here would let the next
        SylanneEngine.shared() build a SECOND engine on this data_dir and
        double-flush. Raise instead, telling the caller to re-acquire via shared().
        """
        if self._status == "closed":
            if self._shared:
                raise RuntimeError(
                    "This shared SylanneEngine was released (status='closed'). "
                    "Re-acquire it via SylanneEngine.shared(data_dir, ...); reusing a "
                    "released instance would duplicate the engine on this data_dir and "
                    "lose updates on flush."
                )
            await self.start()

    def on(self, listener: Callable[[str, Surface], Any]) -> None:
        """注册推送监听器。每次 process() 完成后，listener(session_id, surface) 会被调用。"""
        self._listeners.append(listener)

    def off(self, listener: Callable[[str, Surface], Any]) -> None:
        """移除推送监听器。"""
        self._listeners = [fn for fn in self._listeners if fn is not listener]

    def health(self) -> HealthStatus:
        """引擎健康检查，开发者用于判断计算模块是否正常。"""
        return {
            "status": self._status,
            "active_sessions": len(self._hosts),
            "data_dir_exists": self._data_dir.exists(),
            "llm_configured": self._llm is not None,
            "embedding_configured": self._embedding is not None,
        }

    async def shutdown(self) -> None:
        """Flush all sessions and release resources. Engine becomes 'closed'."""
        had_flush_error = False
        try:
            await self._shutdown_notifications()
        except Exception:
            had_flush_error = True
            logger.warning("notification shutdown failed", exc_info=True)
        trim_task = self._host_trim_task
        self._host_trim_task = None
        if trim_task is not None and not trim_task.done():
            trim_task.cancel()
            await asyncio.gather(trim_task, return_exceptions=True)
        for session_id, host in tuple(self._hosts.items()):
            try:
                await asyncio.to_thread(host.close, flush=True)
            except Exception:
                had_flush_error = True
                logger.warning(
                    "flush failed for session %r during shutdown; its latest state may be lost",
                    session_id,
                    exc_info=True,
                )
        self._hosts.clear()
        self._locks.clear()
        self._session_entries.clear()
        self._host_build_reservations = 0
        self._cancel_and_clear_submissions()
        self._brain_inflight.clear()
        self._last_tick.clear()
        store = self._brain_store
        self._brain_store = None
        if store is not None:
            try:
                await asyncio.to_thread(store.close)
            except Exception:
                had_flush_error = True
                logger.warning("brain store close failed during shutdown", exc_info=True)
        if self._telemetry_sink is not None:
            self._telemetry_sink.close()
        # Preserve a flush failure in the final status instead of masking it as a
        # clean 'closed'.
        self._status = "degraded" if had_flush_error else "closed"

    async def __aenter__(self) -> SylanneEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()

    async def process(
        self,
        session_id: str,
        text: str,
        *,
        confidence: float | None = None,
        flags: list[str] | None = None,
        now: float | None = None,
        values: dict[str, float] | None = None,
        event_id: str | None = None,
    ) -> Surface:
        """Process a user message and return the computed emotional surface.

        Args:
            session_id: Unique session identifier (e.g. user ID or chat ID).
            text: The user's message text.
            confidence: Pre-computed confidence score [0,1]. Overrides LLM assessment.
            flags: Semantic flags (e.g. ["safe"], ["hurt", "boundary"]).
            now: Unix timestamp. Defaults to current time.
            values: Additional numeric signals (e.g. {"group_heat": 0.7}).
                特殊键 ``"dialogue_quality"``（归一化 [0,1]）= 对上一轮回复的质量自评，
                驱动「越聊越校准」人格漂移：高分强化表达欲+拉近关系，低分收敛表达欲。
                滞后反馈——在评分对象的下一轮调用时传入。
            event_id: Stable platform event identity. Brain mode generates a UUID
                when omitted; generated IDs do not provide retry idempotency.

        Returns:
            Surface dict with keys: state, decision, guard, personality, memory, dynamics.

        Note:
            On a ``shared()`` engine, prefer ``submit()`` — it dedups identical
            messages across co-resident plugins; ``process()`` always recomputes,
            so N plugins calling it on the same shared engine still pay N LLM
            calls. This stays public (some callers genuinely want raw, always-fresh
            processing), but a shared engine logs a one-time nudge below.
        """
        if self._shared and not self._process_nudge_logged:
            self._process_nudge_logged = True
            logger.info(
                "process() called directly on the shared engine for %s; if you want "
                "cross-plugin duplicate-message dedup, call submit() instead — process() "
                "always recomputes and never joins",
                self._data_dir,
            )
        await self._ensure_started()
        if self._config.brain_compute.enabled:
            canonical_event_id = (
                str(uuid.uuid4())
                if event_id is None
                else _validated_identifier(event_id, name="event_id")
            )
            return await self._process_brain_event(
                session_id,
                text,
                event_id=canonical_event_id,
                confidence=confidence,
                flags=flags,
                now=now,
                values=values,
            )
        assessment = await self._assess(text) if self._config.assessor_enabled else None
        notification = await self._reserve_notification(session_id)
        enqueued = False
        try:
            async with self._session_lock(session_id):
                host = await self._get_or_create_host(session_id)
                event = {
                    "event_id": event_id,
                    "text": text,
                    "confidence": (
                        confidence
                        if confidence is not None
                        else (assessment or {}).get("confidence", 0.0)
                    ),
                    "flags": (flags if flags is not None else (assessment or {}).get("flags", [])),
                    "now": now if now is not None else time.time(),
                    "values": values or {},
                }
                result = host.on_request(event, assessment=assessment)
                surface = self._to_surface(session_id, host, result)
                delivery = self._enqueue_reserved_notification(
                    notification,
                    session_id=session_id,
                    generation=0,
                    tick_id=surface["turns"],
                    mutation_seq=surface["turns"],
                    surface=surface,
                )
                enqueued = True
        except BaseException:
            if not enqueued:
                notification.slots.release()
            raise
        if _LISTENER_RECORD.get() is None:
            await asyncio.shield(delivery)
        return surface

    async def _process_brain_event(
        self,
        session_id: str,
        text: str,
        *,
        event_id: str,
        confidence: float | None,
        flags: list[str] | None,
        now: float | None,
        values: dict[str, float] | None,
    ) -> Surface:
        from .compute.brain_errors import BrainDurabilityError, BrainOwnershipError
        from .compute.brain_store import event_id_digest, session_digest

        key = (session_digest(session_id), event_id_digest(event_id))
        existing = self._brain_inflight.get(key)
        if existing is not None:
            try:
                outcome = await asyncio.shield(existing)
            except (BrainDurabilityError, BrainOwnershipError):
                await self._discard_cached_host(session_id)
                raise
            if outcome.delivery is not None and _LISTENER_RECORD.get() is None:
                await asyncio.shield(outcome.delivery)
            return outcome.surface

        task = asyncio.create_task(
            self._process_brain_event_once(
                session_id,
                text,
                event_id=event_id,
                confidence=confidence,
                flags=flags,
                now=now,
                values=values,
                session_key=key[0],
                event_key=key[1],
            )
        )
        self._brain_inflight[key] = task

        def forget(done: asyncio.Task[_ProcessResult]) -> None:
            if self._brain_inflight.get(key) is done:
                self._brain_inflight.pop(key, None)

        task.add_done_callback(forget)
        try:
            outcome = await asyncio.shield(task)
        except (BrainDurabilityError, BrainOwnershipError):
            await self._discard_cached_host(session_id)
            raise
        if outcome.delivery is not None and _LISTENER_RECORD.get() is None:
            await asyncio.shield(outcome.delivery)
        return outcome.surface

    async def _process_brain_event_once(
        self,
        session_id: str,
        text: str,
        *,
        event_id: str,
        confidence: float | None,
        flags: list[str] | None,
        now: float | None,
        values: dict[str, float] | None,
        session_key: bytes,
        event_key: bytes,
    ) -> _ProcessResult:
        from .compute.brain_errors import BrainDurabilityError
        from .compute.brain_store import EventDuplicate

        store = self._brain_store
        if store is None:
            raise BrainDurabilityError("brain store is unavailable")

        early = await asyncio.to_thread(store.lookup_event_receipt, session_key, event_key)
        if isinstance(early, EventDuplicate):
            async with self._session_lock(session_id):
                host = await self._get_or_create_host(session_id)
                surface = self._to_surface(session_id, host, host.diagnostics())
            return _ProcessResult(
                self._with_brain_event(
                    surface,
                    self._public_event_receipt(
                        early.receipt,
                        event_id=event_id,
                        status="duplicate",
                    ),
                ),
                None,
            )

        assessment = await self._assess(text) if self._config.assessor_enabled else None
        notification = await self._reserve_notification(session_id)
        enqueued = False
        try:
            async with self._session_lock(session_id):
                host = await self._get_or_create_host(session_id)
                event = {
                    "event_id": event_id,
                    "text": text,
                    "confidence": (
                        confidence
                        if confidence is not None
                        else (assessment or {}).get("confidence", 0.0)
                    ),
                    "flags": (flags if flags is not None else (assessment or {}).get("flags", [])),
                    "now": now if now is not None else time.time(),
                    "values": values or {},
                }
                result = await asyncio.to_thread(host.on_request, event, assessment)
                committed = await asyncio.to_thread(
                    store.lookup_event_receipt,
                    session_key,
                    event_key,
                )
                if not isinstance(committed, EventDuplicate):
                    raise BrainDurabilityError("brain event returned without a durable receipt")
                surface = self._to_surface(session_id, host, result)
                surface = self._with_brain_event(
                    surface,
                    self._public_event_receipt(committed.receipt, event_id=event_id),
                )
                delivery = self._enqueue_reserved_notification(
                    notification,
                    session_id=session_id,
                    generation=committed.receipt.generation,
                    tick_id=committed.receipt.tick_id,
                    mutation_seq=committed.receipt.mutation_seq,
                    surface=surface,
                )
                enqueued = True
        except BaseException:
            if not enqueued:
                notification.slots.release()
            raise
        return _ProcessResult(surface, delivery)

    @staticmethod
    def _public_event_receipt(
        receipt: StoredReceipt,
        *,
        event_id: str,
        status: str | None = None,
    ) -> dict[str, object]:
        return {
            "status": receipt.status if status is None else status,
            "event_id": event_id,
            "generation": receipt.generation,
            "tick_id": receipt.tick_id,
            "history_epoch": receipt.history_epoch,
            "mutation_seq": receipt.mutation_seq,
        }

    @staticmethod
    def _with_brain_event(surface: Surface, receipt: dict[str, object]) -> Surface:
        surface["pipeline"] = {**surface["pipeline"], "brain_event": dict(receipt)}
        return surface

    async def feedback(
        self,
        session_id: str,
        *,
        target_tick: int,
        value: float,
        confidence: float,
        source: str,
        feedback_id: str | None = None,
    ) -> FeedbackReceipt:
        """Apply idempotent delayed feedback to one canonical brain event tick."""
        from .compute.brain_errors import BrainDurabilityError
        from .compute.brain_state import MAX_COUNTER

        if isinstance(target_tick, bool) or not isinstance(target_tick, int):
            raise ValueError("target_tick must be an integer")
        if not 0 <= target_tick <= MAX_COUNTER:
            raise ValueError("target_tick is outside the persisted counter domain")
        signal = max(-1.0, min(1.0, _finite_feedback_number(value, name="value")))
        confidence_value = _finite_feedback_number(confidence, name="confidence")
        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if not isinstance(source, str) or _FEEDBACK_SOURCE.fullmatch(source) is None:
            raise ValueError("source must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
        canonical_feedback_id = (
            _generated_feedback_id(
                source=source,
                target_tick=target_tick,
                value=signal,
                confidence=confidence_value,
            )
            if feedback_id is None
            else _validated_identifier(feedback_id, name="feedback_id")
        )

        await self._ensure_started()
        if not self._config.brain_compute.enabled:
            return FeedbackReceipt(
                status="disabled",
                session_id=session_id,
                target_tick=target_tick,
                feedback_id=canonical_feedback_id,
                applied_dimensions=(),
                applied_synapses=0,
                mutation_seq=0,
            )

        async with self._session_lock(session_id):
            host = await self._get_or_create_host(session_id)
            state = host.brain_state
            if state is None:
                raise RuntimeError("brain-enabled host has no authoritative brain state")
            if target_tick > state.tick_id:
                raise ValueError("target_tick must not be greater than the current tick")
            result = await asyncio.to_thread(
                host.apply_targeted_feedback,
                feedback_id=canonical_feedback_id,
                target_tick=target_tick,
                value=signal,
                confidence=confidence_value,
            )

        status = result.get("status")
        if not isinstance(status, str) or status not in _FEEDBACK_STATUSES:
            raise BrainDurabilityError(f"invalid targeted feedback status {status!r}")
        raw_dimensions = result.get("applied_dimensions", ())
        if not isinstance(raw_dimensions, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in raw_dimensions
        ):
            raise BrainDurabilityError("targeted feedback dimensions are invalid")
        raw_synapses = result.get("applied_synapses", 0)
        if isinstance(raw_synapses, bool) or not isinstance(raw_synapses, int):
            raise BrainDurabilityError("targeted feedback synapse count is invalid")
        raw_mutation_seq = result.get("mutation_seq", state.mutation_seq)
        if isinstance(raw_mutation_seq, bool) or not isinstance(raw_mutation_seq, int):
            raise BrainDurabilityError("targeted feedback mutation_seq is invalid")
        return FeedbackReceipt(
            status=cast(Any, status),
            session_id=session_id,
            target_tick=target_tick,
            feedback_id=canonical_feedback_id,
            applied_dimensions=raw_dimensions,
            applied_synapses=raw_synapses,
            mutation_seq=raw_mutation_seq,
        )

    async def submit(
        self,
        session_id: str,
        text: str,
        *,
        msg_id: str | None = None,
        confidence: float | None = None,
        flags: list[str] | None = None,
        now: float | None = None,
        values: dict[str, float] | None = None,
        dedup: bool = True,
        plugin: str | None = None,
    ) -> Surface:
        """Submit a message for idempotent, cross-plugin-dedup'd processing.

        The front door for a ``shared()`` engine. When several co-resident
        plugins each receive the SAME platform event and each call ``submit()``
        with the same ``session_id`` (and the same text, or better, the same
        ``msg_id``), only the FIRST call actually computes — the rest join that
        computation and all receive the identical :class:`Surface`. There is no
        election, no role, and no dependence on which plugin loaded first or
        how many plugins exist: correctness comes from "has this message been
        submitted before", not from anyone's identity.

        The guarantee is precise, not magical: it is 1x for submit() callers
        that use CONSISTENT keys — pass the platform's raw, unmodified message
        text, or (strongly recommended) a stable ``msg_id`` such as the
        platform's own message id. Mixing "sometimes msg_id, sometimes not" for
        the same logical message is absorbed by the dual index (see below) but
        is not what gives the tightest guarantee. Callers that bypass submit()
        (calling ``process()`` directly) are not covered — that is a deliberate
        escape hatch, not a hole in this contract.

        Args:
            session_id: Unique session identifier (e.g. user ID or chat ID).
            text: The user's message text. Pass the platform's raw text.
            msg_id: The platform's own message id, if available (e.g.
                ``event.message_obj.message_id``). Strongly recommended: it is
                the exact, un-guessable join key and immune to text-normalization
                differences between plugins.
            confidence: Pre-computed confidence score [0,1]. Overrides LLM assessment.
            flags: Semantic flags (e.g. ["safe"], ["hurt", "boundary"]).
            now: Unix timestamp. Defaults to current time.
            values: Additional numeric signals — see ``process()``.
            dedup: When False, this is exactly ``process()`` (no dedup table
                involvement at all) — the raw, always-recompute escape hatch.
            plugin: Optional caller identity string, purely for diagnostics (see
                ``participants()``/``submit_stats()``). Never affects dedup/join
                behavior — identity observes, idempotency guarantees.

        Returns:
            The Surface — either freshly computed (this call was first) or
            joined from another awaiter's in-flight/just-completed computation.

        Dedup mechanics (engine-instance table, engine-loop-affine — no locks):
            Each first-seen submission is indexed under its text-hash key
            always, and ALSO under its ``msg_id`` key when one was given. A
            later call with no ``msg_id`` may join on a hash hit. A later call
            WITH ``msg_id`` joins an existing hash-hit only if that entry
            recorded no ``msg_id`` of its own; if it recorded a DIFFERENT
            ``msg_id``, that is a genuine repeat (same text, distinct message)
            and triggers a fresh computation. A direct ``msg_id`` hit always
            joins — even if its recorded text differs (logged once as a
            warning; reusing a msg_id for different text is a dangerous
            pattern and the platform is trusted here, so the FIRST submission
            for that id wins). Divergent confidence/flags/values on a join is
            logged once at debug level; the first submitter's context wins.

            The real computation runs in a DETACHED task (created, not directly
            awaited) so that any one awaiter cancelling its own await can never
            cancel the shared computation for the others — every awaiter
            (including the first submitter) awaits ``asyncio.shield(task)``.

            Completed entries are pruned lazily on each ``submit()`` call: once
            older than ``config.submit_window_seconds``, or beyond
            ``config.submit_max_entries`` (oldest completed evicted first).
            IN-FLIGHT entries are never capped or evicted by either rule. A
            FAILED task evicts its own keys immediately — a poisoned entry
            does not get to "stick" for the rest of the window.
        """
        if self._config.brain_compute.enabled:
            canonical_event_id = (
                str(uuid.uuid4())
                if msg_id is None
                else _validated_identifier(msg_id, name="msg_id")
            )
            if dedup:
                from .compute.brain_store import event_id_digest, session_digest

                key = (session_digest(session_id), event_id_digest(canonical_event_id))
                joined = key in self._brain_inflight
                if joined:
                    self._stat_joined += 1
                else:
                    self._stat_computed += 1
                self._note_submit_outcome(plugin, joined=joined)
            return await self.process(
                session_id,
                text,
                confidence=confidence,
                flags=flags,
                now=now,
                values=values,
                event_id=canonical_event_id,
            )

        if not dedup:
            return await self.process(
                session_id,
                text,
                confidence=confidence,
                flags=flags,
                now=now,
                values=values,
                event_id=msg_id,
            )

        self._prune_submissions()

        text_hash = "h:" + hashlib.blake2b(text.encode()).hexdigest()[:32]
        key_hash = (session_id, text_hash)
        key_msg = (session_id, msg_id) if msg_id is not None else None
        ctx_fp = self._ctx_fingerprint(confidence, flags, values)

        entry: _Submission | None = None
        joined = False

        if key_msg is not None:
            entry = self._submissions.get(key_msg)
            if entry is not None:
                joined = True
                if entry.text_hash != text_hash and not entry.warned_text_divergence:
                    entry.warned_text_divergence = True
                    logger.warning(
                        "submit(): msg_id %r on session %r was already submitted with "
                        "DIFFERENT text — joining the first submission's result anyway "
                        "(msg_id is trusted as authoritative). Reusing a msg_id across "
                        "distinct message text is a dangerous pattern; make sure msg_id "
                        "is unique per logical message.",
                        msg_id,
                        session_id,
                    )
            else:
                hash_hit = self._submissions.get(key_hash)
                if hash_hit is not None and hash_hit.msg_id is None:
                    # Upgrade the hash-only entry so future msg_id lookups join directly.
                    hash_hit.msg_id = msg_id
                    hash_hit.keys = (*hash_hit.keys, key_msg)
                    self._submissions[key_msg] = hash_hit
                    entry = hash_hit
                    joined = True
                elif hash_hit is not None and hash_hit.msg_id != msg_id:
                    # A different msg_id already claims this text: a genuine repeat,
                    # not a duplicate delivery of the same message -> compute fresh.
                    entry = None
        else:
            hash_hit = self._submissions.get(key_hash)
            if hash_hit is not None:
                entry = hash_hit
                joined = True

        if joined and entry is not None:
            if entry.ctx_fp != ctx_fp and not entry.warned_ctx_divergence:
                entry.warned_ctx_divergence = True
                logger.debug(
                    "submit(): session %r joined a submission whose confidence/flags/"
                    "values differ from this call's; the first submitter's context wins.",
                    session_id,
                )
            self._stat_joined += 1
            self._note_submit_outcome(plugin, joined=True)
            return await asyncio.shield(entry.task)

        # Miss: this call is the one that computes.
        was_recently_evicted = key_hash in self._recent_evicted or (
            key_msg is not None and key_msg in self._recent_evicted
        )
        keys: tuple[Any, ...] = (key_hash,) if key_msg is None else (key_hash, key_msg)
        task: asyncio.Task[Surface] = asyncio.ensure_future(
            self.process(
                session_id,
                text,
                confidence=confidence,
                flags=flags,
                now=now,
                values=values,
                event_id=msg_id,
            )
        )
        new_entry = _Submission(
            task=task,
            created=time.time(),
            text_hash=text_hash,
            msg_id=msg_id,
            ctx_fp=ctx_fp,
            keys=keys,
        )
        for k in keys:
            self._submissions[k] = new_entry
            self._recent_evicted.pop(k, None)
        task.add_done_callback(self._on_submission_done)
        if was_recently_evicted:
            self._stat_recomputed_after_window += 1
        else:
            self._stat_computed += 1
        self._note_submit_outcome(plugin, joined=False)
        return await asyncio.shield(task)

    def submit_stats(self) -> dict[str, Any]:
        """Snapshot of submit() dedup counters since engine construction.

        Returns ``{"computed": int, "joined": int, "recomputed_after_window": int}``.
        ``recomputed_after_window`` is a best-effort heuristic bounded the same
        way the dedup table itself is (an LRU of recently-evicted keys) — a miss
        for a key evicted long enough ago to have aged out of that LRU is
        counted as ``computed`` instead. When any ``submit(..., plugin=...)``
        calls have been made, also includes ``"by_plugin"``: a
        ``{plugin: {"submits": int, "joins": int}}`` breakdown — diagnostics
        only, sourced from the same registry as ``participants()``.
        """
        stats: dict[str, Any] = {
            "computed": self._stat_computed,
            "joined": self._stat_joined,
            "recomputed_after_window": self._stat_recomputed_after_window,
        }
        if self._participants:
            stats["by_plugin"] = {
                name: {"submits": info["submits"], "joins": info["joins"]}
                for name, info in self._participants.items()
            }
        return stats

    def participants(self) -> list[dict[str, Any]]:
        """Snapshot of the participants registry, sorted by ``first_seen``.

        Populated only when callers pass ``plugin=`` to ``shared()``/``submit()``.
        This is diagnostics, never behavior: nothing in submit()'s dedup/join
        logic reads it, and no plugin is treated specially for having (or
        lacking) an entry here. Each item is
        ``{"plugin", "copy_id", "sdk_version", "first_seen", "submits", "joins"}``.
        """
        return [
            {"plugin": name, **info}
            for name, info in sorted(
                self._participants.items(), key=lambda item: item[1]["first_seen"]
            )
        ]

    def set_llm(
        self,
        llm: Callable[[str, str], Awaitable[str]],
        *,
        assessor_llm: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        """Ops escape hatch: swap this engine's llm callback(s) at runtime.

        For replacing a dead builder's closure on a shared engine — e.g. the
        plugin that built the engine was hot-disabled/reloaded and its ``llm``
        callable now points at torn-down state (see the "driver death" note in
        SPEC) — without restarting the process. No auto-magic: nothing calls
        this for you; an operator or a health-check script decides when and
        whether to swap.

        Args:
            llm: Replaces the main llm callback used for ``process()``/``submit()``.
            assessor_llm: If given, also replaces the assessor llm. Omit to
                leave the current assessor (or main-llm fallback) untouched.
        """
        self._llm = llm
        if assessor_llm is not None:
            self._assessor_llm = assessor_llm
        logger.info("SylanneEngine llm swapped via set_llm() for %s", self._data_dir)

    async def tick(
        self,
        session_id: str,
        flags: list[str] | None = None,
        *,
        force: bool = False,
    ) -> Surface:
        """Advance session state without user input (background heartbeat).

        Enforces a per-session ABSOLUTE MINIMUM interval:
        ``config.tick_min_interval_seconds`` (default 45s). A call within that
        interval of the session's last real tick returns the cached Surface
        WITHOUT advancing state at all — no host mutation, no new snapshot.
        This is not a heartbeat scheduler and it does not own any timer of its
        own; it exists so that several co-resident plugins each running their
        own independent ~60s heartbeat loop against the same shared engine
        collapse to roughly one real tick per interval instead of N.

        ``force=True`` bypasses the coalescer and always advances state. This
        is a test/ops escape hatch, not a "heartbeat owner" role — nothing in
        this engine grants special status to a forcing caller, and racing the
        interval on purpose defeats the point of calling tick() at all.
        """
        await self._ensure_started()
        now_ts = time.time()
        if not force:
            cached = self._last_tick.get(session_id)
            if cached is not None and now_ts - cached[0] < self._config.tick_min_interval_seconds:
                return cached[1]
        async with self._session_lock(session_id):
            host = await self._get_or_create_host(session_id)
            event = {
                "text": "",
                "confidence": 0.0,
                "flags": flags or ["idle"],
                "now": now_ts,
                "values": {},
            }
            result = host.on_request(event)
            surface = self._to_surface(session_id, host, result)
        self._last_tick[session_id] = (now_ts, surface)
        return surface

    async def state(self, session_id: str) -> Surface:
        """Get current session state without advancing the pipeline."""
        # Mirror process/tick/inject: a released SHARED engine must refuse to
        # rehydrate a host here too, else an observer reading state() on a
        # released engine would silently rebuild hosts from disk and bypass the
        # resurrection guard. _ensure_started is a no-op on a running engine.
        await self._ensure_started()
        async with self._session_lock(session_id):
            host = await self._get_or_create_host(session_id)
            surface = host.diagnostics()
            return self._to_surface(session_id, host, surface)

    async def reset(self, session_id: str) -> None:
        """Reset session to fresh state. Deletes persisted data."""
        await self._ensure_started()
        async with self._session_lock(session_id) as entry:
            if self._config.brain_compute.enabled:
                from .compute.brain_errors import BrainDurabilityError
                from .compute.brain_store import session_digest

                store = self._brain_store
                if store is None:
                    raise BrainDurabilityError("brain store is unavailable during reset")
                session_key = session_digest(session_id)
                control = await asyncio.to_thread(store.control, session_key)
                if control is not None and control.status == "destroyed":
                    raise BrainDurabilityError("destroyed session cannot be reset")
                expected_generation = 0 if control is None else control.generation
                await self._barrier_notifications(
                    session_id,
                    through_generation=expected_generation,
                )
                await asyncio.to_thread(
                    store.reset,
                    session_key,
                    expected_generation=expected_generation,
                )
            else:
                await self._barrier_notifications(session_id, through_generation=0)
                # Non-brain sessions have no durable generation counter: every
                # process() notification is enqueued with generation 0, so a
                # persistent generation-0 barrier would permanently silence ALL
                # future notifications for this session. The barrier above has
                # already cancelled any in-flight/queued notification, so drop the
                # drained state; the next process() rebuilds a fresh, unblocked
                # notification channel.
                self._notifications.pop(session_id, None)
            await self._detach_entry_host(entry, flush=False)
            safe_name = safe_filename(session_id)
            for suffix in (".alpha.json", ".json"):
                state_file = self._data_dir / f"{safe_name}{suffix}"
                if state_file.exists():
                    state_file.unlink()

    async def destroy(self, session_id: str) -> None:
        """Permanently remove session state and release its lock."""
        if not self._config.brain_compute.enabled:
            await self.reset(session_id)
            return

        from .compute.brain_errors import BrainDurabilityError
        from .compute.brain_store import session_digest

        await self._ensure_started()
        async with self._session_lock(session_id) as entry:
            store = self._brain_store
            if store is None:
                raise BrainDurabilityError("brain store is unavailable during destroy")
            session_key = session_digest(session_id)
            control = await asyncio.to_thread(store.control, session_key)
            expected_generation = 0 if control is None else control.generation
            await self._barrier_notifications(
                session_id,
                through_generation=expected_generation,
            )
            await asyncio.to_thread(
                store.destroy,
                session_key,
                expected_generation=expected_generation,
            )
            await self._detach_entry_host(entry, flush=False)
            safe_name = safe_filename(session_id)
            for suffix in (".alpha.json", ".json"):
                state_file = self._data_dir / f"{safe_name}{suffix}"
                if state_file.exists():
                    state_file.unlink()

    async def inject(
        self,
        session_id: str,
        source: str,
        influence_type: str,
        intensity: float,
        target_dimension: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Inject external influence into a session's hot pool.

        Other plugins call this to affect the emotional state of a session.
        For example, a memory plugin detecting contradiction with a previously
        reflected topic can re-ignite that material in the hot pool.

        Args:
            session_id: Target session identifier.
            source: Plugin identifier (e.g. "memory_plugin", "dialogue_agent").
            influence_type: One of "contradiction", "reinforcement", "revelation",
                           "betrayal", "validation".
            intensity: Influence strength [0, 1].
            target_dimension: Target dimension or material type in the hot pool.
            payload: Optional metadata dict passed through to the influence.

        Raises:
            ValueError: If influence_type is not a recognized type.
        """
        from .compute.hot_pool import _VALID_INFLUENCE_TYPES

        if influence_type not in _VALID_INFLUENCE_TYPES:
            raise ValueError(
                f"Invalid influence_type {influence_type!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_INFLUENCE_TYPES))}"
            )
        await self._ensure_started()
        async with self._session_lock(session_id):
            host = await self._get_or_create_host(session_id)
            from .compute.hot_pool import Influence

            influence = Influence(
                source=source,
                type=influence_type,  # type: ignore[arg-type]
                intensity=max(0.0, min(1.0, intensity)),
                target_dimension=target_dimension,
                payload=payload or {},
            )
            host.kernel.hot_pool.receive_influence(influence)

    def exists(self, session_id: str) -> bool:
        """Check if a session exists without creating it."""
        return session_id in self._hosts

    # --- shared instance registry ---

    @classmethod
    async def shared(
        cls,
        data_dir: str | Path,
        llm: Callable[[str, str], Awaitable[str]],
        embedding: Callable[[str], Awaitable[list[float]]] | None = None,
        config: SylanneConfig | None = None,
        *,
        assessor_llm: Callable[[str, str], Awaitable[str]] | None = None,
        plugin: str | None = None,
    ) -> SylanneEngine:
        """Return (and start) the process-shared engine for ``data_dir``.

        One engine per resolved data_dir is maintained for the process lifetime,
        so independent call sites can share a single instance without passing
        the object around — and one persistence directory is never owned by two
        engines at once WITHIN THIS PROCESS (which would cause lost updates on
        flush). This guarantee is per-process only: there is no cross-process
        lock, so running two OS processes on one data_dir will double-flush and
        lose updates — use one process per data_dir.

        When ``config`` is omitted, the engine self-reads it from
        ``<data_dir>/sylanne.config.json`` (one stable place users edit), so every
        plugin can just call ``shared(data_dir)`` and share the same settings. An
        ``assessor_model`` block in that file routes assessment to a small
        dedicated model; otherwise assessment uses the main ``llm``.

        Later calls with the same data_dir return the existing instance. A
        conflicting ``config`` raises SharedEngineConflictError; a different
        ``llm``/``embedding`` object logs a warning but reuses the original.
        The returned engine is already started (status == "running").

        Every co-resident plugin gets the SAME full engine — there is no
        driver/observer split. Call ``submit()`` (not ``process()``) so
        duplicate deliveries of the same message across plugins dedup instead
        of each paying for their own LLM call.

        Args:
            plugin: Optional caller identity string, purely for diagnostics —
                recorded in ``engine.participants()`` (first attach logged at
                INFO). Never gates behavior: identity observes, submit()'s
                idempotency guarantees.

        Warning: the shared engine is event-loop affine. Do not drive it from a
        different event loop than the one used for the first shared() call. Do
        NOT use ``async with`` on a shared instance — the first context exit
        would shut it down for all holders. Use release_shared() for teardown.
        """
        from ._sharing import get_shared_engine

        engine = await get_shared_engine(
            data_dir, llm, embedding=embedding, config=config, assessor_llm=assessor_llm
        )
        if plugin is not None:
            engine._note_participant(plugin)
        return engine

    @classmethod
    def peek_shared(cls, data_dir: str | Path) -> SylanneEngine | None:
        """Return the live shared engine for ``data_dir`` if one exists, else None.

        Public wrapper over the attach-only registry lookup: this NEVER builds
        or starts an engine — use it (or ``wait_shared``) for a pure-listener
        plugin that has no ``llm`` of its own and must never become the copy
        that constructs the engine.
        """
        from ._sharing import peek_shared_engine

        return peek_shared_engine(data_dir)

    @classmethod
    async def wait_shared(
        cls,
        data_dir: str | Path,
        *,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> SylanneEngine | None:
        """Poll for a shared engine on ``data_dir`` until one appears or timeout.

        For a listen-only plugin (no ``llm`` of its own) that may load before
        whichever co-resident plugin will eventually build the engine. This is
        a plain polling loop, not a wake-on-publish channel — a parked-future
        wakeup protocol was considered and rejected: it can leak or drop
        wakeups across vendored copies under different module names, while a
        0.5s default poll is imperceptible for a chat heartbeat and needs no
        new synchronization primitives shared across copies.

        Returns the engine as soon as ``peek_shared`` sees one. Returns
        ``None`` (after logging once at INFO) if ``timeout`` elapses with no
        engine ever appearing — correct by construction: nobody has declared
        compute for this data_dir yet.

        Args:
            timeout: Max seconds to wait. ``None`` waits forever.
            interval: Seconds between polls. Default 0.5s.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            engine = cls.peek_shared(data_dir)
            if engine is not None:
                return engine
            if deadline is not None and loop.time() >= deadline:
                logger.info(
                    "wait_shared(%s) timed out after %.1fs with no shared engine ever "
                    "appearing on this data_dir",
                    data_dir,
                    timeout,
                )
                return None
            await asyncio.sleep(interval)

    @classmethod
    async def release_shared(cls, data_dir: str | Path) -> None:
        """Shut down and remove the shared engine for ``data_dir``.

        Flushes all sessions to disk and frees the registry slot. After this
        returns, a subsequent shared() call for the same path creates a new
        engine. Call only when no other holder is still using the instance;
        wire this into your application's shutdown path (there is no atexit
        auto-flush — see module docs).
        """
        from ._sharing import release_shared_engine

        await release_shared_engine(data_dir)

    @classmethod
    def clear_shared_registry(cls) -> None:
        """Drop all shared registry entries without shutdown. TEST ISOLATION ONLY.

        DANGER: does NOT flush sessions — unpersisted state is lost and live
        engines are orphaned. Never call this in production; use release_shared()
        for real teardown. Safe to call from sync fixtures (no event loop needed).
        """
        from ._sharing import clear_shared_registry

        clear_shared_registry()

    @classmethod
    def is_shared(cls, data_dir: str | Path) -> bool:
        """Return True if a live shared engine is registered for ``data_dir``."""
        from ._sharing import is_shared

        return is_shared(data_dir)

    @classmethod
    def list_shared(cls) -> list[dict[str, str]]:
        """Snapshot of live shared engines: ``[{"data_dir", "status"}, ...]``.

        Diagnostic helper for spotting redundant engines across plugins sharing
        one process. Slots mid-teardown are omitted.
        """
        from ._sharing import list_shared

        return list_shared()

    @classmethod
    def shared_data_dir(cls, explicit: str | Path | None = None) -> Path:
        """Resolve the canonical host-shared ``data_dir`` for co-deployed plugins.

        Returns ``explicit`` if given, else ``$SYLANNE_DATA_DIR``, else
        ``~/.sylanne/shared`` — resolved and absolute, not created. Route every
        plugin's ``shared()`` call through this so independent plugins converge
        on ONE engine instead of each defaulting to its own directory (which
        would never dedup).
        """
        from ._sharing import resolve_shared_data_dir

        return resolve_shared_data_dir(explicit)

    # --- internal ---

    async def _reserve_notification(self, session_id: str) -> _NotificationState:
        from .compute.brain_errors import BrainNotificationBackpressureError

        if not self._notifications_accepting:
            raise BrainNotificationBackpressureError("notification admission is closed")
        state = self._notifications.get(session_id)
        if state is None:
            state = _NotificationState()
            self._notifications[session_id] = state
        await state.slots.acquire(nonblocking=_LISTENER_RECORD.get() is not None)
        if not self._notifications_accepting:
            state.slots.release()
            raise BrainNotificationBackpressureError("notification admission closed while waiting")
        return state

    def _enqueue_reserved_notification(
        self,
        state: _NotificationState,
        *,
        session_id: str,
        generation: int,
        tick_id: int,
        mutation_seq: int,
        surface: Surface,
    ) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        completion = loop.create_future()
        record = _NotificationRecord(
            session_id=session_id,
            generation=generation,
            tick_id=tick_id,
            mutation_seq=mutation_seq,
            surface=surface,
            listeners=tuple(self._listeners),
            completion=completion,
        )
        self._notification_pending.add(completion)
        completion.add_done_callback(self._notification_pending.discard)
        if generation <= state.blocked_through_generation:
            self._finish_notification(state, record)
            return completion
        if not record.listeners:
            record.released = True
            state.slots.release()
            completion.set_result(None)
            return completion
        state.queue.append(record)
        if state.worker is None or state.worker.done():
            state.worker = asyncio.create_task(self._notification_worker(state))
        return completion

    async def _notification_worker(self, state: _NotificationState) -> None:
        try:
            while state.queue:
                record = state.queue.popleft()
                state.current = record
                try:
                    await self._deliver_notification(record)
                finally:
                    self._finish_notification(state, record)
                    state.current = None
        finally:
            current = state.current
            if current is not None:
                self._finish_notification(state, current)
                state.current = None
            while state.queue:
                self._finish_notification(state, state.queue.popleft())
            state.worker = None

    async def _deliver_notification(self, record: _NotificationRecord) -> None:
        for listener in record.listeners:
            if record.stop_after_current_callback:
                break
            token = _LISTENER_RECORD.set(record)
            try:
                returned = listener(record.session_id, record.surface)
                if inspect.isawaitable(returned):
                    listener_task = asyncio.ensure_future(returned)
                    self._listener_tasks.add(listener_task)
                    listener_task.add_done_callback(self._listener_task_done)
                    done, _ = await asyncio.wait(
                        (listener_task,),
                        timeout=_LISTENER_TIMEOUT_SECONDS,
                    )
                    if not done:
                        listener_task.cancel()
                        logger.warning(
                            "Listener timed out after %.1fs for session %r",
                            _LISTENER_TIMEOUT_SECONDS,
                            record.session_id,
                        )
                    else:
                        try:
                            listener_task.result()
                        except asyncio.CancelledError:
                            logger.warning("Listener cancelled itself", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Listener error", exc_info=True)
            finally:
                _LISTENER_RECORD.reset(token)
            if record.stop_after_current_callback:
                break

    def _listener_task_done(self, future: asyncio.Future[Any]) -> None:
        self._listener_tasks.discard(future)
        if future.cancelled():
            return
        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _finish_notification(
        state: _NotificationState,
        record: _NotificationRecord,
    ) -> None:
        if record.released:
            return
        record.released = True
        state.slots.release()
        if not record.completion.done():
            record.completion.set_result(None)

    async def drain_notifications(self, *, timeout: float | None = None) -> None:
        """Wait until every notification accepted so far has finished delivery."""
        loop = asyncio.get_running_loop()
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0.0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            pending = tuple(future for future in self._notification_pending if not future.done())
            if not pending:
                return
            remaining = None if deadline is None else max(0.0, deadline - loop.time())
            done, _ = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("notification drain timed out")

    async def _barrier_notifications(
        self,
        session_id: str,
        *,
        through_generation: int,
        timeout: float = 30.0,
    ) -> None:
        state = self._notifications.get(session_id)
        if state is None:
            return
        state.blocked_through_generation = max(
            state.blocked_through_generation,
            through_generation,
        )
        current = state.current
        wait_for_current: asyncio.Future[None] | None = None
        if current is not None and current.generation <= through_generation:
            current.stop_after_current_callback = True
            if current is not _LISTENER_RECORD.get():
                wait_for_current = current.completion

        retained: deque[_NotificationRecord] = deque()
        while state.queue:
            record = state.queue.popleft()
            if record.generation <= through_generation:
                record.stop_after_current_callback = True
                self._finish_notification(state, record)
            else:
                retained.append(record)
        state.queue = retained

        if wait_for_current is None or wait_for_current.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(wait_for_current), timeout=timeout)
        except TimeoutError:
            worker = state.worker
            if worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def _shutdown_notifications(self) -> None:
        from .compute.brain_errors import BrainNotificationBackpressureError

        self._notifications_accepting = False
        admission_error = BrainNotificationBackpressureError(
            "engine is shutting down notification admission"
        )
        for state in self._notifications.values():
            state.slots.close(admission_error)
        try:
            await self.drain_notifications(timeout=30.0)
        except TimeoutError:
            workers = [
                state.worker
                for state in self._notifications.values()
                if state.worker is not None and not state.worker.done()
            ]
            for worker in workers:
                worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
        for listener_task in tuple(self._listener_tasks):
            listener_task.cancel()
        self._notifications.clear()
        self._notification_pending.clear()

    def _session_lock(self, session_id: str) -> _SessionLease:
        return _SessionLease(self, session_id)

    async def _retain_session_entry(self, session_id: str) -> _SessionEntry:
        async with self._host_condition:
            while True:
                entry = self._session_entries.get(session_id)
                if entry is None:
                    entry = _SessionEntry(session_id=session_id)
                    self._session_entries[session_id] = entry
                    self._locks[session_id] = entry.lock
                if entry.eviction_reserved:
                    await self._host_condition.wait()
                    continue
                entry.references += 1
                entry.waiters += 1
                self._session_entries.move_to_end(session_id)
                if session_id in self._hosts:
                    self._hosts.move_to_end(session_id)
                return entry

    async def _promote_session_owner(self, entry: _SessionEntry) -> None:
        async with self._host_condition:
            entry.waiters -= 1
            entry.owners += 1

    async def _release_session_waiter(self, entry: _SessionEntry) -> None:
        async with self._host_condition:
            entry.waiters -= 1
            entry.references -= 1
            self._drop_empty_entry_locked(entry)
            self._host_condition.notify_all()

    async def _release_session_owner(self, entry: _SessionEntry) -> None:
        async with self._host_condition:
            entry.owners -= 1
            entry.references -= 1
            self._drop_empty_entry_locked(entry)
            self._schedule_host_trim_locked()
            self._host_condition.notify_all()

    def _drop_empty_entry_locked(self, entry: _SessionEntry) -> None:
        if entry.references or entry.host is not None or entry.eviction_reserved:
            return
        if self._session_entries.get(entry.session_id) is entry:
            self._session_entries.pop(entry.session_id, None)
            self._locks.pop(entry.session_id, None)

    def _schedule_host_trim_locked(self) -> None:
        task = self._host_trim_task
        if len(self._hosts) <= self._config.brain_compute.hot_session_limit:
            return
        if task is None or task.done():
            self._host_trim_task = asyncio.create_task(self._trim_overflow_hosts())

    async def _trim_overflow_hosts(self) -> None:
        failed = False
        deferred = False
        try:
            while True:
                candidate: tuple[_SessionEntry, SylanneHost, int] | None = None
                async with self._host_condition:
                    if len(self._hosts) <= self._config.brain_compute.hot_session_limit:
                        return
                    for entry in self._session_entries.values():
                        if (
                            entry.host is not None
                            and entry.references == 0
                            and not entry.eviction_reserved
                        ):
                            entry.eviction_reserved = True
                            candidate = (entry, entry.host, entry.epoch)
                            break
                    if candidate is None:
                        deferred = True
                        return
                try:
                    await self._evict_reserved_host(*candidate)
                except Exception:
                    failed = True
                    logger.warning("background Host eviction failed", exc_info=True)
                    return
        finally:
            async with self._host_condition:
                if self._host_trim_task is asyncio.current_task():
                    self._host_trim_task = None
                if not failed and not deferred:
                    self._schedule_host_trim_locked()
                self._host_condition.notify_all()

    async def _detach_entry_host(self, entry: _SessionEntry, *, flush: bool) -> None:
        async with self._host_condition:
            host = entry.host
            if host is None:
                return
            entry.epoch += 1
            entry.host = None
            entry.eviction_reserved = False
            if self._hosts.get(entry.session_id) is host:
                self._hosts.pop(entry.session_id, None)
            self._host_condition.notify_all()
        await asyncio.to_thread(host.close, flush=flush)

    async def _discard_cached_host(self, session_id: str) -> None:
        async with self._session_lock(session_id) as entry:
            await self._detach_entry_host(entry, flush=False)

    async def set_relationship(self, session_id: str, relationship: float) -> None:
        """v26 D2：host 显式供给会话的关系相位标量 R ∈ [0,1]（标定呈报 D2 选项 a）。

        R 的语义（"和这个人处到哪一步了"）由宿主定义，引擎只消费：情感均衡 Φ_eq 的
        warmth 行随 R 上移（关系越深、常驻越暖）。仅在 affect_dynamics_enabled 开启时
        有可观测效果；未调用时保持 0.5（今日行为）。越界抛 ValueError；设置随快照持久化。
        """
        r = float(relationship)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"relationship must be in [0,1], got {relationship!r}")
        await self._ensure_started()
        async with self._session_lock(session_id):
            host = await self._get_or_create_host(session_id)
            host.kernel.computation.set_relationship(r)
            host._pending_snapshot = host.kernel.snapshot()
            host._dirty = True
            host._flush()

    def _build_host(self, session_id: str) -> SylanneHost:
        """Construct a host (blocking cold-load disk IO happens in __post_init__)."""
        from .compute import SylanneHost

        return SylanneHost(
            root=self._data_dir,
            session_key=session_id,
            profile=self._config.profile(),
            telemetry_sink=self._telemetry_sink,
            pel_enabled=self._config.pel_core_enabled,
            affect_enabled=self._config.affect_dynamics_enabled,
            affect_takeover=self._config.affect_takeover,
            affect_slowchannel=self._config.affect_slowchannel_enabled,
            affect_plasticity=self._config.affect_plasticity_enabled,
            affect_full_takeover=self._config.affect_full_takeover,
            brain_compute=self._config.brain_compute,
            brain_store=self._brain_store,
        )

    async def _get_or_create_host(self, session_id: str) -> SylanneHost:
        """Get the cached session host, cold-loading it off the event loop.

        v2.6.0 T-Persist: first-touch construction reads the session snapshot from
        disk synchronously (``SylanneAlphaHost.__post_init__`` -> ``AlphaRuntime.load``).
        Hoist that blocking IO via ``asyncio.to_thread`` so a cold session does not
        stall the loop. The dict insert stays on the loop thread (no cross-session
        race); same-id concurrency is already serialized by the per-session lock,
        so the post-await re-check simply joins a host built while we awaited.
        """
        async with self._host_condition:
            entry = self._session_entries.get(session_id)
            if entry is None or entry.references <= 0:
                raise RuntimeError("session Host access requires an active session lease")
            host = entry.host
            if host is not None:
                self._session_entries.move_to_end(session_id)
                self._hosts.move_to_end(session_id)
                return host
            epoch = entry.epoch

        await self._reserve_host_build_slot()
        try:
            built = await asyncio.to_thread(self._build_host, session_id)
        except BaseException:
            await self._release_host_build_slot()
            raise

        stale = False
        existing: SylanneHost | None = None
        async with self._host_condition:
            self._host_build_reservations -= 1
            current = self._session_entries.get(session_id)
            if current is not entry or entry.epoch != epoch:
                stale = True
                existing = None if current is None else current.host
            elif entry.host is None:
                entry.host = built
                self._hosts[session_id] = built
                self._session_entries.move_to_end(session_id)
                self._hosts.move_to_end(session_id)
            else:
                stale = True
                existing = entry.host
            self._host_condition.notify_all()

        if stale:
            await asyncio.to_thread(built.close, flush=False)
            if existing is not None:
                return existing
            return await self._get_or_create_host(session_id)
        return built

    async def _reserve_host_build_slot(self) -> None:
        hot_limit = self._config.brain_compute.hot_session_limit
        overflow_limit = hot_limit + 8
        while True:
            candidate: tuple[_SessionEntry, SylanneHost, int] | None = None
            async with self._host_condition:
                charged = len(self._hosts) + self._host_build_reservations
                if charged < hot_limit:
                    self._host_build_reservations += 1
                    return
                for entry in self._session_entries.values():
                    if (
                        entry.host is not None
                        and entry.references == 0
                        and not entry.eviction_reserved
                    ):
                        entry.eviction_reserved = True
                        candidate = (entry, entry.host, entry.epoch)
                        break
                if candidate is None:
                    if charged < overflow_limit:
                        self._host_build_reservations += 1
                        return
                    await self._host_condition.wait()
                    continue
            await self._evict_reserved_host(*candidate)

    async def _release_host_build_slot(self) -> None:
        async with self._host_condition:
            self._host_build_reservations -= 1
            self._host_condition.notify_all()

    async def _evict_reserved_host(
        self,
        entry: _SessionEntry,
        host: SylanneHost,
        epoch: int,
    ) -> None:
        try:
            await asyncio.to_thread(host.close, flush=True)
        except BaseException:
            async with self._host_condition:
                if self._session_entries.get(entry.session_id) is entry:
                    entry.eviction_reserved = False
                self._host_condition.notify_all()
            raise

        async with self._host_condition:
            if (
                self._session_entries.get(entry.session_id) is entry
                and entry.eviction_reserved
                and entry.references == 0
                and entry.epoch == epoch
                and entry.host is host
            ):
                entry.host = None
                entry.eviction_reserved = False
                self._hosts.pop(entry.session_id, None)
                self._session_entries.pop(entry.session_id, None)
                self._locks.pop(entry.session_id, None)
            elif self._session_entries.get(entry.session_id) is entry:
                entry.eviction_reserved = False
            self._host_condition.notify_all()

    def _ctx_fingerprint(
        self,
        confidence: float | None,
        flags: list[str] | None,
        values: dict[str, float] | None,
    ) -> tuple[Any, ...]:
        """Hashable fingerprint of a submit() call's context, for divergence checks."""
        return (
            confidence,
            tuple(sorted(flags)) if flags else (),
            tuple(sorted((values or {}).items())),
        )

    def _prune_submissions(self) -> None:
        """Lazily evict stale/overflow completed entries from the dedup table.

        Runs at the top of every ``submit()`` call. Completed entries older
        than ``submit_window_seconds`` are evicted; beyond that, completed
        entries in excess of ``submit_max_entries`` are evicted oldest-first.
        IN-FLIGHT entries (``done_at is None``) are never touched by either
        rule — only a finished computation ages out.
        """
        cutoff = time.time() - self._config.submit_window_seconds
        seen: set[int] = set()
        completed: list[_Submission] = []
        for entry in list(self._submissions.values()):
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            if entry.done_at is None:
                continue
            if entry.done_at < cutoff:
                self._evict_submission(entry)
            else:
                completed.append(entry)
        overflow = len(completed) - self._config.submit_max_entries
        if overflow > 0:
            completed.sort(key=lambda e: e.done_at or 0.0)
            for entry in completed[:overflow]:
                self._evict_submission(entry)

    def _evict_submission(self, entry: _Submission) -> None:
        """Remove every key of ``entry`` from the live table into the recent-evicted LRU."""
        for key in entry.keys:
            if self._submissions.get(key) is entry:
                del self._submissions[key]
            self._recent_evicted[key] = None
            self._recent_evicted.move_to_end(key)
        while len(self._recent_evicted) > self._config.submit_max_entries:
            self._recent_evicted.popitem(last=False)

    def _on_submission_done(self, task: asyncio.Task[Surface]) -> None:
        """Done-callback: stamp done_at, evict a failed task's keys immediately.

        Always retrieves a failed task's exception here so it is never
        reported as "exception never retrieved" even in the (rare) case where
        every awaiter's own await got cancelled before reaching the result.
        """
        for entry in {id(e): e for e in self._submissions.values()}.values():
            if entry.task is task:
                entry.done_at = time.time()
                if not task.cancelled() and task.exception() is not None:
                    # Poison must not stick for the rest of the window.
                    for key in entry.keys:
                        if self._submissions.get(key) is entry:
                            del self._submissions[key]
                break

    def _cancel_and_clear_submissions(self) -> None:
        """Cancel every in-flight submit() task and drop the dedup tables.

        Used on shutdown() and on loop-rebind (see ``_sharing.get_shared_engine``)
        — tasks created on a now-dead event loop cannot be awaited further, and
        letting a rebound engine keep pointing at them would hang the next
        ``submit()`` that tries to join. Cheap and idempotent.
        """
        for entry in {id(e): e for e in self._submissions.values()}.values():
            if not entry.task.done():
                entry.task.cancel()
        self._submissions.clear()
        self._recent_evicted.clear()

    def _note_participant(self, plugin: str) -> dict[str, Any]:
        """Register (or fetch) ``plugin``'s participants-registry entry.

        Diagnostics only — see ``participants()``/HARD RULE in module docs.
        Logs once at INFO on first attach for a given plugin name.
        """
        info = self._participants.get(plugin)
        if info is not None:
            return info
        try:
            import importlib

            from ._sharing import _self_identity

            copy_id = _self_identity().get("copy_id")
        except Exception:
            copy_id = None
        try:
            sdk_version = getattr(
                importlib.import_module(__package__ or "sylanne_core"), "__version__", "0+unknown"
            )
        except Exception:
            sdk_version = "0+unknown"
        info = {
            "copy_id": copy_id,
            "sdk_version": sdk_version,
            "first_seen": time.time(),
            "submits": 0,
            "joins": 0,
        }
        self._participants[plugin] = info
        logger.info("plugin %s attached to shared engine on %s", plugin, self._data_dir)
        return info

    def _note_submit_outcome(self, plugin: str | None, *, joined: bool) -> None:
        """Increment ``plugin``'s submits/joins counter, if identity was given.

        Diagnostics only: this is the only place submit() touches the
        participants registry, and it never branches submit()'s own dedup/join
        decision on anything read from it.
        """
        if plugin is None:
            return
        info = self._note_participant(plugin)
        if joined:
            info["joins"] += 1
        else:
            info["submits"] += 1

    def _build_telemetry_sink(self) -> DistillationSink | None:
        """Construct the shared distillation sink if opted in, else None.

        One sink per engine (a single append file) shared across all sessions.
        When the flag is off this returns None and the per-tick path stays
        zero-cost. Never raises into construction — a misconfigured sink
        disables collection rather than breaking the engine.
        """
        cfg = self._config
        if not cfg.training_data_sink:
            return None
        import secrets

        from .telemetry import DistillationSink

        base = self._data_dir / "telemetry"
        fname = Path(cfg.training_data_path or "distill_corpus.jsonl").name
        salt = cfg.training_data_salt
        if not salt:
            salt = secrets.token_hex(8)
            logger.warning(
                "training_data_salt is empty; using a per-process random salt — "
                "cross-run session grouping will be unstable"
            )
        try:
            return DistillationSink(
                enabled=True,
                path=base / (fname or "distill_corpus.jsonl"),
                salt=salt,
                base_dir=base,
            )
        except (OSError, ValueError):
            logger.warning(
                "could not open distillation sink; data collection disabled",
                exc_info=True,
            )
            return None

    async def _assess(self, text: str) -> dict[str, Any] | None:
        if not text.strip():
            return None
        try:
            from .assessor import assess_text

            # v26 A.1：仅在 takeover（Gate B）开时向 LLM 直出 intent——Gate A/关闭态
            # prompt 逐字节不变（intent 通电会点燃遗留手写意图规则、破影子期契约）。
            result = await assess_text(
                text,
                self._assessor_llm or self._llm,
                want_intent=(self._config.affect_takeover and self._config.affect_dynamics_enabled),
            )
            if result and result.pop("_degraded", False):
                if self._status == "running":
                    self._status = "degraded"
            return result
        except Exception:
            if self._status == "running":
                self._status = "degraded"
            return None

    def _to_surface(self, session_id: str, host: SylanneHost, raw: dict[str, Any]) -> Surface:
        from .adapter import to_surface

        return to_surface(session_id, host, raw, diagnostics=self._config.diagnostics)
