"""Typed brain backend boundary with explicit programmatic registration."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class BrainStepRequest:
    event_id: str
    tick_id: int
    expected_state_version: int
    event: tuple[float, ...]
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class BrainStepResult:
    request_id: str
    event_id: str
    tick_id: int
    expected_state_version: int
    state_version: int
    proposal: tuple[float, ...]
    eligibility: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BrainFeedbackRequest:
    feedback_id: str
    target_tick: int
    expected_state_version: int
    value: float
    confidence: float
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class BrainFeedbackResult:
    request_id: str
    feedback_id: str
    target_tick: int
    expected_state_version: int
    state_version: int
    applied_synapses: int


class BrainCheckpointMismatchError(Exception):
    """A persisted opaque token cannot initialize the selected backend."""


class BrainBackend(Protocol):
    """Lifecycle implemented by public or explicitly registered C backends."""

    is_process_isolated: bool

    def open(
        self,
        session_digest: bytes,
        checkpoint_token: bytes | None,
        *,
        timeout_ms: int,
    ) -> None: ...

    def step(self, request: BrainStepRequest, *, timeout_ms: int) -> BrainStepResult: ...

    def apply_feedback(
        self,
        request: BrainFeedbackRequest,
        *,
        timeout_ms: int,
    ) -> BrainFeedbackResult: ...

    def checkpoint(self, *, timeout_ms: int) -> bytes: ...

    def abort(self, reason: str) -> None: ...

    def close(self) -> None: ...


BrainBackendFactory = Callable[[], BrainBackend]


def _builtin_backend_factory() -> BrainBackend:
    """Reserve the built-in name until the Brain C implementation lands."""
    raise NotImplementedError("the builtin brain backend is not implemented yet")


_BACKEND_FACTORIES: dict[str, BrainBackendFactory] = {"builtin": _builtin_backend_factory}
_BACKEND_LOCK = threading.RLock()


def register_brain_backend(name: str, factory: BrainBackendFactory) -> None:
    """Register a factory explicitly; repeated registration is identity-idempotent."""
    if not isinstance(name, str) or not name:
        raise ValueError("brain backend name must be a nonempty string")
    if not callable(factory):
        raise TypeError("brain backend factory must be callable")
    with _BACKEND_LOCK:
        existing = _BACKEND_FACTORIES.get(name)
        if existing is factory:
            return
        if existing is not None:
            raise ValueError(f"brain backend {name!r} is already registered")
        _BACKEND_FACTORIES[name] = factory


def get_brain_backend_factory(name: str) -> BrainBackendFactory:
    """Return the factory registered under ``name``, or raise ``KeyError``."""
    with _BACKEND_LOCK:
        return _BACKEND_FACTORIES[name]
