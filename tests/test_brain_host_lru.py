"""Bounded Engine Host ownership and failure-discard contracts."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from sylanne_core import SylanneEngine
from sylanne_core.compute.brain_errors import BrainDurabilityError
from sylanne_core.compute.host import SylanneAlphaHost
from sylanne_core.config import BrainComputeConfig, SylanneConfig
from sylanne_core.types import Surface


def _config(*, hot_session_limit: int) -> SylanneConfig:
    return SylanneConfig(
        assessor_enabled=False,
        brain_compute=BrainComputeConfig(
            enabled=True,
            hot_session_limit=hot_session_limit,
        ),
    )


def _brain_event(surface: Surface) -> dict[str, Any]:
    event = surface["pipeline"].get("brain_event")
    assert isinstance(event, dict)
    return cast(dict[str, Any], event)


@pytest.mark.asyncio
async def test_lru_evicts_oldest_idle_host_and_restores_durable_state(
    tmp_path: Path,
) -> None:
    engine = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=_config(hot_session_limit=2),
    )
    await engine.start()
    try:
        await engine.process("a", "a1", event_id="a1")
        await engine.process("b", "b1", event_id="b1")
        await engine.state("a")
        await engine.process("c", "c1", event_id="c1")

        assert list(engine._hosts) == ["a", "c"]
        assert not engine.exists("b")

        restored = await engine.process("b", "b2", event_id="b2")
        assert _brain_event(restored)["tick_id"] == 2
        assert list(engine._hosts) == ["c", "b"]
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_strict_brain_failure_discards_hot_host_without_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=_config(hot_session_limit=2),
    )
    await engine.start()
    await engine.process("session", "first", event_id="event-1")
    failed_host = engine._hosts["session"]
    flush_calls = 0
    original_flush = SylanneAlphaHost.flush
    original_request = SylanneAlphaHost.on_request

    def track_flush(self: SylanneAlphaHost) -> None:
        nonlocal flush_calls
        if self is failed_host:
            flush_calls += 1
        original_flush(self)

    def fail_request(self: SylanneAlphaHost, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self is failed_host:
            raise BrainDurabilityError("strict failure")
        return original_request(self, *args, **kwargs)

    monkeypatch.setattr(SylanneAlphaHost, "flush", track_flush)
    monkeypatch.setattr(SylanneAlphaHost, "on_request", fail_request)
    try:
        with pytest.raises(BrainDurabilityError, match="strict failure"):
            await engine.process("session", "second", event_id="event-2")

        assert "session" not in engine._hosts
        assert flush_calls == 0
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_phase_one_durability_failure_also_discards_cached_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=_config(hot_session_limit=2),
    )
    await engine.start()
    await engine.process("session", "first", event_id="event-1")
    failed_host = engine._hosts["session"]
    store = engine._brain_store
    assert store is not None
    flush_calls = 0
    original_flush = SylanneAlphaHost.flush

    def track_flush(self: SylanneAlphaHost) -> None:
        nonlocal flush_calls
        if self is failed_host:
            flush_calls += 1
        original_flush(self)

    def fail_lookup(*_args: Any, **_kwargs: Any) -> Any:
        raise BrainDurabilityError("phase one failed")

    monkeypatch.setattr(SylanneAlphaHost, "flush", track_flush)
    monkeypatch.setattr(store, "lookup_event_receipt", fail_lookup)
    try:
        with pytest.raises(BrainDurabilityError, match="phase one failed"):
            await engine.process("session", "second", event_id="event-2")

        assert "session" not in engine._hosts
        assert flush_calls == 0
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_eight_host_overflow_is_hard_ceiling_and_shrinks_to_hot_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=_config(hot_session_limit=1),
    )
    release = threading.Event()
    counter_lock = threading.Lock()
    entered = 0
    original_request = SylanneAlphaHost.on_request

    def blocking_request(
        self: SylanneAlphaHost,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal entered
        with counter_lock:
            entered += 1
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release blocked Host operation")
        return original_request(self, *args, **kwargs)

    monkeypatch.setattr(SylanneAlphaHost, "on_request", blocking_request)
    await engine.start()
    accepted = [
        asyncio.create_task(engine.process(f"s{index}", "event", event_id=f"e{index}"))
        for index in range(9)
    ]
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while True:
            with counter_lock:
                current_entered = entered
            if current_entered == 9:
                break
            if loop.time() >= deadline:
                raise TimeoutError("nine overflow operations did not enter")
            await asyncio.sleep(0.01)

        assert len(engine._hosts) + engine._host_build_reservations <= 9
        tenth = asyncio.create_task(engine.process("s9", "event", event_id="e9"))
        await asyncio.sleep(0.05)
        with counter_lock:
            assert entered == 9
        assert not tenth.done()

        release.set()
        await asyncio.wait_for(asyncio.gather(*accepted, tenth), timeout=10.0)
        assert entered == 10
        deadline = loop.time() + 5.0
        while len(engine._hosts) > 1 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert len(engine._hosts) <= 1
    finally:
        release.set()
        await engine.shutdown()


@pytest.mark.asyncio
async def test_overflow_shrinks_without_a_later_host_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=_config(hot_session_limit=1),
    )
    release = threading.Event()
    entered_lock = threading.Lock()
    entered = 0
    original_request = SylanneAlphaHost.on_request

    def blocking_request(
        self: SylanneAlphaHost,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal entered
        with entered_lock:
            entered += 1
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release blocked Host operation")
        return original_request(self, *args, **kwargs)

    monkeypatch.setattr(SylanneAlphaHost, "on_request", blocking_request)
    await engine.start()
    tasks = [
        asyncio.create_task(engine.process(f"s{index}", "event", event_id=f"e{index}"))
        for index in range(9)
    ]
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while True:
            with entered_lock:
                current_entered = entered
            if current_entered == 9:
                break
            if loop.time() >= deadline:
                raise TimeoutError("overflow operations did not all enter")
            await asyncio.sleep(0.01)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)

        deadline = loop.time() + 5.0
        while len(engine._hosts) > 1 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert len(engine._hosts) <= 1
    finally:
        release.set()
        await engine.shutdown()


@pytest.mark.asyncio
async def test_destroy_persists_tombstone_across_restart_and_is_repeatable(
    tmp_path: Path,
) -> None:
    config = _config(hot_session_limit=2)
    first = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=config,
    )
    await first.start()
    await first.process("session", "event", event_id="event-1")
    await first.destroy("session")
    assert not first.exists("session")
    await first.shutdown()

    second = SylanneEngine(
        data_dir=tmp_path,
        llm=AsyncMock(return_value="unused"),
        config=config,
    )
    await second.start()
    try:
        await second.destroy("session")
        with pytest.raises(BrainDurabilityError, match="destroyed"):
            await second.process("session", "must not revive", event_id="event-2")
        assert not second.exists("session")
    finally:
        await second.shutdown()
