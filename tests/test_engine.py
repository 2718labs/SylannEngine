"""Tests for sylanne_core.engine module (public SDK API)."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sylanne_core import SylanneEngine
from sylanne_core.config import SylanneConfig


@pytest.fixture
def engine(tmp_path: Path) -> SylanneEngine:
    llm = AsyncMock(return_value="mocked response")
    return SylanneEngine(data_dir=tmp_path, llm=llm)


class TestEngineLifecycle:
    @pytest.mark.asyncio
    async def test_start(self, engine: SylanneEngine, tmp_path: Path):
        await engine.start()
        assert engine.status == "running"
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_shutdown(self, engine: SylanneEngine):
        await engine.start()
        await engine.shutdown()
        assert engine.status == "closed"

    @pytest.mark.asyncio
    async def test_double_shutdown(self, engine: SylanneEngine):
        await engine.start()
        await engine.shutdown()
        await engine.shutdown()
        assert engine.status == "closed"


class TestEngineProcess:
    @pytest.mark.asyncio
    async def test_process_returns_surface(self, engine: SylanneEngine):
        await engine.start()
        surface = await engine.process("s1", "hello")
        assert surface["session_id"] == "s1"
        assert "state" in surface
        assert "decision" in surface
        assert "guard" in surface

    @pytest.mark.asyncio
    async def test_process_multiple_sessions(self, engine: SylanneEngine):
        await engine.start()
        s1 = await engine.process("s1", "hi")
        s2 = await engine.process("s2", "hello")
        assert s1["session_id"] == "s1"
        assert s2["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_process_with_flags(self, engine: SylanneEngine):
        await engine.start()
        surface = await engine.process("s1", "ouch", flags=["hurt"])
        assert surface["decision"]["action"] in {
            "wait",
            "explore",
            "express",
            "reach_out",
            "repair",
            "withdraw",
            "recover",
        }

    @pytest.mark.asyncio
    async def test_process_empty_text(self, engine: SylanneEngine):
        await engine.start()
        surface = await engine.process("s1", "")
        assert "state" in surface


class TestEngineTick:
    @pytest.mark.asyncio
    async def test_tick(self, engine: SylanneEngine):
        await engine.start()
        surface = await engine.tick("s1")
        assert "state" in surface

    @pytest.mark.asyncio
    async def test_tick_with_flags(self, engine: SylanneEngine):
        await engine.start()
        surface = await engine.tick("s1", flags=["idle"])
        assert "state" in surface


class TestEngineState:
    @pytest.mark.asyncio
    async def test_state(self, engine: SylanneEngine):
        await engine.start()
        await engine.process("s1", "hello")
        surface = await engine.state("s1")
        assert surface["session_id"] == "s1"


class TestEngineReset:
    @pytest.mark.asyncio
    async def test_reset(self, engine: SylanneEngine):
        await engine.start()
        await engine.process("s1", "hello")
        await engine.reset("s1")
        surface = await engine.state("s1")
        assert surface["turns"] == 0

    @pytest.mark.asyncio
    async def test_destroy(self, engine: SylanneEngine):
        await engine.start()
        await engine.process("s1", "hello")
        await engine.destroy("s1")
        surface = await engine.state("s1")
        assert surface["turns"] == 0


class TestEngineHealth:
    @pytest.mark.asyncio
    async def test_health_running(self, engine: SylanneEngine):
        await engine.start()
        h = engine.health()
        assert h["status"] == "running"
        assert h["llm_configured"] is True

    def test_health_init(self, engine: SylanneEngine):
        h = engine.health()
        assert h["status"] == "init"


class TestEngineListeners:
    @pytest.mark.asyncio
    async def test_on_listener_called(self, engine: SylanneEngine):
        await engine.start()
        received = []
        engine.on(lambda sid, s: received.append((sid, s)))
        await engine.process("s1", "hello")
        assert len(received) == 1
        assert received[0][0] == "s1"

    @pytest.mark.asyncio
    async def test_off_removes_listener(self, engine: SylanneEngine):
        await engine.start()
        received = []

        def listener(sid, s):
            received.append(sid)

        engine.on(listener)
        engine.off(listener)
        await engine.process("s1", "hello")
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_async_listener(self, engine: SylanneEngine):
        await engine.start()
        received = []

        async def async_listener(sid, s):
            received.append(sid)

        engine.on(async_listener)
        await engine.process("s1", "hello")
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_listener_survives_reset(self, engine: SylanneEngine):
        # Regression (P0): in non-brain (default) mode, reset() must not
        # permanently silence push-notification listeners. Non-brain
        # notifications are all enqueued with generation 0, so a persistent
        # generation-0 barrier used to drop every post-reset notification.
        await engine.start()
        received: list[str] = []
        engine.on(lambda sid, s: received.append(sid))
        await engine.process("s1", "hello")
        assert len(received) == 1
        await engine.reset("s1")
        await engine.process("s1", "again")
        assert len(received) == 2  # would stay 1 before the fix

    @pytest.mark.asyncio
    async def test_listener_survives_destroy(self, engine: SylanneEngine):
        # Regression (P0): destroy() delegates to reset() in non-brain mode, so
        # it must not silence listeners either.
        await engine.start()
        received: list[str] = []
        engine.on(lambda sid, s: received.append(sid))
        await engine.process("s1", "hello")
        assert len(received) == 1
        await engine.destroy("s1")
        await engine.process("s1", "again")
        assert len(received) == 2


class TestEngineConfig:
    @pytest.mark.asyncio
    async def test_assessor_disabled(self, tmp_path: Path):
        config = SylanneConfig(assessor_enabled=False)
        llm = AsyncMock(return_value="response")
        engine = SylanneEngine(data_dir=tmp_path, llm=llm, config=config)
        await engine.start()
        await engine.process("s1", "hello")
        llm.assert_not_called()
