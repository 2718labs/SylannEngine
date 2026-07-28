"""v2.6.0 T-Persist 契约：base 版本号（休眠）、feedback 单调性修复、冷载 hoist。

对照 docs/design/v26-upgrade-path.md §2 T-PERSIST。守护：
- ``_e_ver`` 随每次 base 变异自增，仅在启用时落盘/复原（byte-identical off）；
- feedback()（timestamp=0.0）**不再**清零 ``_last_step_time``（persist #1 修复），
  真实步之后的静默奖励愈合不再被吞；
- ``_get_or_create_host`` 变为协程、把冷载阻塞 IO 挪出事件循环，且缓存/并发语义不变。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sylanne_core import SylanneEngine
from sylanne_core.compute.scar_algebra import ScarredState
from sylanne_core.config import SylanneConfig

_EV = [0.2, 0.1, 0.3, -0.1, 0.2, -0.2, 0.1, 0.0]


class TestBaseVersion:
    def test_ver_increments_on_step(self) -> None:
        st = ScarredState(n_dims=8, affect_enabled=True)
        assert st._e_ver == 0
        st.step(_EV, timestamp=100.0)
        st.step(_EV, timestamp=200.0)
        assert st._e_ver == 2

    def test_ver_persisted_only_when_enabled(self) -> None:
        on = ScarredState(n_dims=8, affect_enabled=True)
        on.step(_EV, timestamp=100.0)
        d = on.to_dict()
        assert d["e_ver"] == 1
        restored = ScarredState.from_dict(d, affect_enabled=True)
        assert restored._e_ver == 1

        off = ScarredState(n_dims=8, affect_enabled=False)
        off.step(_EV, timestamp=100.0)
        assert "e_ver" not in off.to_dict()  # byte-identical legacy snapshot

    def test_legacy_snapshot_defaults_ver_zero(self) -> None:
        on = ScarredState(n_dims=8, affect_enabled=True)
        on.step(_EV, timestamp=100.0)
        d = on.to_dict()
        del d["e_ver"]
        assert ScarredState.from_dict(d, affect_enabled=True)._e_ver == 0


class TestFeedbackMonotonicity:
    def test_zero_timestamp_does_not_clobber_clock(self) -> None:
        # The monotonicity fix is GATED on affect_enabled (byte-identical off).
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.step(_EV, timestamp=1000.0)
        assert st._last_step_time == 1000.0
        # feedback-style call: timestamp=0.0 must NOT reset the healing clock.
        st.step(_EV, timestamp=0.0)
        assert st._last_step_time == 1000.0
        # a later real step still measures elapsed from the true last real time.
        st.step(_EV, timestamp=2000.0)
        assert st._last_step_time == 2000.0

    def test_off_preserves_legacy_clobber(self) -> None:
        # affect OFF => exact legacy behavior: ts=0 (feedback) still zeroes the clock.
        st = ScarredState(n_dims=8)  # affect disabled
        st.step(_EV, timestamp=1000.0)
        st.step(_EV, timestamp=0.0)
        assert st._last_step_time == 0.0  # legacy byte-identical (unconditional assign)

    def test_silence_bonus_survives_feedback(self) -> None:
        # Build a scar, then interleave a feedback (ts=0) between two real steps
        # separated by a long silence. With the fix, the second real step still
        # grants silence-bonus healing (clock not zeroed by feedback).
        kept = ScarredState(n_dims=8, affect_enabled=True)
        kept.wound_threshold = 0.05
        kept.step([1.0] * 8, timestamp=100.0)  # form scars
        ticks_before = [s.ticks_in_stage for s in kept.scars]
        kept.step([0.0] * 8, timestamp=0.0)  # feedback: no clock reset
        kept.step([0.0] * 8, timestamp=100.0 + 3600.0)  # 1h later -> bonus ticks
        ticks_after = [s.ticks_in_stage for s in kept.scars]
        # At least one scar accrued more than the single non-bonus tick would give.
        assert sum(ticks_after) > sum(ticks_before) + len(kept.scars)


class TestColdLoadHoist:
    @pytest.mark.asyncio
    async def test_get_or_create_host_is_async_and_caches(self, tmp_path: Path) -> None:
        engine = SylanneEngine(data_dir=tmp_path, llm=AsyncMock(return_value="ok"))
        await engine.start()
        # v26: direct host access now requires an active session lease; hold one
        # per session so the cold-load/caching semantics can be probed directly.
        async with engine._session_lock("s1"):
            h1 = await engine._get_or_create_host("s1")
            h2 = await engine._get_or_create_host("s1")
            assert h1 is h2  # cached, one build
            async with engine._session_lock("s2"):
                h3 = await engine._get_or_create_host("s2")
                assert h3 is not h1  # distinct session
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_process_still_works_through_async_host(self, tmp_path: Path) -> None:
        engine = SylanneEngine(
            data_dir=tmp_path,
            llm=AsyncMock(return_value="ok"),
            config=SylanneConfig(assessor_enabled=False),
        )
        await engine.start()
        surface = await engine.process("sess", "你好")
        assert "decision" in surface
        await engine.shutdown()
