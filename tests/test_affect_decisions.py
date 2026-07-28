"""v26 决策落地契约：D1(b) E 律全权（affect_full_takeover）+ D2 R 接活（set_relationship）。

对照 docs/design/affect-calibration-memo.md（决策：D1=b、D2=a、D3=a）。守护：
- 全权下静默 tick = decay-only：隔夜残留 → Φ_eq（MLP 像差消失，h 先验变活杠杆）；
- 全权下伤痕照常形成（scar 粘滞不丢）、appraisal 照常驱动 base；
- flag 门控：full 需 takeover；关闭态字节一致（步进演化照旧）；
- D2：set_relationship 端到端（spine→scar→Φ_eq warmth 上移）、持久化 roundtrip、
  越界拒绝、affect 关时快照无新增键。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sylanne_core import SylanneEngine
from sylanne_core.compute import affect_dynamics
from sylanne_core.compute.resonance_integration import ResonanceSpine
from sylanne_core.compute.scar_algebra import ScarredState
from sylanne_core.config import SylanneConfig, build_profile

_TRAITS: dict[str, float] = {
    "warmth_bias": 0.6,
    "perception_acuity": 0.7,
    "curiosity": 0.7,
    "expression_drive_trait": 0.6,
    "relational_gravity": 0.7,
    "sovereignty_guard": 0.8,
    "inner_order": 0.6,
}


def _full_state() -> ScarredState:
    st = ScarredState(n_dims=8, affect_enabled=True)
    st.set_affect_params(_TRAITS, takeover=True, full_takeover=True)
    st.base = affect_dynamics.from_unit_interval(affect_dynamics.equilibrium(_TRAITS, 0.5))
    st._e_last_wall_ts = 1000.0
    return st


class TestFullTakeover:
    def test_silence_tick_is_decay_only(self) -> None:
        # D1 的核心验收：吵架后隔夜，醒来那步不再被 MLP 拽走——残留 → Φ_eq。
        st = _full_state()
        st.apply_affect_takeover(-0.8, 0.9, 0.75, "生气")  # 吵架
        st.step([0.0] * 8, timestamp=1000.0 + 8 * 3600.0)  # 8h 后零事件 step
        eq_native = affect_dynamics.from_unit_interval(affect_dynamics.equilibrium(_TRAITS, 0.5))
        for i in range(8):
            assert abs(st.base[i] - eq_native[i]) < 0.02, i  # 像差消失（混血下是 +0.166）

    def test_h_priors_now_observable(self, monkeypatch) -> None:
        # 全权下 h 成为活杠杆：同样 2h 静默，h×0.25 应比 h×2 收敛得多。
        def residual(h_scale: float) -> float:
            monkeypatch.setattr(
                affect_dynamics,
                "_H_BASE_MIN",
                tuple(h * h_scale for h in (90.0, 30.0, 60.0, 45.0, 40.0, 50.0, 25.0, 120.0)),
            )
            st = _full_state()
            st.apply_affect_takeover(-0.8, 0.9, 0.75, "生气")
            st.step([0.0] * 8, timestamp=1000.0 + 2 * 3600.0)
            eq = affect_dynamics.from_unit_interval(affect_dynamics.equilibrium(_TRAITS, 0.5))
            return sum(abs(st.base[i] - eq[i]) for i in range(8))

        assert residual(0.25) < residual(2.0) - 1e-3  # 混血下两者差 <0.01

    def test_scars_still_form(self) -> None:
        st = _full_state()
        st.wound_threshold = 0.05
        st.step([1.0] * 8, timestamp=1100.0)  # 全权下伤痕形成不被旁路
        assert len(st.scars) > 0

    def test_appraisal_still_drives_base(self) -> None:
        st = _full_state()
        before = list(st.base)
        assert st.apply_affect_takeover(0.9, 0.7, 0.0, "撒娇") is True
        assert st.base != before

    def test_off_keeps_mlp_evolution(self) -> None:
        # full 关（仅 takeover）：主步演化照旧发生——零事件 step 也会移动 base。
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=True, full_takeover=False)
        st.base = affect_dynamics.from_unit_interval(affect_dynamics.equilibrium(_TRAITS, 0.5))
        before = list(st.base)
        st.step([0.0] * 8, timestamp=1000.0)
        assert st.base != before  # MLP 仍在演化

    def test_gamma_wound_and_feedback_mood_inert(self) -> None:
        # 红队 major（披露钉死）：非 assessor 创伤通道（Γ 耦合/feedback 向量）在全权下
        # 变哑——base 不动；但幅度过阈时伤痕照常形成（粘滞角色保留）。
        st = _full_state()
        base_before = list(st.base)
        st.step([0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], 0.0, heal=False)  # Γ 式小创伤
        assert st.base == base_before  # mood-inert（混血下 MLP 会动全维）
        st.wound_threshold = 0.3
        st.step([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 0.0, heal=False)  # 过阈
        assert st.base == base_before  # 仍不动 base
        assert len(st.scars) > 0  # 但伤痕形成了（粘滞不丢）

    def test_config_requires_takeover(self) -> None:
        with pytest.raises(ValueError):
            SylanneConfig(
                affect_dynamics_enabled=True, affect_takeover=False, affect_full_takeover=True
            )
        SylanneConfig(
            affect_dynamics_enabled=True, affect_takeover=True, affect_full_takeover=True
        )  # no raise


class TestRelationshipWiring:
    def _spine(self) -> ResonanceSpine:
        sp = ResonanceSpine(
            profile=build_profile("lite"), affect_enabled=True, affect_takeover=True
        )
        sp.apply_personality(_TRAITS)
        return sp

    def test_r_raises_warmth_equilibrium(self) -> None:
        sp = self._spine()
        sp.set_relationship(0.9)
        sp.process("你好", timestamp=1.0)  # personality dirty → 重应用 → R 注入
        scar = sp._engine.scar_state
        assert scar._relationship == 0.9
        eq_hi = affect_dynamics.equilibrium(_TRAITS, 0.9)
        eq_mid = affect_dynamics.equilibrium(_TRAITS, 0.5)
        assert eq_hi[0] > eq_mid[0]  # warmth 行随 R 上移（Φ_eq 语义）

    def test_r_persisted_roundtrip(self) -> None:
        sp = self._spine()
        sp.set_relationship(0.8)
        sp.process("你好", timestamp=1.0)
        d = sp.to_dict()
        assert d["affect_relationship"] == 0.8
        sp2 = ResonanceSpine(
            profile=build_profile("lite"), affect_enabled=True, affect_takeover=True
        )
        sp2.apply_personality(_TRAITS)
        sp2.from_dict(d)
        assert sp2._affect_relationship == 0.8
        assert sp2._engine.scar_state._relationship == 0.8  # restore mirror 注入

    def test_no_key_when_affect_off(self) -> None:
        sp = ResonanceSpine(profile=build_profile("lite"))  # affect off
        sp.apply_personality(_TRAITS)
        sp.process("你好", timestamp=1.0)
        assert "affect_relationship" not in sp.to_dict()  # 字节一致

    def test_computation_spine_restore_order(self) -> None:
        # 红队 fatal：ComputationSpine.from_dict 曾在 scar 镜像**之后**才恢复 R——
        # 恢复后 live scar_state._relationship 停在 0.5。钉死两个 spine 的恢复顺序。
        from sylanne_core.compute.computation_spine import ComputationSpine

        sp = ComputationSpine(profile=build_profile("lite"), affect_enabled=True)
        sp.apply_personality(_TRAITS)
        sp.set_relationship(0.8)
        d = sp.to_dict()
        sp2 = ComputationSpine(profile=build_profile("lite"), affect_enabled=True)
        sp2.apply_personality(_TRAITS)
        sp2.from_dict(d)
        assert sp2._affect_relationship == 0.8
        assert sp2.engine.scar_state._relationship == 0.8  # 镜像用的是恢复后的 R

    def test_repeated_set_relationship_on_bare_spine(self) -> None:
        # 红队 major：脏标机制在人格比较到达不动点后永久失效——set_relationship 现在
        # 直接注入 live scar 参数，连续调用必须逐次生效（不依赖 kernel 每 tick 重应用）。
        sp = self._spine()
        for r in (0.6, 0.7, 0.8):
            sp.set_relationship(r)
            sp.process("你好", timestamp=r * 10.0)
            assert sp._engine.scar_state._relationship == r, r

    @pytest.mark.asyncio
    async def test_engine_api_end_to_end(self, tmp_path: Path) -> None:
        engine = SylanneEngine(
            data_dir=tmp_path,
            llm=AsyncMock(return_value="ok"),
            config=SylanneConfig(assessor_enabled=False, affect_dynamics_enabled=True),
        )
        await engine.start()
        with pytest.raises(ValueError):
            await engine.set_relationship("s", 1.5)  # 越界拒绝
        await engine.set_relationship("s", 0.9)
        await engine.process("s", "你好")
        # v26: direct host access now requires an active session lease (LRU-safety
        # invariant added by the brain single-writer work); public APIs above take
        # their own lease, so only this internal probe needs to hold one.
        async with engine._session_lock("s"):
            host = await engine._get_or_create_host("s")
            assert host.kernel.computation._engine.scar_state._relationship == 0.9
        await engine.shutdown()
