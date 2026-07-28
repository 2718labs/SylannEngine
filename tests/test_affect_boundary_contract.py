"""边界契约执行测试 —— 数学红队 composition fatal / slow-channel major 的代码闸锚。

对照 docs/design/affect-dynamics-derivation.md v2（定理 2 前提、定理 3 A6、定理 4 双闸）。守护：
- ``ScarredState.from_dict`` 在复原边界把损坏快照的越界 base 夹回存储契约 [-1,1]
  （否则裸凸组合 decay 对越界输入不自愈，base=1.5 经衰减仍 1.49…，不变集失守）；
- ``_affect_decay`` 入口皮带：即使 base 被未来写入者污染，衰减输出仍有界；
- ``affect_dynamics._trait`` 末端夹 [0,1]（A6 边界强制）：越域 trait 不能把 g_tension
  推到 10、半衰期上界失守（k̲→0 = 定理 3"永不冻结"被打穿）。
- 合法输入下三闸恒等（不改任何在域行为）。
"""

from __future__ import annotations

import math

from sylanne_core.compute import affect_dynamics
from sylanne_core.compute.scar_algebra import ScarredState

_TRAITS: dict[str, float] = {"perception_acuity": 0.7, "warmth_bias": 0.6}


class TestRestoreClampsBase:
    def _snapshot_with_base(self, base: list[float]) -> dict:
        st = ScarredState(n_dims=8, affect_enabled=True)
        d = st.to_dict()
        d["base"] = base
        return d

    def test_corrupt_snapshot_clamped(self) -> None:
        d = self._snapshot_with_base([1.5, -2.0, 0.3, 99.0, -0.5, 0.0, 1.0, -1.0])
        restored = ScarredState.from_dict(d, affect_enabled=True)
        assert restored.base == [1.0, -1.0, 0.3, 1.0, -0.5, 0.0, 1.0, -1.0]

    def test_legal_snapshot_identity(self) -> None:
        legal = [-1.0, -0.7, -0.25, 0.0, 0.1, 0.5, 0.9, 1.0]
        restored = ScarredState.from_dict(self._snapshot_with_base(list(legal)))
        assert restored.base == legal  # 在域值逐位恒等，闸不改合法行为

    def test_corrupt_then_decay_stays_bounded(self) -> None:
        # 红队复现的完整攻击链：坏快照 -> 复原 -> takeover 衰减。闸后必须有界。
        d = self._snapshot_with_base([1.5] + [0.0] * 7)
        st = ScarredState.from_dict(d, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=True)
        st._e_last_wall_ts = 100.0
        st._affect_decay(150.0)
        assert all(-1.0 <= x <= 1.0 for x in st.base), st.base


class TestDecayEntryBelt:
    def test_poisoned_base_takeover_bounded(self) -> None:
        # from_dict 之外的假想污染源（未来 bug）：入口皮带独立兜底。
        # 注：公开 setter `base` 现在会校验并拒绝越界值（brain 单写门加的合理硬化），
        # 所以这里直写内部 `_base` 模拟“绕过 setter 的未来污染源”，仍验证 decay 兜底。
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=True)
        st._base = [1.5, -3.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        st._e_last_wall_ts = 100.0
        st._affect_decay(200.0)
        assert all(-1.0 <= x <= 1.0 for x in st.base), st.base

    def test_poisoned_shadow_bounded(self) -> None:
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=False)
        st._affect_shadow_base = [2.0] * 8
        st._e_last_wall_ts = 100.0
        st._affect_decay(200.0)
        assert st._affect_shadow_base is not None
        assert all(-1.0 <= x <= 1.0 for x in st._affect_shadow_base)


class TestTraitDomainEnforced:
    def test_out_of_domain_trait_clamped(self) -> None:
        # percept=5.0 越域：闸后与 percept=1.0 完全一致（g_tension 封在 2.0，非 10.0）。
        wild = affect_dynamics.half_lives({"perception_acuity": 5.0})
        capped = affect_dynamics.half_lives({"perception_acuity": 1.0})
        assert wild == capped

    def test_half_life_upper_bound_holds(self) -> None:
        # 定理 3 的 k̲>0 依赖 h 有上界：h ≤ h_base_max·g_max·S̄（percept/scarload 全极端）。
        h = affect_dynamics.half_lives({"perception_acuity": 5.0}, scarload=[100.0] * 8)
        h_bound = 120.0 * 2.0 * 3.0 * 60.0  # h_base_max(boundary=120min)·g_max·S̄，秒
        assert all(0.0 < x <= h_bound for x in h), h

    def test_gain_and_equilibrium_survive_wild_traits(self) -> None:
        wild = {"perception_acuity": 99.0, "expression_drive_trait": -7.0, "warmth_bias": 42.0}
        gain = affect_dynamics.gain_vector(wild)
        affect_dynamics.validate_gain(gain)  # 不抛：G ∈ (0,1] 仍守
        eq = affect_dynamics.equilibrium(wild, 0.5)
        assert all(0.15 <= x <= 0.85 for x in eq)
        assert all(math.isfinite(x) for x in gain + eq)

    def test_in_domain_traits_identity(self) -> None:
        # 在域 trait 逐位恒等：闸不改任何已标定行为。
        t = {"perception_acuity": 0.7, "expression_drive_trait": 0.3}
        assert affect_dynamics.half_lives(t) == affect_dynamics.half_lives(dict(t))
        g = affect_dynamics.gain_vector(t)
        assert abs(g[3] - (0.40 + 0.30 * 0.7)) < 1e-12  # 系数原样


class TestGeminiReviewHardening:
    """PR #26 gemini-code-assist review：NaN/越界/损坏快照的健壮性回归锚。"""

    def test_poignancy_magnitude_short_a_k_no_indexerror(self) -> None:
        # a_k 短于 N_DIMS 不得 IndexError（缺维补 0）；NaN 分量被 _finite 消毒。
        assert affect_dynamics.poignancy_magnitude([0.5, 0.5]) >= 0.0
        assert affect_dynamics.poignancy_magnitude([]) == 0.0
        assert math.isfinite(affect_dynamics.poignancy_magnitude([float("nan")] * 8))

    def test_slow_channel_nan_not_into_pending(self) -> None:
        from sylanne_core.compute.slow_channel import SlowChannel

        sc = SlowChannel(active=True)
        sc.observe([float("nan"), float("inf"), 0.5, -0.4] + [0.0] * 4)
        assert all(math.isfinite(v) for v in sc._pending.values()), sc._pending

    def test_from_dict_corrupt_learned_state_no_crash(self) -> None:
        # 损坏的 affect_gain/phi（非数值）不得让 from_dict 崩，回落未学习/中性。
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=True, plasticity=True)
        st.apply_affect_takeover(0.8, 0.7, 0.1, "撒娇")
        st.apply_affect_quality(1.0)
        d = st.to_dict()
        d["affect_gain"] = ["oops"] * 8
        d["affect_phi"] = [None] * 8
        rt = ScarredState.from_dict(d, affect_enabled=True)  # 不得抛
        assert rt._affect_gain is None
        assert rt._affect_phi == [0.0] * 8

    def test_from_dict_nan_learned_state_rejected(self) -> None:
        # PR #28 gemini：float("nan") 不抛 → 会绕过 try/except 静默钉进学习态。
        # isfinite 闸必须把 NaN/inf 也判成损坏 → gain 回落 None、phi 回落中性、q_ema 回落 0.5。
        st = ScarredState(n_dims=8, affect_enabled=True)
        st.set_affect_params(_TRAITS, takeover=True, plasticity=True)
        st.apply_affect_takeover(0.8, 0.7, 0.1, "撒娇")
        st.apply_affect_quality(1.0)
        d = st.to_dict()
        d["affect_gain"] = [float("nan")] * 8
        d["affect_phi"] = [float("inf")] * 8
        d["affect_q_ema"] = float("nan")
        rt = ScarredState.from_dict(d, affect_enabled=True)
        assert rt._affect_gain is None  # 非有限 gain 不得钉成 0.05
        assert rt._affect_phi == [0.0] * 8  # 非有限 phi 回落中性资格迹
        assert rt._affect_q_ema == 0.5  # 非有限 q_ema 回落先验

    def test_resolve_label_short_prev_key_no_indexerror(self) -> None:
        from sylanne_core.compute.affect_output_contract import HysteresisState, resolve_label

        bad_prev = HysteresisState(key=(1,), label="中性")  # 比 _KEY_DIMS 短
        label, state = resolve_label([0.5] * 8, bad_prev)
        assert isinstance(label, str) and len(state.key) == 2
