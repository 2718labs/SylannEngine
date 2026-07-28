"""Sylanne-Embodiment 计算核心层：伤痕代数（Scar Algebra）。

在 7 层计算栈中的位置：L3 VoidScar 层的"伤痕"部分。
职责：实现一种自修改算子代数——过去的操作会不可逆地改变未来操作的语义。
伤痕是不可逆的标记，它们调制系统处理未来输入的方式。

核心概念：
  - Scar（伤痕）：附着在某个维度上的不可逆标记，有 RAW→CLOSING→SCARRED→FADED 四阶段愈合
  - ScarredState（伤痕状态）：基向量 + 伤痕序列，通过 ⊳ 算子实现状态转移
  - modifier（调制因子）：伤痕对维度的累积放大/麻木效应
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from . import affect_dynamics, affect_projection
from .brain_errors import BrainOwnershipError
from .pel_core import N as _PEL_N
from .pel_core import PELCore

logger = logging.getLogger("sylanne_core")

# D-3/D-7: wound/feedback steps never advance the PEL latent ``mu``. When PEL is
# active they apply a cheap, bounded affine bias on ``base`` instead of the
# legacy MLP, so scar side-effects stay alive while ``mu`` evolves from the main
# step alone. Small gain keeps ``tanh(base + gain*modulated) in [-1, 1]``.
_PEL_AFFINE_GAIN: float = 0.3

# 更脑 v2 (must-fix #4): steady-window cross-dim precision spread below this and the
# divisive "attention" has gone dead (flat traffic re-saturates it). Mirrors the
# T-DIV acceptance tol so the production witness uses the same bar as CI.
_PEL_PRECISION_LIVE_TOL: float = 0.15


class HealingStage(IntEnum):
    """伤痕愈合阶段枚举。

    RAW(0) → CLOSING(1) → SCARRED(2) → FADED(3)
    每个阶段有不同的 alpha 调制因子和持续时间。
    """

    RAW = 0
    CLOSING = 1
    SCARRED = 2
    FADED = 3


# 各阶段的 alpha 调制因子：RAW 阶段放大最强（2.0），FADED 阶段衰减（0.7）
_STAGE_ALPHA = {
    HealingStage.RAW: 2.0,
    HealingStage.CLOSING: 1.5,
    HealingStage.SCARRED: 1.0,
    HealingStage.FADED: 0.7,
}

# 各阶段的默认持续时间（tick 数），FADED 阶段无限期
_STAGE_DURATION = {
    HealingStage.RAW: 10,
    HealingStage.CLOSING: 40,
    HealingStage.SCARRED: 150,
}


@dataclass(slots=True)
class Scar:
    """单个伤痕对象。

    附着在特定维度上，有四阶段愈合过程。
    alpha 属性决定该伤痕对所在维度的调制强度：
      RAW=2.0（新伤放大）, CLOSING=1.5, SCARRED=1.0（中性）, FADED=0.7（衰减）
    """

    dimension: int
    timestamp: float
    stage: HealingStage = HealingStage.RAW
    ticks_in_stage: int = 0

    @property
    def alpha(self) -> float:
        return _STAGE_ALPHA[self.stage]

    def heal_tick(self) -> bool:
        """推进愈合一个 tick。如果阶段发生变化返回 True。"""
        if self.stage == HealingStage.FADED:
            return False
        self.ticks_in_stage += 1
        threshold = _STAGE_DURATION.get(self.stage)
        if threshold is not None and self.ticks_in_stage >= threshold:
            self.stage = HealingStage(self.stage + 1)
            self.ticks_in_stage = 0
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "timestamp": self.timestamp,
            "stage": self.stage.name,
            "ticks_in_stage": self.ticks_in_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scar:
        return cls(
            dimension=data["dimension"],
            timestamp=data["timestamp"],
            stage=HealingStage[data["stage"]],
            ticks_in_stage=data.get("ticks_in_stage", 0),
        )


class ScarredState:
    """伤痕代数核心状态：基向量 + 不可逆伤痕序列。

    状态转移通过 ⊳ 算子（step 方法）实现：
      1. 伤痕调制输入（modulate）：每个维度的输入乘以该维度的累积 modifier
      2. 基向量演化（_evolve_base）：2 层 MLP 将 [当前状态; 调制后输入] 映射为新状态
      3. 伤痕形成（conditional）：调制后输入超过阈值的维度产生新伤痕
      4. 愈合（heal）：已有伤痕按阶段推进

    与其他组件的关系：
      - 被 VoidScarEngine 调用，接收 HDC 压缩后的 8 维输入
      - 通过 Φ 耦合影响 VoidSpace 的检测灵敏度
      - observe() 输出 8 维情感状态给下游层
    """

    __slots__ = (
        "_base",
        "_brain_capability",
        "scars",
        "n_dims",
        "wound_threshold",
        "_tick",
        "_t_raw",
        "_t_closing",
        "_t_scarred",
        "_mlp_w1",
        "_mlp_w2",
        "_mlp_hidden_dim",
        "_mlp_passes",
        "_neuroticism",
        # Session scar cap (sovereignty immune system)
        "_session_scar_count",
        "_session_scar_cap",
        # Circuit breaker (protective dissociation)
        "_circuit_breaker_active",
        "_circuit_breaker_remaining",
        "_recent_scar_ticks",
        # Time-aware healing
        "_last_step_time",
        # 每维度 modifier 缓存（避免 observe/modulate 重复遍历伤痕列表）
        "_modifier_cache",
        "_modifier_cache_valid",
        # PEL-Core (v2.5): optional predictive-coding latent core, gated by config.
        # ``_pel is None`` => legacy MLP path runs and behaviour is byte-identical.
        "_pel",
        "_pel_enabled",
        # v2.6.0 affect-dynamics E-law shadow (Gate A: computed + logged, NEVER
        # written into ``base``; ``observe()`` never reads it; discarded at T3
        # promotion). ``_affect_enabled`` off => byte-identical legacy.
        "_affect_enabled",
        "_affect_traits",
        "_relationship",
        "_affect_shadow_base",
        "_e_last_wall_ts",
        "_last_affect_shadow",
        # v2.6.0 T3 takeover: when True the E-law is AUTHORITATIVE (writes base):
        # decay-to-Phi_eq at top of step + saturating appraisal replaces hand-rules.
        "_affect_takeover",
        # v26 A.2 delta-rule gain plasticity (Lemma 6 projection contract). Learned
        # gain state survives restarts; None until first takeover use (lazy init
        # from gain_vector(traits), then decoupled from T).
        "_affect_plasticity",
        "_affect_gain",
        "_affect_phi",
        "_affect_q_ema",
        # v26 D1(b) full takeover: main-step MLP/PEL base evolution bypassed on the
        # 8-dim core — base evolves only via decay + appraisal + wound scars.
        "_affect_full",
        # v2.6.0 T-Persist: monotonic version of ``base`` (dormant — bumped on every
        # base mutation, never gates logic; reserved for future cross-writer detect).
        "_e_ver",
    )

    def __init__(
        self,
        n_dims: int = 8,
        wound_threshold: float = 0.6,
        mlp_passes: int = 1,
        *,
        pel_enabled: bool = False,
        affect_enabled: bool = False,
        brain_capability: object | None = None,
        authoritative_base: Sequence[float] | None = None,
    ):
        self.n_dims = n_dims
        self.wound_threshold = wound_threshold
        self._brain_capability = brain_capability
        self._base: list[float] | tuple[float, ...] = (
            (0.0,) * n_dims if brain_capability is not None else [0.0] * n_dims
        )
        if authoritative_base is not None:
            self._replace_base(authoritative_base, brain_capability)
        self.scars: list[Scar] = []
        self._tick = 0
        self._neuroticism: float = 0.5
        # Configurable healing rates (defaults match original _STAGE_DURATION)
        self._t_raw: int = 10
        self._t_closing: int = 40
        self._t_scarred: int = 150
        # MLP hidden dim scales with n_dims for higher-dim modes
        self._mlp_hidden_dim: int = max(12, n_dims + 4)
        self._mlp_passes: int = max(1, mlp_passes)
        self._mlp_w1: list[list[float]] | None = None
        self._mlp_w2: list[list[float]] | None = None
        # Session scar cap (sovereignty immune system)
        self._session_scar_count: int = 0
        self._session_scar_cap: int = 3
        # Circuit breaker (protective dissociation)
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_remaining: int = 0
        self._recent_scar_ticks: list[int] = []
        # Time-aware healing
        self._last_step_time: float = 0.0
        # 每维度 modifier 缓存：避免 observe() 和 modulate() 每次都遍历全部伤痕
        # 任何伤痕变动（新增/愈合/移除）都会使缓存失效
        self._modifier_cache: dict[int, float] = {}
        self._modifier_cache_valid: bool = False
        # PEL-Core: enabled flag + (lazily set) latent core. The core is only
        # built by ``set_pel_priors`` and only for the frozen 8-dim emotion space.
        self._pel_enabled: bool = pel_enabled
        self._pel: PELCore | None = None
        # v2.6.0 affect E-law shadow. ``_affect_enabled`` is construction-time (from
        # config, mirrors ``_pel_enabled``); traits/relationship arrive later via
        # ``set_affect_params`` (mirrors ``set_pel_priors``). Shadow buffer + affect
        # wall-clock + last diagnostic snapshot are all diagnostic-only.
        self._affect_enabled: bool = affect_enabled
        self._affect_traits: dict[str, float] = {}
        self._relationship: float = 0.5
        self._affect_shadow_base: list[float] | None = None
        self._e_last_wall_ts: float = 0.0
        self._last_affect_shadow: dict[str, Any] | None = None
        self._affect_takeover: bool = False
        self._affect_plasticity: bool = False
        self._affect_full: bool = False
        self._affect_gain: list[float] | None = None
        self._affect_phi: list[float] = [0.0] * n_dims
        self._affect_q_ema: float = 0.5
        self._e_ver: int = 0

    @property
    def base(self) -> list[float] | tuple[float, ...]:
        """Current base vector, immutable while the authoritative brain owns it."""
        return self._base

    @base.setter
    def base(self, candidate: Sequence[float]) -> None:
        if self._brain_capability is not None:
            raise BrainOwnershipError("authoritative brain base requires its matching capability")
        self._replace_legacy_base(candidate)

    def _replace_base(self, candidate: Sequence[float], capability: object | None) -> None:
        """Replace the complete base vector after proving writer ownership."""
        if capability is not self._brain_capability:
            raise BrainOwnershipError("stale or foreign brain base capability")
        try:
            values = tuple(float(value) for value in candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("base must be a finite numeric sequence") from exc
        if len(values) != self.n_dims:
            raise ValueError(f"base must contain exactly {self.n_dims} dimensions")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("base dimensions must be finite values in [-1, 1]")
        self._base = values if self._brain_capability is not None else list(values)

    def _replace_legacy_base(self, candidate: Sequence[float]) -> None:
        """Route legacy whole-vector writes through the single replacement gate."""
        if self._brain_capability is not None:
            raise BrainOwnershipError(
                "legacy base evolution cannot write authoritative brain state"
            )
        self._replace_base(candidate, None)

    def set_healing_rates(
        self, t_raw: int, t_closing: int, t_scarred: int, neuroticism: float = 0.5
    ) -> None:
        """设置各愈合阶段的持续时间（由人格参数驱动）。

        高神经质 → 愈合更慢（各阶段持续时间更长）。

        Args:
            t_raw: RAW 阶段持续 tick 数
            t_closing: CLOSING 阶段持续 tick 数
            t_scarred: SCARRED 阶段持续 tick 数
            neuroticism: 人格神经质值，影响 modifier 上限
        """
        self._t_raw = max(1, int(t_raw))
        self._t_closing = max(1, int(t_closing))
        self._t_scarred = max(1, int(t_scarred))
        self._neuroticism = float(neuroticism)
        # 神经质值影响 modifier 饱和上限，需要使缓存失效
        self._invalidate_modifier_cache()

    def pel_active(self) -> bool:
        """True iff the PEL-Core latent path drives the main step (vs legacy MLP)."""
        return self._pel is not None

    def pel_diagnostics(self) -> dict[str, Any] | None:
        """Lightweight PEL signal surface (D-1/D-10), or ``None`` when PEL is off.

        Non-semantic observability only: the latest free energy ``F``, the
        per-dim precisions ``pi_obs``/``pi_top`` and the mean absolute bottom-up /
        top-down errors. Pure read — never gates anything (downstream call-
        skipping is explicitly out of scope, design D-10).
        """
        if self._pel is None:
            return None
        st = self._pel.state
        n = len(self._pel.last_e0)
        pi_obs = list(st.pi_obs)
        pi_top = list(st.pi_top)
        # 更脑 v2 production liveness witness (must-fix #4): divisive precision's
        # liveness is DATA-CONTINGENT — if real traffic ever drove near-equal errors
        # the cross-dim spread would collapse and "attention" would silently go dead
        # again while every CORPUS-fed CI gate stayed green. Surface the cross-dim
        # precision spread + the BCM*precision product spread so a downstream monitor
        # can window them on REAL traffic and alert when steady-state spread falls
        # below the same tol CI uses. This makes the DEAD->LIVE claim falsifiable
        # post-deploy instead of only on the curated fixture.
        prod = [pi_obs[i] * self._pel.last_m[i] for i in range(n)] if n else []
        pi_obs_pstd = statistics.pstdev(pi_obs) if len(pi_obs) > 1 else 0.0
        pi_top_pstd = statistics.pstdev(pi_top) if len(pi_top) > 1 else 0.0
        prod_spread = statistics.pvariance(prod) if len(prod) > 1 else 0.0
        # 更脑 v2 (M3) identity witness: distance of the live setpoint pi from its
        # frozen trait prior pi0. Symmetric with the M1 witness — surfaces anchor
        # erosion on REAL traffic (the anchor retains ~80% of pi0 in theory; if pi
        # ran away to <z> on real input this drift would climb toward ||pi0||~O(1)
        # and a downstream monitor could alert), not just on the synthetic fixture.
        pi_anchor_drift = math.sqrt(sum((st.pi[i] - st.pi0[i]) ** 2 for i in range(len(st.pi))))
        return {
            "free_energy": st.free_energy,
            "pi_obs": pi_obs,
            "pi_top": pi_top,
            "pi_obs_pstd": pi_obs_pstd,
            "pi_top_pstd": pi_top_pstd,
            "prod_spread": prod_spread,
            "precision_live": pi_obs_pstd > _PEL_PRECISION_LIVE_TOL,
            "pi_anchor_drift": pi_anchor_drift,
            "mean_abs_e0": sum(abs(v) for v in self._pel.last_e0) / n if n else 0.0,
            "mean_abs_e1": sum(abs(v) for v in self._pel.last_e1) / n if n else 0.0,
        }

    def set_pel_priors(self, personality: dict[str, float]) -> None:
        """Initialise the PEL-Core latent micro-circuit from Big-Five personality.

        Sets the attractor prior ``pi``, the generative matrix ``W_gen``, the
        precisions and ``mu0`` from the personality traits (techspec §4.3). This
        is a no-op unless PEL is enabled *and* this is the frozen 8-dim emotion
        core — PEL targets the 8 canonical emotion dimensions only, so pro/max
        (16/128-dim) cores keep running the legacy MLP. Idempotent; re-applying
        personality (per-relationship overlays, tier switches) rebuilds the core.
        """
        if not self._pel_enabled or self.n_dims != _PEL_N:
            return
        self._pel = PELCore.from_personality(personality, base=list(self.base))

    # ------------------------------------------------------------------
    # v2.6.0 affect-dynamics E-law shadow (Gate A: shadow-only, never touches base)
    # ------------------------------------------------------------------

    def set_affect_params(
        self,
        traits: dict[str, float],
        relationship: float = 0.5,
        *,
        takeover: bool = False,
        plasticity: bool = False,
        full_takeover: bool = False,
    ) -> None:
        """注入 E 律人格 traits + 关系相位 + 夺权/可塑性开关（由 apply_personality 调用）。

        镜像 ``set_pel_priors`` 的注入位，随人格覆盖幂等重设。traits/relationship/开关
        **不落盘**——复原后由 apply_personality 重新注入（PEL must-fix #3 同型）。relationship
        缺省 0.5（canonical 尚无关系相位标量接线；真实 R 是后续跟进项）。``takeover`` 由 config
        经 spine 传入：True ⇒ T3 E 律夺权写 base；False ⇒ T1 影子（默认）。``plasticity``
        （A.2）开启 delta-rule 增益学习；**不重置已学的 ``_affect_gain``**——人格重复注入
        （关系覆盖/档位切换）不得清洗学习态。
        """
        self._affect_traits = dict(traits) if traits else {}
        r = float(relationship)
        self._relationship = r if math.isfinite(r) and 0.0 <= r <= 1.0 else 0.5
        self._affect_takeover = bool(takeover)
        self._affect_plasticity = bool(plasticity)
        self._affect_full = bool(full_takeover)

    def _affect_active(self) -> bool:
        """影子仅对 8 维情感核生效（affect_dynamics 全按 N_DIMS=8 立式，pro/max 核跳过）。"""
        return self._affect_enabled and self.n_dims == affect_dynamics.N_DIMS

    def _record_affect_shadow(self, source: str, matched: str | None = None) -> dict[str, Any]:
        """构建影子诊断快照 + 落 debug 日志（散度 = 影子 E 与真实 base 的 L2 距离）。"""
        shadow = (
            self._affect_shadow_base if self._affect_shadow_base is not None else list(self.base)
        )
        divergence = math.sqrt(sum((shadow[i] - self.base[i]) ** 2 for i in range(self.n_dims)))
        diag: dict[str, Any] = {
            "source": source,
            "intent_class": matched,
            "divergence_l2": divergence,
            "shadow": list(shadow),
        }
        self._last_affect_shadow = diag
        logger.debug("affect-shadow[%s] div=%.4f intent=%s", source, divergence, matched)
        return diag

    def _affect_decay(self, timestamp: float) -> None:
        """E 律墙钟惰性衰减，在 ``step()`` **顶部**（事件演化之前）应用。

        - T1 影子（``_affect_takeover`` off）：衰减只动 ``_affect_shadow_base``，绝不碰 ``base``。
        - T3 夺权（``_affect_takeover`` on）：衰减动**权威 base**——settle 先于事件写回（设计 §9），
          杜绝"事件后衰减擦掉刚算出的回复"的双衰减禁忌（e-core #2 BLOCKER）。

        用 affect 层自有墙钟 ``_e_last_wall_ts``（不复用被 feedback() 清零的 ``_last_step_time``），
        仅在真实 timestamp>0 且有前次基准时推进；懒初始化影子 = base 快照。decay 仿射等变，故
        base 留原生 (-1,1) 帧、只把 Φ_eq 折回 native（Phase 0 已证等价）。异常自吞、绝不外逃主回合。
        """
        if not self._affect_active() or not (timestamp > 0.0):
            return
        try:
            # PEL owns base evolution (its read-out overwrites base each tick), so the
            # E-law takeover-to-base is inert under PEL — fall back to shadow so decay
            # is never silently discarded (red-team #1; config also rejects the combo).
            takeover = self._affect_takeover and not self.pel_active()
            # 入口夹回存储契约 [-1,1]：decay 是裸凸组合（定理 2 前提 u₀ 在界内），
            # 对越界输入不自愈——from_dict 已在复原边界执行契约，此处是对未来
            # 其它写入者的皮带（数学红队 composition fatal 的双闸修复）。
            if takeover:
                cur = [min(1.0, max(-1.0, x)) for x in self.base]
            else:
                if self._affect_shadow_base is None:
                    self._affect_shadow_base = list(self.base)
                cur = [min(1.0, max(-1.0, x)) for x in self._affect_shadow_base]
            prev = self._e_last_wall_ts
            ts = float(timestamp)
            if not (prev > 0.0):
                self._e_last_wall_ts = ts  # 首次播种时钟；无衰减可施
                return
            dt = ts - prev
            if not (dt > 0.0):
                return  # 非单调/零间隔：时钟留在 prev，不回拨
            eq_native = affect_dynamics.from_unit_interval(
                affect_dynamics.equilibrium(self._affect_traits, self._relationship)
            )
            scarload = [self.scar_density(d) for d in range(self.n_dims)]
            h_secs = affect_dynamics.half_lives(self._affect_traits, scarload)
            decayed = affect_dynamics.decay(cur, eq_native, h_secs, dt)
            # 原子提交：状态与时钟同批落地（AD5——C 轨红队修订：此前时钟在易错计算
            # **之前**推进，异常时时钟白走、该区间的衰减被永久静默丢弃；现在异常
            # ⇒ 时钟留在 prev，恢复后的下一次成功衰减补齐全部真实间隔）。
            if takeover:
                self._replace_legacy_base(decayed)  # T3 legacy takeover path
                self._e_ver += 1
            else:
                self._affect_shadow_base = decayed
            self._e_last_wall_ts = ts
            self._record_affect_shadow("decay")
        except Exception:  # pragma: no cover - fail-closed, diagnostic path only
            logger.debug("affect decay skipped (exception)", exc_info=True)

    def apply_affect_takeover(
        self, valence: float, arousal: float, wound_risk: float, intent: str | None
    ) -> bool:
        """T3 夺权：快通道 appraisal 直接写**权威 base**（替代 assessor 手写意图规则）。

        返回 True ⇒ 已接管本回合的语义快更新（调用方须跳过遗留手写规则）。返回 False ⇒ 未夺权
        （未启用/非 8 维/或 E 律异常 fail-closed）——调用方回落遗留手写规则（assessor #2）。投影 →
        gain → 饱和更新，折进 [0,1] 折回 native（saturating 非仿射等变）。
        """
        if not (self._affect_active() and self._affect_takeover) or self.pel_active():
            return False
        try:
            a_k, matched = affect_projection.project_appraisal(valence, arousal, wound_risk, intent)
            gain = self._effective_gain()
            affect_dynamics.validate_gain(gain)  # 越界抛 → 下方兜底回落手写规则
            unit = affect_dynamics.to_unit_interval(list(self.base))
            updated = affect_dynamics.saturating_update(unit, a_k, gain)
            self._replace_legacy_base(affect_dynamics.from_unit_interval(updated))
            self._e_ver += 1
            # A.2：可塑性开启时更新资格迹——只有参与本轮情绪反应的维度在下一次
            # quality 反馈到达时领赏罚（注 6.2 信用分配）。
            if self._affect_plasticity:
                self._affect_phi = affect_dynamics.eligibility_update(self._affect_phi, a_k)
            self._record_affect_shadow("takeover", matched=matched)
            return True
        except Exception:
            logger.debug("affect takeover failed; falling back to hand-rules", exc_info=True)
            return False

    def _effective_gain(self) -> list[float]:
        """当前生效增益：可塑性开 ⇒ 学习态 G（懒初始化自 gain_vector(traits) 后与 T 解耦）；
        关 ⇒ 每次由人格现算（遗留语义，人格漂移会即时移动 G）。"""
        if self._affect_plasticity:
            if self._affect_gain is None:
                self._affect_gain = affect_dynamics.gain_vector(self._affect_traits)
            return self._affect_gain
        return affect_dynamics.gain_vector(self._affect_traits)

    def apply_affect_quality(self, quality: float) -> bool:
        """A.2 delta-rule 增益学习步（quality 为上一轮回复质量的滞后反馈 ∈ [0,1]）。

        门：affect 活 ∧ takeover ∧ plasticity ∧ 非 PEL。δ = clip(q − q̂)，赏罚经资格迹
        分配到近期活跃维；投影 Π_{[ε,1]} 无条件执行（引理 6——学习信号再错也破不了
        定理 1–4）。基线 q̂ 在算完 δ 后推进。返回 True ⇒ 本步确实学习了。fail-closed。
        """
        if not (
            self._affect_active()
            and self._affect_takeover
            and self._affect_plasticity
            and not self.pel_active()
        ):
            return False
        try:
            if self._affect_gain is None:
                self._affect_gain = affect_dynamics.gain_vector(self._affect_traits)
            self._affect_gain = affect_dynamics.plasticity_step(
                self._affect_gain, quality, self._affect_q_ema, self._affect_phi
            )
            self._affect_q_ema = affect_dynamics.quality_baseline_update(
                self._affect_q_ema, quality
            )
            logger.debug("affect-plasticity q=%.3f q_ema=%.3f", float(quality), self._affect_q_ema)
            return True
        except Exception:  # pragma: no cover - fail-closed, learning must never crash a turn
            logger.debug("affect plasticity step skipped (exception)", exc_info=True)
            return False

    def apply_affect_appraisal_shadow(
        self, valence: float, arousal: float, wound_risk: float, intent: str | None
    ) -> dict[str, Any] | None:
        """快通道 appraisal 对影子 E 的饱和更新（Gate A：只动 ``_affect_shadow_base``，绝不碰 ``base``）。

        由两个 assessor 写入点在既有手写规则之后调用。投影 → gain_vector(traits) → 饱和更新，
        全程折进 [0,1] 折回 native（saturating_update 非仿射等变，必须整体折进折出）。返回诊断快照
        （命中意图类、影子-base 散度）供落日志；未启用/非 8 维返回 None。调用方仍须 try/except
        兜底（本方法内也自吞，双保险不外逃主回合）。
        """
        if not self._affect_active():
            return None
        try:
            if self._affect_shadow_base is None:
                self._affect_shadow_base = list(self.base)
            a_k, matched = affect_projection.project_appraisal(valence, arousal, wound_risk, intent)
            gain = affect_dynamics.gain_vector(self._affect_traits)
            affect_dynamics.validate_gain(gain)  # fail-closed：越界抛→本地兜底落日志
            unit = affect_dynamics.to_unit_interval(self._affect_shadow_base)
            updated = affect_dynamics.saturating_update(unit, a_k, gain)
            self._affect_shadow_base = affect_dynamics.from_unit_interval(updated)
            return self._record_affect_shadow("appraisal", matched=matched)
        except Exception:  # pragma: no cover - fail-closed, diagnostic path only
            logger.debug("affect-shadow appraisal skipped (exception)", exc_info=True)
            return None

    def healing_duration(
        self,
        stage: HealingStage,
        dim: int | None = None,
        _dim_counts: dict[int, int] | None = None,
    ) -> int:
        """获取某阶段的愈合持续时间，可选按维度调整。

        如果某维度的伤痕数 > 3，愈合速度降低 50%（反复受伤的地方更难愈合）。
        """
        base_duration = {
            HealingStage.RAW: self._t_raw,
            HealingStage.CLOSING: self._t_closing,
            HealingStage.SCARRED: self._t_scarred,
        }.get(stage, 0)
        if dim is not None:
            count = (_dim_counts or {}).get(dim) if _dim_counts else None
            if count is None:
                count = self.scar_count(dim)
            if count > 3:
                base_duration = int(base_duration * 1.5)
        return base_duration

    def scar_count(self, dim: int) -> int:
        """Count total scars on a given dimension."""
        return sum(1 for s in self.scars if s.dimension == dim)

    def _init_mlp_weights(self, seed: int = 42) -> None:
        """从确定性种子初始化 MLP 权重，并应用谱归一化。"""
        import random

        rng = random.Random(seed)
        input_dim = self.n_dims * 2  # [x; e_tilde] concatenated
        hidden_dim = self._mlp_hidden_dim

        # Layer 1: hidden_dim x input_dim
        self._mlp_w1 = [[rng.gauss(0, 0.5) for _ in range(input_dim)] for _ in range(hidden_dim)]
        # Layer 2: n_dims x hidden_dim
        self._mlp_w2 = [[rng.gauss(0, 0.5) for _ in range(hidden_dim)] for _ in range(self.n_dims)]
        # Apply spectral normalization to both weight matrices
        self._mlp_w1 = self._spectral_normalize(self._mlp_w1, max_sigma=0.7)
        self._mlp_w2 = self._spectral_normalize(self._mlp_w2, max_sigma=0.7)

    def _spectral_normalize(
        self, W: list[list[float]], max_sigma: float = 0.7
    ) -> list[list[float]]:
        """谱归一化：通过幂迭代估计最大奇异值，超过 max_sigma 时缩放矩阵。

        确保 ||W||_2 <= max_sigma，这是状态演化收敛的关键保证。
        10 次幂迭代足以收敛到合理精度。
        """
        rows = len(W)
        cols = len(W[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return W

        # Power iteration (10 iterations is sufficient for convergence)
        # Initialize u as unit vector
        u = [1.0 / math.sqrt(rows)] * rows
        v = [0.0] * cols

        for _ in range(10):
            # v = W^T u / ||W^T u||
            for j in range(cols):
                v[j] = sum(W[i][j] * u[i] for i in range(rows))
            v_norm = math.sqrt(sum(x * x for x in v)) + 1e-12
            v = [x / v_norm for x in v]

            # u = W v / ||W v||
            for i in range(rows):
                u[i] = sum(W[i][j] * v[j] for j in range(cols))
            u_norm = math.sqrt(sum(x * x for x in u)) + 1e-12
            u = [x / u_norm for x in u]

        # Estimate sigma = u^T W v
        sigma = 0.0
        for i in range(rows):
            sigma += u[i] * sum(W[i][j] * v[j] for j in range(cols))

        # Scale if needed
        if sigma > max_sigma:
            scale = max_sigma / sigma
            return [[W[i][j] * scale for j in range(cols)] for i in range(rows)]
        return W

    def _evolve_base(self, x: list[float], e_tilde: list[float]) -> list[float]:
        """通过 2 层 MLP 演化基向量（带谱归一化保证收敛）。

        Layer 1: hidden = tanh(W1 * [x; e_tilde])
        Layer 2: output = tanh(W2 * hidden)

        收敛保证：||W1||_2 * ||W2||_2 < 0.7 * 0.7 = 0.49 < 1
        这确保了状态演化是收缩映射，不会发散。
        """
        if self._mlp_w1 is None or self._mlp_w2 is None:
            self._init_mlp_weights()
        assert self._mlp_w1 is not None and self._mlp_w2 is not None

        # Concatenate input: [x; e_tilde]
        inp = list(x) + list(e_tilde)
        hidden_dim = len(self._mlp_w1)
        out_dim = len(self._mlp_w2)

        # Layer 1: hidden = tanh(W1 * inp)
        hidden = [0.0] * hidden_dim
        for i in range(hidden_dim):
            val = sum(self._mlp_w1[i][j] * inp[j] for j in range(len(inp)))
            hidden[i] = math.tanh(val)

        # Layer 2: output = tanh(W2 * hidden)
        output = [0.0] * out_dim
        for i in range(out_dim):
            val = sum(self._mlp_w2[i][j] * hidden[j] for j in range(hidden_dim))
            output[i] = math.tanh(val)

        return output

    def _invalidate_modifier_cache(self) -> None:
        """使 modifier 缓存失效（伤痕新增/愈合/移除时调用）。"""
        self._modifier_cache_valid = False

    def _ensure_modifier_cache(self) -> None:
        """按需重建全维度 modifier 缓存。

        一次遍历伤痕列表，计算所有维度的 product，再统一做饱和压缩。
        复杂度从 O(n_dims * num_scars) 降为 O(num_scars + n_dims)。
        """
        if self._modifier_cache_valid:
            return
        # 一次遍历收集每维度的 alpha 乘积
        products = [1.0] * self.n_dims
        for scar in self.scars:
            products[scar.dimension] *= scar.alpha
        # 对每个维度做饱和压缩
        max_mod = 2.0 + self._neuroticism * 3.0
        cache = {}
        for d in range(self.n_dims):
            p = products[d]
            if p <= 1.0:
                cache[d] = max(0.05, p)
            else:
                cache[d] = 1.0 + (max_mod - 1.0) * (1.0 - 1.0 / (p + 1e-10))
        self._modifier_cache = cache
        self._modifier_cache_valid = True

    def modifier(self, dim: int) -> float:
        """计算某维度的累积伤痕调制因子（带缓存）。

        使用对数压缩 + 人格驱动上限，防止多个伤痕的 alpha 乘积无限增长。
        公式：当 product > 1 时，modifier = 1 + (max_mod - 1) * (1 - 1/product)
        这是一个渐近线为 max_mod 的饱和函数。

        缓存策略：首次调用时一次性计算全部维度并缓存，后续直接查表。
        伤痕变动（wound/heal/remove）时缓存自动失效。

        Returns:
            调制因子，范围 [0.05, max_mod]。< 0.5 表示"麻木"，> 1.0 表示"敏感化"
        """
        self._ensure_modifier_cache()
        return self._modifier_cache.get(dim, 1.0)

    def modulate(self, event: list[float]) -> list[float]:
        """对输入事件应用伤痕调制（⊳ 算子的第 1 步）。

        每个维度的输入值乘以该维度的 modifier：
          - modifier > 1：该维度被"敏感化"，微小输入也会被放大
          - modifier < 1：该维度被"麻木"，需要更大输入才能产生效果
        """
        result = []
        for d in range(self.n_dims):
            e_d = event[d] if d < len(event) else 0.0
            result.append(e_d * self.modifier(d))
        return result

    def step(
        self,
        event: list[float],
        timestamp: float = 0.0,
        *,
        heal: bool = True,
        pel_ctx: tuple[list[float], float, list[float] | None, float] | None = None,
    ) -> dict[str, Any]:
        """应用 ⊳ 算子：完整状态转移。

        四步流程：
          1. 伤痕调制输入
          2. 基向量演化（PEL 潜核 或 遗留 MLP）
          3. 条件性伤痕形成（受会话上限和断路器保护）
          4. 已有伤痕愈合推进

        Args:
            event: 8 维输入事件向量
            timestamp: 事件时间戳（用于时间感知愈合）
            heal: 是否执行愈合步骤（Γ 耦合创伤事件设为 False）
            pel_ctx: 可选的 PEL 主步上下文 ``(x_t, surprise, a_vec, confidence)``。仅主 step 传入。
                当 PEL 激活且 ``pel_ctx`` 在场 ⇒ 潜核推进 ``mu`` 并写 ``base``；
                PEL 激活但 ``pel_ctx is None``（wound/feedback 步）⇒ 走廉价 affine
                bias（D-3/D-7，不推进 ``mu``）；PEL 未激活 ⇒ 遗留 MLP（字节一致）。

        Returns:
            诊断字典，包含调制后输入、新伤痕、愈合维度等信息
        """
        # v2.6.0: E-law wall-clock decay at the TOP of step (before event evolution).
        # Shadow-only under Gate A (never touches base); writes authoritative base
        # under T3 takeover. No-op unless affect_enabled & 8-dim & real timestamp.
        brain_owned = self._brain_capability is not None
        if not brain_owned:
            self._affect_decay(timestamp)

        if heal:
            self._tick += 1

        # --- Circuit breaker: protective dissociation ---
        if self._circuit_breaker_active:
            self._circuit_breaker_remaining -= 1
            if self._circuit_breaker_remaining <= 0:
                self._circuit_breaker_active = False
            effective_threshold = 0.95
        else:
            effective_threshold = self.wound_threshold

        # Step 1: Scar-modulated input
        modulated = self.modulate(event)

        # Step 2: Base state evolution.
        # v26 D1(b) FULL takeover: bypass the legacy MLP/PEL main-step evolution on
        # the 8-dim core. base then evolves ONLY via the top-of-step E-law decay,
        # the assessor appraisal (saturating update) and wound-terms — the
        # observable resting mood becomes Phi_eq and h priors become live levers
        # (calibration memo D1: the MLP attractor image previously dominated every
        # observation point). Scar formation/healing below are untouched (they
        # read `modulated`, not base). Gated: full ∧ takeover ∧ affect ∧ 8-dim ∧ 非PEL.
        skip_evolution = brain_owned or (
            self._affect_full
            and self._affect_takeover
            and self._affect_active()
            and not self.pel_active()
        )
        if skip_evolution:
            pass
        elif self._pel is not None:
            if pel_ctx is not None:
                # Main step: PEL latent free-energy descent + read-out -> base.
                # PEL's K is internal and fixed; it ignores ``_mlp_passes`` (G3).
                # v2.5 (B): a_vec/confidence carry the assessor in as a precision-
                # weighted semantic prior (inert when absent / confidence 0).
                x_t, surprise, a_vec, confidence = pel_ctx
                z, _free_energy = self._pel.step(x_t, surprise, a_vec, confidence)
                self._replace_legacy_base(z)
            else:
                # wound/feedback step: cheap bounded affine bias on base (D-3/D-7).
                self._replace_legacy_base(
                    [
                        math.tanh(self.base[d] + _PEL_AFFINE_GAIN * modulated[d])
                        for d in range(self.n_dims)
                    ]
                )
        else:
            # Legacy path: 2-layer MLP with spectral normalization.
            # Multi-pass refinement: pro/max modes run multiple passes.
            for _pass in range(self._mlp_passes):
                self._replace_legacy_base(self._evolve_base(list(self.base), modulated))

        # v2.6.0 T-Persist: bump the dormant base version on every mutation. Gated on
        # _affect_enabled so the counter truly never moves off-flag (red-team #3-minor);
        # skipped when full takeover bypassed the evolution (no mutation happened here).
        if self._affect_enabled and not skip_evolution:
            self._e_ver += 1

        # Step 3: Scar formation (conditional, with session cap)
        existing_count = len(self.scars)
        new_scars = []
        for d in range(self.n_dims):
            if abs(modulated[d]) > effective_threshold:
                # Session scar cap check
                if self._session_scar_count >= self._session_scar_cap:
                    # Skip scar creation when cap reached
                    continue
                scar = Scar(dimension=d, timestamp=timestamp)
                self.scars.append(scar)
                new_scars.append(d)
                self._session_scar_count += 1

        # Circuit breaker trigger: check for rapid scar formation
        if new_scars:
            # 新伤痕产生，使 modifier 缓存失效
            self._invalidate_modifier_cache()
            self._recent_scar_ticks.append(self._tick)
            self._recent_scar_ticks = [t for t in self._recent_scar_ticks if self._tick - t <= 10]
            if len(self._recent_scar_ticks) >= 5 and not self._circuit_breaker_active:
                self._circuit_breaker_active = True
                self._circuit_breaker_remaining = 30

        # Step 4: Healing (using configurable per-dimension rates)
        # Only heal pre-existing scars; newly formed scars skip their birth tick.
        healed: list[int] = []
        if heal:
            # 预计算 per-dim scar count，避免 O(n²)——主循环和奖励愈合共用
            _dim_counts: dict[int, int] = {}
            for s in self.scars[:existing_count]:
                _dim_counts[s.dimension] = _dim_counts.get(s.dimension, 0) + 1

            # Time-aware healing: grant bonus ticks for real-time silence
            if timestamp > 0 and self._last_step_time > 0:
                elapsed_minutes = (timestamp - self._last_step_time) / 60.0
                bonus_ticks = int(elapsed_minutes / 5.0)  # 1 bonus tick per 5 min silence
                bonus_ticks = min(bonus_ticks, 10)  # cap at 10 bonus ticks
                for _ in range(bonus_ticks):
                    self._heal_one_tick(existing_count, healed, _dim_counts)
            # v2.6.0 T-Persist (persist #1): only a REAL wall-clock advances the
            # healing clock. feedback() calls step() with timestamp=0.0 ("no time
            # signal"); the old unconditional assignment zeroed _last_step_time,
            # which silently dropped the next real step's silence-bonus healing.
            # GATED on _affect_enabled (red-team #5): affect OFF keeps the exact legacy
            # unconditional assignment (byte-identical); the fix engages only under the
            # v2.6 affect feature.
            if timestamp > 0 or not self._affect_enabled:
                self._last_step_time = timestamp

            for scar in self.scars[:existing_count]:
                if scar.stage == HealingStage.FADED:
                    continue
                scar.ticks_in_stage += 1
                threshold = self.healing_duration(
                    scar.stage, dim=scar.dimension, _dim_counts=_dim_counts
                )
                if threshold > 0 and scar.ticks_in_stage >= threshold:
                    scar.stage = HealingStage(scar.stage + 1)
                    scar.ticks_in_stage = 0
                    healed.append(scar.dimension)

            # Prune excess FADED scars to prevent unbounded growth
            faded = [s for s in self.scars if s.stage == HealingStage.FADED]
            if len(faded) > 50:
                self.scars = [s for s in self.scars if s.stage != HealingStage.FADED] + faded[-50:]

            # 愈合/修剪导致伤痕阶段变化或数量变化，使缓存失效
            if healed or len(faded) > 50:
                self._invalidate_modifier_cache()

        return {
            "modulated": modulated,
            "new_scars": new_scars,
            "healed_dimensions": healed,
            "total_scars": len(self.scars),
            "base": list(self.base),
        }

    def _heal_one_tick(
        self,
        existing_count: int,
        healed: list[int],
        _dim_counts: dict[int, int] | None = None,
    ) -> None:
        """执行一次愈合 tick（用于时间感知的奖励愈合）。"""
        for scar in self.scars[:existing_count]:
            if scar.stage == HealingStage.FADED:
                continue
            scar.ticks_in_stage += 1
            threshold = self.healing_duration(
                scar.stage, dim=scar.dimension, _dim_counts=_dim_counts
            )
            if threshold > 0 and scar.ticks_in_stage >= threshold:
                scar.stage = HealingStage(scar.stage + 1)
                scar.ticks_in_stage = 0
                healed.append(scar.dimension)
                # 阶段转换改变 alpha，使缓存失效
                self._invalidate_modifier_cache()

    def reset_session(self) -> None:
        """重置会话伤痕计数器（在会话边界调用）。"""
        self._session_scar_count = 0

    def set_session_cap(self, sovereignty: float) -> None:
        """根据主权性设置会话伤痕上限。

        高主权性 = 更低的上限（更受保护）：范围 2-8。
        这是"免疫系统"机制——防止单次会话中被过度伤害。
        """
        self._session_scar_cap = max(2, int(3 + (1 - sovereignty) * 5))

    def observe(self) -> dict[str, float]:
        """可观测输出：基向量状态 + 每维度灵敏度（供下游层使用）。

        优化：预先确保 modifier 缓存有效，避免 8 次重复遍历伤痕列表。
        """
        # 一次性构建缓存，后续 modifier(d) 直接查表
        self._ensure_modifier_cache()
        obs = {}
        for d in range(self.n_dims):
            obs[f"dim_{d}"] = self.base[d]
            obs[f"sensitivity_{d}"] = self._modifier_cache[d]
        obs["total_scars"] = float(len(self.scars))
        obs["numbed_dimensions"] = float(
            sum(1 for d in range(self.n_dims) if self._modifier_cache[d] < 0.5)
        )
        return obs

    def is_numbed(self, dim: int) -> bool:
        """判断某维度是否已被伤痕"麻木"（modifier < 0.5）。"""
        return self.modifier(dim) < 0.5

    def scar_density(self, dim: int) -> float:
        """计算某维度的加权伤痕密度（RAW 权重最高，FADED 最低）。"""
        weights = {
            HealingStage.RAW: 1.0,
            HealingStage.CLOSING: 0.8,
            HealingStage.SCARRED: 0.5,
            HealingStage.FADED: 0.3,
        }
        return sum(weights[s.stage] for s in self.scars if s.dimension == dim)

    # ------------------------------------------------------------------
    # Item 38: 伤痕愈合仪式
    # ------------------------------------------------------------------

    def check_heal_ritual(self) -> str | None:
        """检查是否有伤痕满足愈合仪式条件。

        条件：某个伤痕的 repair_count >= 5 且 temperature < 0.2（已冷却）。
        由于 Scar 数据类没有 repair_count/temperature 字段，
        这里使用 ticks_in_stage 作为修复计数代理（FADED 阶段 tick 数 >= 5），
        并以 alpha < 0.8 作为冷却判断（FADED 阶段 alpha=0.7 满足）。

        满足条件时：
        - 将该伤痕标记为已愈合（设置 stage 为 FADED，ticks_in_stage 归零）
        - 返回愈合提示文本

        Returns:
            愈合提示字符串，或 None（无符合条件的伤痕）。
        """
        for scar in self.scars:
            # repair_count 代理：SCARRED/FADED 阶段且累计 tick >= 5
            # temperature 代理：alpha < 0.8（FADED 阶段 alpha=0.7）
            repair_proxy = scar.ticks_in_stage
            temp_proxy = scar.alpha
            if repair_proxy >= 5 and temp_proxy < 0.8:
                # 标记为已愈合
                scar.stage = HealingStage.FADED
                scar.ticks_in_stage = 0
                self._invalidate_modifier_cache()
                return "一道旧伤正在愈合——曾经敏感的地方，现在可以轻轻触碰了"
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base": list(self.base),
            "scars": [s.to_dict() for s in self.scars],
            "n_dims": self.n_dims,
            "wound_threshold": self.wound_threshold,
            "tick": self._tick,
            "t_raw": self._t_raw,
            "t_closing": self._t_closing,
            "t_scarred": self._t_scarred,
            # Session scar cap
            "session_scar_count": self._session_scar_count,
            "session_scar_cap": self._session_scar_cap,
            # Circuit breaker
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_remaining": self._circuit_breaker_remaining,
            "recent_scar_ticks": self._recent_scar_ticks,
            # Time-aware healing
            "last_step_time": self._last_step_time,
        }
        # PEL-Core: additive sub-key, only present when the latent core is live.
        # Absent entirely when PEL is off => byte-identical legacy snapshots.
        if self._pel is not None:
            out["pel"] = self._pel.to_dict()
        # v2.6.0 affect: only the affect wall-clock survives restart (shadow buffer +
        # traits are re-supplied via apply_personality, never persisted). Emitted ONLY
        # when affect is enabled => byte-identical legacy snapshots when off.
        # ``e_ver`` (dormant base version) rides the same enable gate.
        if self._affect_enabled:
            out["e_last_wall_ts"] = self._e_last_wall_ts
            out["e_ver"] = self._e_ver
            # A.2：学习态（增益/资格迹/quality 基线）仅在可塑性开启时落盘——
            # 学习到的 G 是必须跨重启延续的状态（区别于可由人格重导出的参数）。
            if self._affect_plasticity and self._affect_gain is not None:
                out["affect_gain"] = list(self._affect_gain)
                out["affect_phi"] = list(self._affect_phi)
                out["affect_q_ema"] = self._affect_q_ema
            elif self._affect_gain is not None:
                # 可塑性被降级但内存里还有学习态：本次落盘将**永久**丢掉学到的增益
                # （数天聊天的学习量）。按字节一致纪律仍不落盘，但把数据丢失喊出来
                # （红队：静默不可逆丢失是运维暗坑）。
                logger.warning(
                    "affect plasticity disabled with learned gains in memory; "
                    "learned state will NOT be persisted and is lost on restart"
                )
        return out

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        pel_enabled: bool = False,
        affect_enabled: bool = False,
        brain_capability: object | None = None,
        authoritative_base: Sequence[float] | None = None,
    ) -> ScarredState:
        state = cls(
            n_dims=data["n_dims"],
            wound_threshold=data["wound_threshold"],
            pel_enabled=pel_enabled,
            affect_enabled=affect_enabled,
            brain_capability=brain_capability,
            authoritative_base=authoritative_base,
        )
        # base 的存储契约是 tanh 值域 [-1,1]（合法写入者只有 tanh 演化 / PEL 读出 /
        # E 律折返，全部有界）。复原时硬性执行该契约：损坏/手改快照的越界值在此夹回，
        # 否则会穿透 E 律衰减的凸组合前提（定理 2 需 u₀∈[0,1]）并永久越界（数学红队
        # composition fatal：base=1.5 经 decay 得 1.49…，不变集失守）。合法快照恒等。
        if brain_capability is None:
            state._replace_legacy_base([min(1.0, max(-1.0, float(x))) for x in data["base"]])
        state.scars = [Scar.from_dict(s) for s in data.get("scars", [])]
        state._tick = data.get("tick", 0)
        state._t_raw = data.get("t_raw", 10)
        state._t_closing = data.get("t_closing", 40)
        state._t_scarred = data.get("t_scarred", 150)
        # Session scar cap
        state._session_scar_count = data.get("session_scar_count", 0)
        state._session_scar_cap = data.get("session_scar_cap", 3)
        # Circuit breaker
        state._circuit_breaker_active = data.get("circuit_breaker_active", False)
        state._circuit_breaker_remaining = data.get("circuit_breaker_remaining", 0)
        state._recent_scar_ticks = data.get("recent_scar_ticks", [])
        # Time-aware healing
        state._last_step_time = data.get("last_step_time", 0.0)
        # v2.6.0 affect wall-clock + dormant base version (additive; old snapshots
        # default). from_dict then does one base mutation? No — restore never steps,
        # so the restored _e_ver is exactly the persisted value.
        state._e_last_wall_ts = data.get("e_last_wall_ts", 0.0)
        state._e_ver = int(data.get("e_ver", 0))
        # A.2 学习态复原（additive；缺键 = 未学习过，懒初始化会重新从人格导出）。
        # 增益经 Π_{[ε,1]} 语义夹回（复原边界执行学习态自己的域契约）。
        # gemini review：gain/phi 与下方 q_ema 一样包 try/except——损坏快照里的非数值
        # 学习态不得让整个会话 from_dict 崩（复原边界 fail-soft，回落未学习/中性）。
        # gemini review：float("nan") 不抛，会绕过 try/except 静默钉进学习态——显式
        # isfinite 闸把非有限值也判成损坏；sourcery review：回落时 warning 使坏快照可观测。
        # isfinite 闸必须验**原始** float——clamp 会把 nan 先钉成边界（max(0.05, nan)=0.05），
        # 若在 clamp 后检查就永远看不到非有限值（PR #28 gemini：float("nan") 绕过守卫）。
        raw_gain = data.get("affect_gain")
        if isinstance(raw_gain, list) and len(raw_gain) == state.n_dims:
            try:
                floats = [float(x) for x in raw_gain]
                if any(not math.isfinite(f) for f in floats):
                    raise ValueError("non-finite affect_gain")
                state._affect_gain = [min(1.0, max(0.05, f)) for f in floats]
            except (TypeError, ValueError):
                logger.warning("corrupt affect_gain in snapshot; dropping learned gains")
                state._affect_gain = None
        raw_phi = data.get("affect_phi")
        if isinstance(raw_phi, list) and len(raw_phi) == state.n_dims:
            try:
                floats = [float(x) for x in raw_phi]
                if any(not math.isfinite(f) for f in floats):
                    raise ValueError("non-finite affect_phi")
                state._affect_phi = [min(1.0, max(0.0, f)) for f in floats]
            except (TypeError, ValueError):
                logger.warning("corrupt affect_phi in snapshot; resetting eligibility trace")
                state._affect_phi = [0.0] * state.n_dims
        try:
            q = float(data.get("affect_q_ema", 0.5))
            state._affect_q_ema = min(1.0, max(0.0, q)) if math.isfinite(q) else 0.5
        except (TypeError, ValueError):
            state._affect_q_ema = 0.5
        # PEL-Core: migration-safe restore, GATED ON THE HOST'S CONFIG FLAG
        # (``pel_enabled``), never on snapshot contents (must-fix #3). A present
        # "pel" sub-key alone must NOT re-enable PEL when the caller has the flag
        # off — otherwise a snapshot could smuggle PEL on and break the
        # "flag off => byte-identical legacy" invariant. Old snapshots (no "pel")
        # always stay on the legacy path. NOTE (must-fix #2): a restored v1 "pel"
        # (no ``pi0``) anchors to the already-drifted ``pi``, not the true trait
        # prior; recover real identity by re-calling ``set_pel_priors`` on load.
        pel = data.get("pel")
        if pel is not None and pel_enabled:
            state._pel = PELCore.from_dict(pel)
        return state
