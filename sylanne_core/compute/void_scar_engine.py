"""Sylanne-Embodiment 计算核心层：虚空-伤痕耦合引擎（Void-Scar Coupled Engine）。

在 7 层计算栈中的位置：L3 层的统一入口，替代了原始架构中的 SSM + TDA 层。
职责：将伤痕代数（不可逆状态动力学）与虚空微积分（一等缺席计算）通过双向耦合整合：
  Γ 耦合：虚空压力 → 伤痕创伤事件（压力积累到阈值时触发创伤）
  Φ 耦合：伤痕麻木 → 虚空检测灵敏度（麻木维度降低虚空检测阈值）

输出 8 维情感空间：warmth, arousal, valence, tension, curiosity,
repair_pressure, expression_drive, boundary_firmness。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from . import affect_projection
from . import pel_core as _pel_core  # module ref so SEMANTIC_PRIOR stays monkeypatchable
from .brain_backend import BrainFeedbackRequest, BrainStepRequest
from .brain_backend_coordinator import (
    BrainBackendCoordinator,
    CoordinatedFeedback,
    CoordinatedStep,
)
from .brain_c import BrainCState, Route
from .brain_codec import BrainBundle
from .brain_compute import BrainComputeCore, BrainEvent, compose_appraisal
from .brain_errors import (
    BrainAllocationError,
    BrainDurabilityError,
    BrainOwnershipError,
    BrainValidationError,
)
from .brain_state import BrainState
from .brain_store import (
    AppliedFeedbackCommit,
    EventAllocated,
    EventCommit,
    EventCommitted,
    EventDuplicate,
    FeedbackCommitted,
    FeedbackDuplicate,
    ReceiptOnlyFeedbackCommit,
    SessionLoaded,
    StoredReceipt,
    event_id_digest,
    feedback_id_digest,
    session_digest,
)
from .pel_core import N as _PEL_N
from .scar_algebra import ScarredState
from .void_calculus import VoidSpace

if TYPE_CHECKING:
    from ..config import BrainComputeConfig
    from .brain_store import BrainStateStore


class BrainSessionContext:
    """Bind one authoritative store row to B, optional C, and one base capability."""

    __slots__ = (
        "_bound_scar_state",
        "_capability",
        "_checkpoint",
        "_config",
        "_coordinator",
        "_core",
        "_needs_reload",
        "_session_digest",
        "_store",
        "last_receipt",
    )

    def __init__(
        self,
        *,
        config: BrainComputeConfig,
        store: BrainStateStore,
        session_id: str,
    ) -> None:
        if not config.enabled:
            raise BrainValidationError("BrainSessionContext requires enabled brain compute")
        self._config = config
        self._store = store
        self._session_digest = session_digest(session_id)
        self._capability = object()
        self._bound_scar_state: ScarredState | None = None
        self._needs_reload = False
        self.last_receipt: dict[str, object] | None = None

        loaded = self._store.load(self._session_digest)
        if isinstance(loaded, SessionLoaded):
            bundle = loaded.bundle
            self._checkpoint = loaded.checkpoint
        else:
            bundle = BrainBundle(
                BrainComputeCore.fresh(feedback_horizon=config.feedback_horizon).state,
                BrainCState.fresh(feedback_horizon=config.feedback_horizon),
            )
            self._checkpoint = None
        self._core = BrainComputeCore(bundle.b)
        self._coordinator = self._new_coordinator(bundle)

    @property
    def _authoritative_base(self) -> tuple[float, ...]:
        return tuple(self._core.state.e)

    @property
    def state(self) -> BrainState:
        return self._core.state

    @property
    def sparse_routing(self) -> bool:
        return self._config.sparse_routing

    @property
    def reference(self) -> dict[str, object]:
        state = self._core.state
        return {
            "schema": 1,
            "session_digest": self._session_digest.hex(),
            "generation": state.generation,
            "lineage_id": state.lineage_id,
            "mutation_seq": state.mutation_seq,
        }

    def _create_scar_state(
        self,
        *,
        n_dims: int,
        wound_threshold: float,
        mlp_passes: int,
        pel_enabled: bool,
        affect_enabled: bool,
    ) -> ScarredState:
        state = ScarredState(
            n_dims=n_dims,
            wound_threshold=wound_threshold,
            mlp_passes=mlp_passes,
            pel_enabled=pel_enabled,
            affect_enabled=affect_enabled,
            brain_capability=self._capability,
            authoritative_base=self._authoritative_base,
        )
        self._bound_scar_state = state
        return state

    def _restore_scar_state(
        self,
        data: dict[str, Any],
        *,
        pel_enabled: bool,
        affect_enabled: bool,
    ) -> ScarredState:
        state = ScarredState.from_dict(
            data,
            pel_enabled=pel_enabled,
            affect_enabled=affect_enabled,
            brain_capability=self._capability,
            authoritative_base=self._authoritative_base,
        )
        self._bound_scar_state = state
        return state

    def _new_coordinator(self, bundle: BrainBundle) -> BrainBackendCoordinator | None:
        if not self._config.c_enabled:
            return None
        return BrainBackendCoordinator(
            session_digest=self._session_digest,
            c_state=bundle.c,
            primary_name=self._config.c_backend,
            acknowledged_checkpoint=self._checkpoint,
            primary_alpha=self._config.c_authority,
            promoted=self._config.c_promoted,
        )

    def _replace_runtime(self, loaded: SessionLoaded) -> None:
        coordinator = self._coordinator
        if coordinator is not None:
            coordinator.close()
        self._checkpoint = loaded.checkpoint
        self._core = BrainComputeCore(loaded.bundle.b)
        self._coordinator = self._new_coordinator(loaded.bundle)
        self._needs_reload = False
        if self._bound_scar_state is not None:
            self._bound_scar_state._replace_base(self._authoritative_base, self._capability)

    def _recover_authoritative(self) -> None:
        loaded = self._store.load(self._session_digest)
        if not isinstance(loaded, SessionLoaded):
            self._needs_reload = True
            raise BrainDurabilityError("authoritative brain session disappeared during recovery")
        self._replace_runtime(loaded)

    def close(self) -> None:
        """Release the optional process-backed C coordinator for this session."""
        coordinator = self._coordinator
        self._coordinator = None
        if coordinator is not None:
            coordinator.close()

    @staticmethod
    def _public_receipt(receipt: StoredReceipt, *, status: str | None = None) -> dict[str, object]:
        public: dict[str, object] = {
            "status": receipt.status if status is None else status,
            "generation": receipt.generation,
            "tick_id": receipt.tick_id,
            "history_epoch": receipt.history_epoch,
            "mutation_seq": receipt.mutation_seq,
        }
        if receipt.target_tick is not None:
            public["target_tick"] = receipt.target_tick
            public["applied_dimensions"] = receipt.applied_dimensions
            public["applied_synapses"] = receipt.applied_synapses
        return public

    def preflight_event(self, event_id: str) -> EventAllocated | dict[str, object]:
        allocated_or_duplicate = self._store.preflight_allocate(
            self._session_digest,
            event_id_digest(event_id),
        )
        if isinstance(allocated_or_duplicate, EventDuplicate):
            self.last_receipt = self._public_receipt(
                allocated_or_duplicate.receipt,
                status="duplicate",
            )
            return dict(self.last_receipt)
        return allocated_or_duplicate

    def process_event(
        self,
        *,
        event_id: str,
        allocated: EventAllocated | None = None,
        assessment: list[float] | tuple[float, ...] | None,
        hdc: list[float] | tuple[float, ...],
        wound_sum: list[float] | tuple[float, ...],
        surprise: float,
        perception_acuity: float,
        route: Route,
    ) -> dict[str, object]:
        if allocated is None:
            preflight = self.preflight_event(event_id)
            if isinstance(preflight, dict):
                return preflight
            allocated = preflight
        now = time.time()
        appraisal = tuple(compose_appraisal(assessment, hdc, wound_sum))
        current = self._core.state
        authoritative = allocated.bundle.b
        runtime_rebased = self._needs_reload or (
            current.generation,
            current.lineage_id,
            current.mutation_seq,
        ) != (
            authoritative.generation,
            authoritative.lineage_id,
            authoritative.mutation_seq,
        )
        working_core = BrainComputeCore(authoritative) if runtime_rebased else self._core
        coordinator = (
            self._new_coordinator(allocated.bundle) if runtime_rebased else self._coordinator
        )
        temporary_coordinator = coordinator is not self._coordinator
        coordinated: CoordinatedStep | None = None
        proposal: tuple[float, ...] = (0.0,) * 8
        c_state = allocated.bundle.c
        alpha_c = 0.0
        checkpoint = None
        try:
            if coordinator is not None:
                coordinated = coordinator.prepare_step(
                    BrainStepRequest(
                        request_id=(
                            f"event:{allocated.allocation.generation}:"
                            f"{allocated.allocation.mutation_seq}"
                        ),
                        event_id=event_id,
                        tick_id=allocated.allocation.tick_id,
                        expected_state_version=coordinator.primary_state_version,
                        event=appraisal,
                    ),
                    allocation=allocated.allocation,
                    route=route,
                    created_at=now,
                    delta_t=max(0.0, now - allocated.bundle.b.clock),
                    timeout_ms=max(1, math.ceil(self._config.c_timeout_ms)),
                )
                proposal = coordinated.proposal
                c_state = coordinated.c_candidate.state
                alpha_c = coordinated.authority_alpha
                checkpoint = coordinated.provisional_checkpoint

            event = BrainEvent(
                event_id=event_id,
                assessment=assessment,
                hdc=hdc,
                wound_sum=wound_sum,
                surprise=surprise,
                perception_acuity=perception_acuity,
                proposal_c=proposal,
            )
            candidate = working_core.prepare_event(
                event,
                allocation=allocated.allocation,
                trusted_now=now,
                alpha_c=alpha_c,
                c_residual_cap=self._config.c_residual_cap,
            )
            bundle = BrainBundle(candidate.state, c_state)
            receipt = StoredReceipt(
                kind="event",
                status="degraded"
                if coordinated is not None and coordinated.degraded
                else "applied",
                generation=candidate.state.generation,
                tick_id=candidate.state.tick_id,
                history_epoch=candidate.state.history_epoch,
                mutation_seq=candidate.state.mutation_seq,
            )
            commit = EventCommit(
                allocated=allocated,
                bundle=bundle,
                receipt=receipt,
                checkpoint=checkpoint,
            )

            if coordinated is None:
                acknowledgement = self._store.commit_event(
                    self._session_digest,
                    event_id_digest(event_id),
                    commit,
                )
                if not isinstance(acknowledgement, EventCommitted):
                    raise BrainDurabilityError("event became duplicate during serialized commit")
            else:
                assert coordinator is not None
                coordinator.commit(
                    coordinated,
                    lambda _coordinated: cast(
                        EventCommitted,
                        self._store.commit_event(
                            self._session_digest,
                            event_id_digest(event_id),
                            commit,
                        ),
                    ),
                )
        except BaseException:
            self._needs_reload = True
            if coordinated is not None and coordinator is not None:
                try:
                    coordinator.reject(
                        coordinated, "event integration failed before acknowledgement"
                    )
                except BaseException:
                    pass
            if temporary_coordinator and coordinator is not None:
                try:
                    coordinator.close()
                except BaseException:
                    pass
            try:
                loaded_after_failure = self._store.load(self._session_digest)
            except BaseException:
                loaded_after_failure = None
            if (
                isinstance(loaded_after_failure, SessionLoaded)
                and loaded_after_failure.bundle.b.generation == allocated.allocation.generation
                and loaded_after_failure.bundle.b.lineage_id == allocated.allocation.lineage_id
                and loaded_after_failure.bundle.b.mutation_seq >= allocated.allocation.mutation_seq
            ):
                try:
                    self._replace_runtime(loaded_after_failure)
                except BaseException:
                    self._needs_reload = True
            raise

        try:
            committed = working_core.commit(candidate)
            if self._bound_scar_state is None:
                raise BrainOwnershipError("brain base has no bound ScarredState")
            self._bound_scar_state._replace_base(tuple(committed.e), self._capability)
            previous_coordinator = self._coordinator
            self._core = working_core
            if temporary_coordinator:
                self._coordinator = coordinator
            self._checkpoint = checkpoint
            self._needs_reload = False
            if temporary_coordinator and previous_coordinator is not None:
                previous_coordinator.close()
        except BaseException:
            self._needs_reload = True
            if (
                temporary_coordinator
                and self._coordinator is not coordinator
                and coordinator is not None
            ):
                try:
                    coordinator.close()
                except BaseException:
                    pass
            try:
                self._recover_authoritative()
            except BaseException:
                pass
            raise

        self.last_receipt = self._public_receipt(receipt)
        return dict(self.last_receipt)

    def apply_targeted_feedback(
        self,
        *,
        feedback_id: str,
        target_tick: int,
        value: float,
        confidence: float,
    ) -> dict[str, object]:
        """Apply delayed feedback to one persisted eligibility trace."""
        feedback_key = feedback_id_digest(feedback_id)
        duplicate = self._store.lookup_feedback_receipt(self._session_digest, feedback_key)
        if isinstance(duplicate, FeedbackDuplicate):
            self.last_receipt = self._public_receipt(duplicate.receipt, status="duplicate")
            return dict(self.last_receipt)

        loaded = self._store.load(self._session_digest)
        if not isinstance(loaded, SessionLoaded):
            raise BrainDurabilityError("targeted feedback requires an active brain session")
        trusted_now = time.time()
        probe_core = BrainComputeCore(loaded.bundle.b)
        try:
            receipt_only_candidate = probe_core.prepare_feedback(
                target_tick=target_tick,
                value=value,
                confidence=confidence,
                trusted_now=trusted_now,
                feedback_ttl_seconds=self._config.feedback_ttl_seconds,
                allocation=None,
            )
        except BrainAllocationError:
            receipt_only_candidate = None

        if receipt_only_candidate is not None:
            state = receipt_only_candidate.state
            receipt = StoredReceipt(
                kind="feedback",
                status=receipt_only_candidate.status,
                generation=state.generation,
                tick_id=state.tick_id,
                history_epoch=state.history_epoch,
                mutation_seq=state.mutation_seq,
                target_tick=target_tick,
            )
            acknowledgement = self._store.commit_feedback(
                self._session_digest,
                feedback_key,
                ReceiptOnlyFeedbackCommit(receipt),
            )
            if isinstance(acknowledgement, FeedbackDuplicate):
                receipt = acknowledgement.receipt
                self.last_receipt = self._public_receipt(receipt, status="duplicate")
            else:
                self.last_receipt = self._public_receipt(receipt)
            return dict(self.last_receipt)

        allocated_or_duplicate = self._store.preflight_feedback(
            self._session_digest,
            feedback_key,
            target_tick=target_tick,
        )
        if isinstance(allocated_or_duplicate, FeedbackDuplicate):
            self.last_receipt = self._public_receipt(
                allocated_or_duplicate.receipt,
                status="duplicate",
            )
            return dict(self.last_receipt)
        allocated = allocated_or_duplicate
        current = self._core.state
        authoritative = allocated.bundle.b
        runtime_rebased = self._needs_reload or (
            current.generation,
            current.lineage_id,
            current.mutation_seq,
        ) != (
            authoritative.generation,
            authoritative.lineage_id,
            authoritative.mutation_seq,
        )
        working_core = BrainComputeCore(authoritative) if runtime_rebased else self._core
        coordinator = (
            self._new_coordinator(allocated.bundle) if runtime_rebased else self._coordinator
        )
        temporary_coordinator = coordinator is not self._coordinator
        coordinated: CoordinatedFeedback | None = None
        checkpoint = None
        candidate = working_core.prepare_feedback(
            target_tick=target_tick,
            value=value,
            confidence=confidence,
            trusted_now=trusted_now,
            feedback_ttl_seconds=self._config.feedback_ttl_seconds,
            allocation=allocated.allocation,
        )
        c_state = allocated.bundle.c
        applied_synapses = 0

        try:
            if coordinator is not None:
                coordinated = coordinator.prepare_feedback(
                    BrainFeedbackRequest(
                        request_id=(
                            f"feedback:{allocated.allocation.generation}:"
                            f"{allocated.allocation.next_mutation_seq}"
                        ),
                        feedback_id=feedback_id,
                        target_tick=target_tick,
                        expected_state_version=coordinator.primary_state_version,
                        value=value,
                        confidence=confidence,
                    ),
                    allocation=allocated.allocation,
                    state_tick=authoritative.tick_id,
                    trusted_now=trusted_now,
                    state_clock=authoritative.clock,
                    feedback_ttl_seconds=self._config.feedback_ttl_seconds,
                    b_changed=candidate.status == "applied",
                    timeout_ms=max(1, math.ceil(self._config.c_timeout_ms)),
                )
                c_state = coordinated.c_candidate.state
                checkpoint = coordinated.provisional_checkpoint
                applied_synapses = coordinated.applied_synapses

            status: Literal["applied", "degraded"] = (
                "degraded" if coordinated is not None and coordinated.degraded else "applied"
            )
            receipt = StoredReceipt(
                kind="feedback",
                status=status,
                generation=candidate.state.generation,
                tick_id=candidate.state.tick_id,
                history_epoch=candidate.state.history_epoch,
                mutation_seq=candidate.state.mutation_seq,
                target_tick=target_tick,
                applied_dimensions=candidate.applied_dimensions,
                applied_synapses=applied_synapses,
            )
            commit = AppliedFeedbackCommit(
                allocated,
                BrainBundle(candidate.state, c_state),
                receipt,
                checkpoint,
            )
            if coordinated is None:
                acknowledgement = self._store.commit_feedback(
                    self._session_digest,
                    feedback_key,
                    commit,
                )
                if not isinstance(acknowledgement, FeedbackCommitted):
                    raise BrainDurabilityError("feedback became duplicate during serialized commit")
            else:
                assert coordinator is not None
                coordinator.commit(
                    coordinated,
                    lambda _coordinated: cast(
                        FeedbackCommitted,
                        self._store.commit_feedback(
                            self._session_digest,
                            feedback_key,
                            commit,
                        ),
                    ),
                )
        except BaseException:
            self._needs_reload = True
            if coordinated is not None and coordinator is not None:
                try:
                    coordinator.reject(coordinated, "feedback integration failed")
                except BaseException:
                    pass
            if temporary_coordinator and coordinator is not None:
                try:
                    coordinator.close()
                except BaseException:
                    pass
            try:
                loaded_after_failure = self._store.load(self._session_digest)
            except BaseException:
                loaded_after_failure = None
            if (
                isinstance(loaded_after_failure, SessionLoaded)
                and loaded_after_failure.bundle.b.generation == allocated.allocation.generation
                and loaded_after_failure.bundle.b.lineage_id == allocated.allocation.lineage_id
                and loaded_after_failure.bundle.b.mutation_seq
                >= allocated.allocation.next_mutation_seq
            ):
                try:
                    self._replace_runtime(loaded_after_failure)
                except BaseException:
                    self._needs_reload = True
            raise

        try:
            committed = working_core.commit(candidate)
            if self._bound_scar_state is None:
                raise BrainOwnershipError("brain base has no bound ScarredState")
            self._bound_scar_state._replace_base(tuple(committed.e), self._capability)
            previous_coordinator = self._coordinator
            self._core = working_core
            if temporary_coordinator:
                self._coordinator = coordinator
            self._checkpoint = checkpoint
            self._needs_reload = False
            if temporary_coordinator and previous_coordinator is not None:
                previous_coordinator.close()
        except BaseException:
            self._needs_reload = True
            if (
                temporary_coordinator
                and self._coordinator is not coordinator
                and coordinator is not None
            ):
                try:
                    coordinator.close()
                except BaseException:
                    pass
            try:
                self._recover_authoritative()
            except BaseException:
                pass
            raise

        self.last_receipt = self._public_receipt(receipt)
        return dict(self.last_receipt)


def _assessment_float(value: object, *, lower: float, upper: float) -> float:
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(converted):
        return 0.0
    return min(upper, max(lower, converted))


def project_brain_assessment(
    assessment: dict[str, Any] | None,
) -> tuple[list[float] | None, list[float]]:
    """Project assessor fields before VoidScar and derive its scar-forming wound."""
    if not assessment:
        return None, [0.0] * 8
    valence = _assessment_float(assessment.get("valence"), lower=-1.0, upper=1.0)
    arousal = _assessment_float(assessment.get("arousal"), lower=0.0, upper=1.0)
    wound_risk = _assessment_float(assessment.get("wound_risk"), lower=0.0, upper=1.0)
    intent_value = assessment.get("intent")
    intent = None if intent_value is None else str(intent_value)
    projected, _matched = affect_projection.project_appraisal(
        valence,
        arousal,
        wound_risk,
        intent,
    )
    wound = [0.0] * 8
    if wound_risk > 0.7:
        wound[3] = wound_risk * 0.8
        wound[5] = wound_risk * 0.5
    return projected, wound


class SocialVoid:
    """群聊沉默虚空——当 agent 在活跃群聊中保持沉默时，压力持续积累。

    模拟"群里大家都在聊，我却没说话"的社交压力。
    与 VoidSpace 中的个人虚空不同，这是纯社交层面的压力源。
    """

    __slots__ = ("pressure", "silence_ticks", "group_activity", "topic_boundary")

    def __init__(self) -> None:
        self.pressure = 0.0
        self.silence_ticks = 0
        self.group_activity = 0.0
        self.topic_boundary = 0.5

    def tick(self, group_active: bool = True) -> None:
        if not group_active:
            self.pressure *= 0.95
            return
        self.silence_ticks += 1
        depth = self.group_activity
        beta = self.topic_boundary
        if depth > 0 and self.silence_ticks > 0:
            self.pressure += depth * math.log(self.silence_ticks + 1) * (1.0 - beta) * 0.1
        self.pressure = min(5.0, self.pressure)

    def reset(self) -> None:
        self.silence_ticks = 0
        self.pressure *= 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "pressure": self.pressure,
            "silence_ticks": self.silence_ticks,
            "group_activity": self.group_activity,
            "topic_boundary": self.topic_boundary,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        self.pressure = float(data.get("pressure", 0.0))
        self.silence_ticks = int(data.get("silence_ticks", 0))
        self.group_activity = float(data.get("group_activity", 0.0))
        self.topic_boundary = float(data.get("topic_boundary", 0.5))


class VoidScarEngine:
    """虚空-伤痕耦合计算引擎。

    替代计算脊柱中原始的 SSM（L3）和 TDA（L4）层。
    通过双向耦合将两个独立的数学系统整合为统一的情感计算引擎：
      - Γ 耦合（虚空→伤痕）：虚空压力超过阈值时，向伤痕状态注入创伤事件
      - Φ 耦合（伤痕→虚空）：伤痕麻木的维度降低虚空检测阈值（更容易感知缺席）

    与其他组件的关系：
      - 被 ComputationSpine.process() 在 L3 层调用
      - 接收 L1 HDC 编码和 L2 惊讶度
      - 输出 8 维情感观测给 L5 HGT 和 L7 表达层
      - expression_drive() 输出给 L7 PhaseTransitionExpression
    """

    __slots__ = (
        "scar_state",
        "void_space",
        "social_void",
        "similarity_fn",
        "_coherence",
        "_last_event_vec",
        "_tick",
        "_void_pressure_coupling_rate",
        "_void_drive_weight",
        "_social_drive_weight",
        "_accepted_decay",
        "_ignored_deepening",
        "_personality_detection_floor",
        "_cached_observe",
        # PEL-Core: 1-tick-deferred assessor affect store (D-2) for x_t assembly.
        "_pel_affect",
        "_pel_confidence",
        "_brain",
    )

    def __init__(
        self,
        n_dims: int = 8,
        wound_threshold: float = 0.6,
        similarity_fn: Callable[[bytes, bytes], float] | None = None,
        max_voids: int = 50,
        pressure_threshold: float = 10.0,
        scar_mlp_passes: int = 1,
        *,
        pel_enabled: bool = False,
        affect_enabled: bool = False,
        brain_context: BrainSessionContext | None = None,
    ):
        self._brain = brain_context
        if brain_context is None:
            self.scar_state = ScarredState(
                n_dims=n_dims,
                wound_threshold=wound_threshold,
                mlp_passes=scar_mlp_passes,
                pel_enabled=pel_enabled,
                affect_enabled=affect_enabled,
            )
        else:
            self.scar_state = brain_context._create_scar_state(
                n_dims=n_dims,
                wound_threshold=wound_threshold,
                mlp_passes=scar_mlp_passes,
                pel_enabled=pel_enabled,
                affect_enabled=affect_enabled,
            )
        self.similarity_fn = similarity_fn or _default_similarity
        self.void_space = VoidSpace(
            similarity_fn=self.similarity_fn,
            max_voids=max_voids,
            pressure_threshold=pressure_threshold,
        )
        self.social_void = SocialVoid()
        self._coherence = 1.0
        self._last_event_vec: bytes | None = None
        self._tick = 0
        self._void_pressure_coupling_rate = 0.3
        self._void_drive_weight = 0.5
        self._social_drive_weight = 0.3
        self._accepted_decay = 0.7
        self._ignored_deepening = 0.05
        self._personality_detection_floor: float = 0.1
        self._cached_observe: dict[str, float] | None = None
        # PEL deferred affect: the last assessor read (8-dim a_vec + confidence),
        # folded into next main tick's x_t. Zero/0.0 => x_t is surprise-scaled HDC
        # only (the unevaluated-tick case). Populated via ``store_pel_affect``.
        self._pel_affect: list[float] = [0.0] * _PEL_N
        self._pel_confidence: float = 0.0

    def process(
        self,
        event_vec: bytes,
        ssm_input: list[float],
        surprise: float,
        timestamp: float = 0.0,
        *,
        event_id: str | None = None,
        projected_assessment: list[float] | None = None,
        assessment_wound: list[float] | None = None,
        route: Route = "normal",
        perception_acuity: float = 0.5,
    ) -> dict[str, Any]:
        """处理一个事件通过耦合的虚空-伤痕引擎。

        执行顺序：
          1. Φ 耦合：伤痕麻木 → 降低虚空检测阈值
          2. 虚空微积分步进
          3. Γ 耦合：虚空压力超阈值 → 向伤痕注入创伤
          4. 伤痕代数步进（主事件）
          5. 计算全局一致性

        Args:
            event_vec: HDC 编码的事件向量（用于虚空边界操作）
            ssm_input: 8 维输入向量（用于伤痕状态演化）
            surprise: 来自预测编码门控的惊讶度
            timestamp: 事件时间戳

        Returns:
            包含伤痕状态、虚空状态、耦合信息和一致性的综合结果
        """
        allocated: EventAllocated | None = None
        if self._brain is not None:
            if not event_id:
                raise BrainValidationError("brain-enabled VoidScar events require event_id")
            preflight = self._brain.preflight_event(event_id)
            if isinstance(preflight, dict):
                return {
                    "scar": {
                        "modulated": [0.0] * self.scar_state.n_dims,
                        "new_scars": [],
                        "healed_dimensions": [],
                        "total_scars": len(self.scar_state.scars),
                        "base": list(self.scar_state.base),
                    },
                    "void": {},
                    "coupling_wounds": [],
                    "coherence": self._coherence,
                    "observation": self.observe(),
                    "brain_event": preflight,
                }
            allocated = preflight

        self._tick += 1
        self._cached_observe = None

        # Compute similarity to previous event (for void detection)
        prev_sim = 0.0
        if self._last_event_vec is not None:
            prev_sim = self.similarity_fn(event_vec, self._last_event_vec)
        self._last_event_vec = event_vec

        # --- Coupling Φ: Scars → Void sensitivity ---
        # Numbed dimensions lower void detection threshold, but respect personality floor
        numbed_count = sum(1 for d in range(self.scar_state.n_dims) if self.scar_state.is_numbed(d))
        if numbed_count > 0:
            # Phi coupling: numbed dims lower detection threshold, but respect floor
            personality_base = self.void_space._detection_threshold
            phi_floor = self._personality_detection_floor
            phi_adjusted = max(phi_floor, personality_base - numbed_count * 0.03)
            self.void_space._detection_threshold = phi_adjusted

        # --- Void Calculus step ---
        void_result = self.void_space.process(event_vec, surprise, prev_sim)

        # --- Coupling Γ: Void pressure → Scar wounding ---
        coupling_wounds: list[dict[str, Any]] = []
        wound_vectors: list[list[float]] = []
        for coupling in void_result["coupling_events"]:
            wound_event = [0.0] * self.scar_state.n_dims
            dim_hint = int(coupling.get("dim_hint", 0)) % self.scar_state.n_dims
            wound_event[dim_hint] = coupling["pressure"] * self._void_pressure_coupling_rate
            wound_result = self.scar_state.step(wound_event, timestamp, heal=False)
            coupling_wounds.append(wound_result)
            wound_vectors.append(list(wound_result["modulated"]))

        if self._brain is not None and assessment_wound and any(assessment_wound):
            assessment_result = self.scar_state.step(assessment_wound, timestamp, heal=False)
            wound_vectors.append(list(assessment_result["modulated"]))

        # --- Scar Algebra step (main event) ---
        # When PEL is active, assemble x_t = c*a_vec + (1-c)*s*h_t (design §3.1)
        # and pass it as the main-step context so the latent core drives base.
        pel_ctx = self._build_pel_ctx(ssm_input, surprise)
        scar_result = self.scar_state.step(ssm_input, timestamp, pel_ctx=pel_ctx)

        brain_event: dict[str, object] | None = None
        if self._brain is not None:
            wound_sum = [
                min(
                    1.0,
                    max(
                        -1.0,
                        math.fsum(
                            value
                            for vector in wound_vectors
                            for value in [vector[index] if index < len(vector) else 0.0]
                            if math.isfinite(value)
                        ),
                    ),
                )
                for index in range(8)
            ]
            brain_event = self._brain.process_event(
                event_id=cast(str, event_id),
                allocated=allocated,
                assessment=projected_assessment,
                hdc=ssm_input,
                wound_sum=wound_sum,
                surprise=surprise,
                perception_acuity=perception_acuity,
                route=route,
            )

        # --- Compute coherence (emergent resonance) ---
        self._coherence = self._compute_coherence()

        result = {
            "scar": scar_result,
            "void": void_result,
            "coupling_wounds": coupling_wounds,
            "coherence": self._coherence,
            "observation": self.observe(),
        }
        if brain_event is not None:
            result["brain_event"] = brain_event
        return result

    @property
    def brain_enabled(self) -> bool:
        return self._brain is not None

    @property
    def sparse_routing(self) -> bool:
        return self._brain is not None and self._brain.sparse_routing

    @property
    def brain_state(self) -> BrainState | None:
        if self._brain is None:
            return None
        return self._brain.state

    @property
    def brain_reference(self) -> dict[str, object] | None:
        if self._brain is None:
            return None
        return self._brain.reference

    def restore_scar_state(
        self,
        data: dict[str, Any],
        *,
        pel_enabled: bool,
        affect_enabled: bool,
    ) -> ScarredState:
        """Restore scar metadata without allowing JSON to replace authoritative B."""
        brain = self._brain
        restored = (
            ScarredState.from_dict(
                data,
                pel_enabled=pel_enabled,
                affect_enabled=affect_enabled,
            )
            if brain is None
            else brain._restore_scar_state(
                data,
                pel_enabled=pel_enabled,
                affect_enabled=affect_enabled,
            )
        )
        self.scar_state = restored
        return restored

    def commit_neutral_brain_event(self, event_id: str | None) -> dict[str, object]:
        """Advance authoritative B once when the legacy Void pipeline is unavailable."""
        if self._brain is None:
            raise BrainOwnershipError("neutral brain commits require an enabled brain context")
        if not event_id:
            raise BrainValidationError("brain-enabled VoidScar events require event_id")
        return self._brain.process_event(
            event_id=event_id,
            assessment=None,
            hdc=(0.0,) * 8,
            wound_sum=(0.0,) * 8,
            surprise=0.0,
            perception_acuity=0.5,
            route="normal",
        )

    def apply_targeted_feedback(
        self,
        *,
        feedback_id: str,
        target_tick: int,
        value: float,
        confidence: float,
    ) -> dict[str, object]:
        if self._brain is None:
            raise BrainOwnershipError("targeted feedback requires an enabled brain context")
        return self._brain.apply_targeted_feedback(
            feedback_id=feedback_id,
            target_tick=target_tick,
            value=value,
            confidence=confidence,
        )

    def close(self) -> None:
        """Release brain-session resources owned by this VoidScar instance."""
        if self._brain is not None:
            self._brain.close()

    def _build_pel_ctx(
        self, ssm_input: list[float], surprise: float
    ) -> tuple[list[float], float, list[float] | None, float] | None:
        """Assemble the PEL main-step context, or ``None`` when PEL is inactive.

        v2.5 redesign (B): ``x_t = s*h_t`` predicts the LIVE surprise-scaled HDC
        afferent, so ``e0 = x_t - W_gen*mu`` is a genuine prediction error (the fix
        for dead M1 precision); the deferred assessor affect (``a_vec``, confidence
        ``c``) is carried SEPARATELY as a precision-weighted semantic prior, NOT
        blended into ``x_t``. With ``SEMANTIC_PRIOR`` off this falls back to the
        legacy value-blend ``x_t = c*a_vec + (1-c)*s*h_t`` (no prior). ``x_t`` never
        contains prior latent state. Returns ``None`` => legacy MLP path (byte-
        identical to today).
        """
        if not self.scar_state.pel_active():
            return None
        c = self._pel_confidence
        a_vec = self._pel_affect
        if _pel_core.SEMANTIC_PRIOR:
            x_t = [surprise * (ssm_input[i] if i < len(ssm_input) else 0.0) for i in range(_PEL_N)]
            return x_t, surprise, list(a_vec), c
        x_t = [
            c * (a_vec[i] if i < len(a_vec) else 0.0)
            + (1.0 - c) * surprise * (ssm_input[i] if i < len(ssm_input) else 0.0)
            for i in range(_PEL_N)
        ]
        return x_t, surprise, None, 0.0

    def store_pel_affect(self, affect_vec: list[float], confidence: float) -> None:
        """Stash the assessor's affect read for the NEXT main tick's PEL input.

        Implements the 1-tick deferred fold (D-2): the main ``step()`` runs before
        the assessor result is available, so the read drives ``mu`` on the
        following tick. Safe to call regardless of whether PEL is enabled.
        """
        self._pel_affect = list(affect_vec)
        self._pel_confidence = max(0.0, min(1.0, confidence))

    # Canonical dimension names for the 8-dim emotion space
    _DIM_NAMES: tuple[str, ...] = (
        "warmth",
        "arousal",
        "valence",
        "tension",
        "curiosity",
        "repair_pressure",
        "expression_drive",
        "boundary_firmness",
    )

    def observe(self) -> dict[str, float]:
        """可观测输出：供下游层使用的命名情感维度。

        返回 8 个命名情感维度（warmth, arousal, valence, tension,
        curiosity, repair_pressure, expression_drive, boundary_firmness）
        加上 coherence, void_pressure, active_voids, ghost_count 等元信息。
        """
        if self._cached_observe is not None:
            return self._cached_observe
        raw = self.scar_state.observe()
        obs: dict[str, float] = {}
        for i, name in enumerate(self._DIM_NAMES):
            obs[name] = raw.get(f"dim_{i}", 0.0)
        for i, name in enumerate(self._DIM_NAMES):
            obs[f"sensitivity_{name}"] = raw.get(f"sensitivity_{i}", 1.0)
        obs["total_scars"] = raw.get("total_scars", 0.0)
        obs["numbed_dimensions"] = raw.get("numbed_dimensions", 0.0)
        obs["coherence"] = self._coherence
        obs["void_pressure"] = self.void_space.total_pressure()
        obs["active_voids"] = float(len(self.void_space.voids))
        obs["ghost_count"] = float(len(self.void_space.ghosts))
        self._cached_observe = obs
        return obs

    def expression_drive(self) -> float:
        """计算综合表达驱动力（供 L7 相变表达层使用）。

        三个来源加权求和：
          - scar_drive: 伤痕基向量第 6 维（expression_drive 维度）的绝对值
          - void_drive: 虚空总压力归一化后乘以权重
          - social_drive: 社交虚空压力归一化后乘以权重
        """
        scar_drive = abs(self.scar_state.base[6]) if len(self.scar_state.base) > 6 else 0.0
        void_drive = min(1.0, self.void_space.total_pressure() / 50.0)
        social_drive = min(1.0, self.social_void.pressure / 3.0)
        return min(
            1.0,
            scar_drive
            + void_drive * self._void_drive_weight
            + social_drive * self._social_drive_weight,
        )

    def _compute_coherence(self) -> float:
        """计算全局一致性：虚空与伤痕的对齐程度。

        r → 1: 虚空和伤痕对齐（系统一致——痛的地方也在回避）
        r → 0: 压力积累在麻木区域（解离状态——回避的不是真正痛的地方）

        这是系统健康度的重要指标：低一致性暗示需要干预。
        """
        if not self.void_space.voids:
            return 1.0
        total_pressure = 0.0
        numbed_pressure = 0.0
        for v in self.void_space.voids:
            total_pressure += v.pressure
            dim_hint = len(v.boundary) % self.scar_state.n_dims
            if self.scar_state.modifier(dim_hint) < 0.5:
                numbed_pressure += v.pressure
        if total_pressure < 0.01:
            return 1.0
        return 1.0 - (numbed_pressure / total_pressure)

    def feedback(self, outcome: str, dt: float = 1.0) -> dict[str, float]:
        """注入表达结果作为反馈。

        'accepted' → 减少虚空压力，正向伤痕输入（温暖、修复）
        'ignored' → 增加虚空深度，负向伤痕输入（退缩）
        'rejected' → 创伤事件注入伤痕状态（伤害）
        """
        self._cached_observe = None
        if outcome == "accepted":
            for v in self.void_space.voids:
                v.pressure *= self._accepted_decay
            feedback_vec = [0.3, 0.0, 0.2, -0.2, 0.1, -0.3, 0.0, 0.0]
        elif outcome == "ignored":
            for v in self.void_space.voids:
                v.depth = min(5.0, v.depth + self._ignored_deepening)
            feedback_vec = [0.0, -0.1, -0.1, 0.2, -0.1, 0.0, -0.3, 0.0]
        elif outcome == "rejected":
            feedback_vec = [-0.3, 0.1, -0.3, 0.3, -0.1, 0.4, -0.2, 0.3]
        else:
            feedback_vec = [0.0] * 8

        self.scar_state.step(feedback_vec, 0.0)
        return self.scar_state.observe()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scar": self.scar_state.to_dict(),
            "void": self.void_space.to_dict(),
            "social_void": self.social_void.to_dict(),
            "coherence": self._coherence,
            "tick": self._tick,
        }

    def diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scar": self.scar_state.observe(),
            "void": self.void_space.diagnostics(),
            "coherence": self._coherence,
            "expression_drive": self.expression_drive(),
            "tick": self._tick,
        }
        # PEL signal surface (additive; absent entirely when PEL is inactive).
        pel = self.scar_state.pel_diagnostics()
        if pel is not None:
            out["pel"] = pel
        return out

    def set_personality_params(
        self,
        coupling_rate: float,
        pressure_threshold: float,
        void_drive_weight: float,
        social_drive_weight: float,
        accepted_decay: float,
        ignored_deepening: float,
    ) -> None:
        self._void_pressure_coupling_rate = coupling_rate
        self.void_space._pressure_threshold = pressure_threshold
        self._void_drive_weight = void_drive_weight
        self._social_drive_weight = social_drive_weight
        self._accepted_decay = accepted_decay
        self._ignored_deepening = ignored_deepening


def _default_similarity(a: bytes, b: bytes) -> float:
    """默认相似度函数：基于 Hamming 距离的二进制向量相似度。"""
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    xor_bits = sum((a[i] ^ b[i]).bit_count() for i in range(min_len))
    total_bits = min_len * 8
    return 1.0 - (xor_bits / total_bits) if total_bits > 0 else 0.0
