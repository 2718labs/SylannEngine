"""会话宿主模块。

SylanneAlphaHost 是每个会话的顶层容器，持有：
- AlphaRuntime: 文件系统持久化运行时（负责 load/save）
- AlphaKernel: 计算核心调度器（负责 tick/surface/snapshot）

宿主对外暴露 on_request / on_response / on_chat / on_proactive_check 四个生命周期方法，
每次调用都会驱动 kernel.tick() 并自动持久化状态。

与其他组件的关系：
- main.py 的 EmotionalStatePlugin 通过 _host(session_key) 获取或创建宿主实例
- 宿主内部将 SylanneAlphaHostEvent 转换为 AlphaKernelEvent 传递给 kernel
- AlphaRuntime 负责将 kernel snapshot 序列化到磁盘
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import BrainComputeConfig
from .brain_errors import BrainDurabilityError
from .brain_store import BrainStateStore
from .kernel import AlphaKernel, AlphaKernelEvent
from .runtime import AlphaRuntime

if TYPE_CHECKING:
    from ..config import DimensionProfile
    from ..telemetry import DistillationSink

_FLUSH_INTERVAL: float = 5.0
_FLUSH_TICK_THRESHOLD: int = 8


@dataclass(slots=True)
class SylanneAlphaHostEvent:
    """宿主层事件数据类。

    是外部调用者传入的事件格式，比 AlphaKernelEvent 少一些内部字段。
    通过 to_kernel_event() 转换为 kernel 可消费的格式。
    """

    event_id: str | None = None
    text: str = ""
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    now: float = 0.0
    values: dict[str, float] = field(default_factory=dict)
    event_time: dict[str, Any] = field(default_factory=dict)

    def to_kernel_event(self) -> AlphaKernelEvent:
        return AlphaKernelEvent(
            event_id=self.event_id,
            text=self.text,
            values=dict(self.values),
            confidence=self.confidence,
            flags=list(self.flags),
            now=self.now,
            event_time=dict(self.event_time),
        )


@dataclass(slots=True)
class SylanneAlphaHost:
    """每个会话的宿主对象。

    负责：
    1. 初始化时从磁盘加载或新建 kernel
    2. 将外部事件转换为 kernel 事件并驱动 tick
    3. 每次 tick 后自动持久化 kernel 状态
    4. 提供 on_chat 的简易对话循环（request → 生成回复 → response）

    持久化策略（CoW）：
    - tick 完成后立即取 snapshot（纯内存操作，微秒级）
    - 按间隔/tick 数阈值决定是否落盘
    - 落盘时写入的是之前缓存的 snapshot，不再访问 kernel 状态
    - 这确保 tick 计算和磁盘 I/O 完全解耦，避免慢盘阻塞计算

    Args:
        root: 持久化根目录路径
        session_key: 会话标识符
        legacy: 可选的旧版 3.x 数据，用于首次迁移
        profile: 计算维度配置（lite/pro/max）
    """

    root: Path | str
    session_key: str = "default"
    legacy: dict[str, Any] | None = None
    profile: DimensionProfile | None = None
    telemetry_sink: DistillationSink | None = None
    pel_enabled: bool = False
    affect_enabled: bool = False
    affect_takeover: bool = False
    affect_slowchannel: bool = False
    affect_plasticity: bool = False
    affect_full_takeover: bool = False
    brain_compute: BrainComputeConfig = field(default_factory=BrainComputeConfig)
    brain_store: BrainStateStore | None = None
    runtime: AlphaRuntime = field(init=False)
    kernel: AlphaKernel = field(init=False)
    _dirty: bool = field(init=False, default=False)
    _ticks_since_flush: int = field(init=False, default=0)
    _last_flush_time: float = field(init=False, default=0.0)
    _pending_snapshot: dict[str, Any] | None = field(init=False, default=None)
    _operation_lock: threading.RLock = field(
        init=False,
        default_factory=threading.RLock,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.brain_compute.enabled and self.brain_store is None:
            raise BrainDurabilityError("brain-enabled hosts require a shared BrainStateStore")
        self.runtime = AlphaRuntime(
            Path(self.root),
            profile=self.profile,
            pel_enabled=self.pel_enabled,
            affect_enabled=self.affect_enabled,
            affect_takeover=self.affect_takeover,
            affect_slowchannel=self.affect_slowchannel,
            affect_plasticity=self.affect_plasticity,
            affect_full_takeover=self.affect_full_takeover,
            brain_compute=self.brain_compute,
            brain_store=self.brain_store,
        )
        self.kernel = self.runtime.load(self.session_key, legacy=self.legacy)
        self.kernel.set_telemetry(self.telemetry_sink)
        self._last_flush_time = time.time()

    def on_request(
        self,
        event: SylanneAlphaHostEvent | dict[str, Any] | None = None,
        assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理用户请求事件，驱动一次 tick 并返回 surface。"""
        return self._tick(event, phase="request", assessment=assessment)

    def on_response(
        self, event: SylanneAlphaHostEvent | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """处理 LLM 回复事件，驱动一次 tick 并返回 surface。"""
        return self._tick(event, phase="response")

    def on_chat(
        self, event: SylanneAlphaHostEvent | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """简易对话循环：request tick → 生成回复文本 → response tick。

        返回包含 reply_text 和双向 surface 的完整对话结果。
        """
        request_surface = self._tick(event, phase="chat_request")
        reply_text = self._reply_text(request_surface)
        response_event = self._event(event)
        response_surface = self._tick(
            SylanneAlphaHostEvent(
                text=reply_text,
                confidence=0.7,
                flags=["chat_response", "safe"],
                now=response_event.now,
                values=dict(response_event.values),
                event_time=dict(response_event.event_time),
            ),
            phase="response",
        )
        return {
            "schema_version": "sylanne.alpha.chat.v1",
            "session_key": self.session_key,
            "ok": True,
            "reply_text": reply_text,
            "action": response_surface["decision"]["action"],
            "request": request_surface,
            "surface": response_surface,
        }

    def on_proactive_check(
        self, event: SylanneAlphaHostEvent | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """主动发言检查：tick 后若 host_payload 指示应发送，则消耗中断预算并进入冷却。"""
        surface = self._tick(event, phase="proactive")
        if surface["host_payload"].get("should_send"):
            self.kernel.body.immunity.interruption_budget = max(
                0.0, self.kernel.body.immunity.interruption_budget - 0.2
            )
            self.kernel.body.immunity.cooldown = max(self.kernel.body.immunity.cooldown, 0.35)
            # Spend the emotional debt that drove this reach-out so it can't re-fire
            # every time cooldown recovers (delayed-talkative guard), keeping the
            # affect-debt clock in sync with the real-time cooldown/budget clocks.
            self.kernel.discharge_affect_debt()
            self._dirty = True
            self._pending_snapshot = self.kernel.snapshot()
            self._flush()
        return surface

    def diagnostics(self) -> dict[str, Any]:
        return self.kernel.surface()

    @property
    def brain_state(self) -> Any | None:
        """Return the authoritative B state for Engine validation, if enabled."""
        return self.kernel.computation.engine.brain_state

    def apply_targeted_feedback(
        self,
        *,
        feedback_id: str,
        target_tick: int,
        value: float,
        confidence: float,
    ) -> dict[str, object]:
        """Serialize one targeted feedback mutation with this host's operations."""
        with self._operation_lock:
            return self.kernel.computation.engine.apply_targeted_feedback(
                feedback_id=feedback_id,
                target_tick=target_tick,
                value=value,
                confidence=confidence,
            )

    def snapshot(self) -> dict[str, Any]:
        return self.kernel.snapshot()

    def _tick(
        self,
        event: SylanneAlphaHostEvent | dict[str, Any] | None,
        *,
        phase: str,
        assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """内部 tick 实现：转换事件 → 注入 phase flag → 驱动 kernel → CoW snapshot → 按需持久化。"""
        with self._operation_lock:
            host_event = self._event(event)
            if self.brain_compute.enabled and not host_event.event_id:
                raise ValueError("brain-enabled host events require a nonempty event_id")
            flags = list(dict.fromkeys([phase, *host_event.flags]))
            surface: dict[str, Any] = self.kernel.tick(
                AlphaKernelEvent(
                    event_id=host_event.event_id,
                    text=host_event.text,
                    values=dict(host_event.values),
                    confidence=host_event.confidence,
                    flags=flags,
                    now=host_event.now,
                    event_time=dict(host_event.event_time),
                ),
                assessment=assessment,
            )["surface"]
            # CoW: take snapshot immediately (pure memory, fast)
            self._pending_snapshot = self.kernel.snapshot()
            self._dirty = True
            self._ticks_since_flush += 1
            self._maybe_flush()
            return surface

    def _maybe_flush(self) -> None:
        """按间隔或 tick 数阈值决定是否落盘。"""
        now = time.time()
        elapsed = now - self._last_flush_time
        if self._ticks_since_flush >= _FLUSH_TICK_THRESHOLD or elapsed >= _FLUSH_INTERVAL:
            self._flush(now)

    def _flush(self, now: float = 0.0) -> None:
        if not self._dirty or self._pending_snapshot is None:
            return
        self.runtime.save_snapshot(self.kernel.session_key, self._pending_snapshot)
        self._pending_snapshot = None
        self._dirty = False
        self._ticks_since_flush = 0
        self._last_flush_time = now or time.time()

    def flush(self) -> None:
        """外部强制落盘入口。"""
        if self._pending_snapshot is None and self._dirty:
            self._pending_snapshot = self.kernel.snapshot()
        self._flush()

    def close(self, *, flush: bool = True) -> None:
        """Optionally persist legacy state, then release brain backend resources."""
        with self._operation_lock:
            if flush:
                self.flush()
            self.kernel.computation.engine.close()

    def _reply_text(self, surface: dict[str, Any]) -> str:
        """根据 decision/guard 生成简短的内置回复文本（on_chat 专用）。"""
        decision = surface["decision"]
        guard = surface["guard"]
        if not guard["allowed"]:
            return "我先退一步。"
        if decision["action"] == "repair":
            return "刚才那一下我会放轻一点。"
        if decision["action"] == "withdraw":
            return "我听到了，先安静一点。"
        if decision["action"] in {"express", "reach_out", "explore"}:
            return "我在听，你继续说。"
        return "嗯，我记下了。"

    def _event(self, event: SylanneAlphaHostEvent | dict[str, Any] | None) -> SylanneAlphaHostEvent:
        if isinstance(event, SylanneAlphaHostEvent):
            return event
        payload = event or {}
        return SylanneAlphaHostEvent(
            event_id=(
                str(payload.get("event_id")) if payload.get("event_id") is not None else None
            ),
            text=str(payload.get("text") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            flags=list(payload.get("flags") or []),
            now=float(payload.get("now") or 0.0),
            values=dict(payload.get("values") or {}),
            event_time=dict(
                payload.get("event_time") if isinstance(payload.get("event_time"), dict) else {}  # type: ignore[arg-type]
            ),
        )
