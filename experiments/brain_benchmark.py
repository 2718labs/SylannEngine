"""Deterministic resource and latency benchmark for v26 brain compute."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_SCHEMA_VERSION = 2
MIN_GATE_SAMPLES = 10_000
MIB = 1024**2
GIB = 1024**3
ENGINE_RSS_LIMIT = 500 * MIB
B_HOT_LIMIT = 16 * 1024
C_HOT_LIMIT = 64 * 1024
# Keep these scalar limits import-free so loading this CLI never pulls optional
# numeric stacks through the package initializer. The real codec constants are
# cross-checked when resource measurement actually runs.
COLD_BLOB_LIMIT = 32 * 1024
COLD_PAYLOAD_LIMIT = 64 * 1024
LATENCY_LIMITS_MS = {
    "compute_fast_ms": 15.0,
    "compute_full_ms": 30.0,
    "sqlite_full_ms": 20.0,
    "engine_e2e_fast_ms": 45.0,
    "engine_e2e_full_ms": 60.0,
}
_ZERO8 = (0.0,) * 8


def _finite_samples(values: Sequence[float]) -> tuple[float, ...]:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("at least one sample is required")
    if any(not math.isfinite(value) or value < 0.0 for value in materialized):
        raise ValueError("samples must be finite and nonnegative")
    return materialized


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(_finite_samples(values))
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    samples = _finite_samples(values)
    return {
        "p50": round(_percentile(samples, 0.50), 9),
        "p95": round(_percentile(samples, 0.95), 9),
        "p99": round(_percentile(samples, 0.99), 9),
    }


def _read_positive_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value in {"", "max"}:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _cpu_affinity_count() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return len(affinity(0))
        except OSError:
            pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE
            get_process_affinity = kernel32.GetProcessAffinityMask
            get_process_affinity.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            get_process_affinity.restype = wintypes.BOOL
            if get_process_affinity(
                get_current_process(),
                ctypes.byref(process_mask),
                ctypes.byref(system_mask),
            ):
                return int(process_mask.value).bit_count()
        except (AttributeError, OSError, ValueError):
            return None
    return None


def _cpu_quota() -> float | None:
    if os.name == "nt":
        return None
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()
        if quota_text != "max":
            parsed_period = int(period_text)
            parsed_quota = int(quota_text)
            if parsed_period > 0 and parsed_quota > 0:
                return parsed_quota / parsed_period
    except (OSError, ValueError):
        pass
    quota = _read_positive_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_positive_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    return quota / period if quota is not None and period is not None else None


def _windows_memory() -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        memory_status = ctypes.windll.kernel32.GlobalMemoryStatusEx
        memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
        memory_status.restype = ctypes.c_int
        if memory_status(ctypes.byref(status)):
            return int(status.total_physical), int(status.available_physical)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def _posix_memory() -> tuple[int, int]:
    try:
        sysconf = cast(Any, os).sysconf
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total = page_size * int(sysconf("SC_PHYS_PAGES"))
        available = page_size * int(sysconf("SC_AVPHYS_PAGES"))
        return max(0, total), max(0, available)
    except (AttributeError, OSError, TypeError, ValueError):
        return 0, 0


def _memory_limit() -> int | None:
    if os.name == "nt":
        return None
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        value = _read_positive_integer(path)
        if value is not None and value < 1 << 60:
            return value
    return None


def _is_target_environment(environment: Mapping[str, object]) -> bool:
    affinity = environment.get("affinity_cpus")
    quota = environment.get("cpu_quota")
    memory_limit = environment.get("memory_limit_bytes")
    return (
        isinstance(affinity, int)
        and not isinstance(affinity, bool)
        and 0 < affinity <= 2
        and isinstance(quota, (int, float))
        and not isinstance(quota, bool)
        and 0.0 < float(quota) <= 2.0
        and isinstance(memory_limit, int)
        and not isinstance(memory_limit, bool)
        and 0 < memory_limit <= 2 * GIB
    )


def detect_environment() -> dict[str, object]:
    memory = _windows_memory()
    physical, available = memory if memory is not None else _posix_memory()
    environment: dict[str, object] = {
        "platform": platform.platform(),
        "logical_cpus": max(1, os.cpu_count() or 1),
        "affinity_cpus": _cpu_affinity_count(),
        "cpu_quota": _cpu_quota(),
        "physical_memory_bytes": physical,
        "available_memory_bytes": available,
        "memory_limit_bytes": _memory_limit(),
        "target_verified": False,
    }
    environment["target_verified"] = _is_target_environment(environment)
    return environment


def process_rss_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                    ("private_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE
            get_process_memory = psapi.GetProcessMemoryInfo
            get_process_memory.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory.restype = wintypes.BOOL
            if get_process_memory(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.working_set_size)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        sysconf = cast(Any, os).sysconf
        return int(statm[1]) * int(sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError):
        pass
    try:
        resource: Any = __import__("resource")
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return 0


def _max_bundle(horizon: int) -> Any:
    from sylanne_core.compute.brain_c import (
        N_EDGES,
        N_NEURONS,
        TOPOLOGY,
        BrainCState,
        CEligibilityRecord,
    )
    from sylanne_core.compute.brain_codec import BrainBundle
    from sylanne_core.compute.brain_state import BEligibilityRecord, BrainState

    lineage = "11111111-1111-4111-8111-111111111111"
    b_records = tuple(
        BEligibilityRecord(tick, float(tick), (tick / 64.0,) * 8) for tick in range(1, horizon + 1)
    )
    c_records = tuple(
        CEligibilityRecord(tick, float(tick), (float(tick % 2),) * N_EDGES)
        for tick in range(1, horizon + 1)
    )
    return BrainBundle(
        BrainState(
            generation=0,
            lineage_id=lineage,
            e=(0.25,) * 8,
            d_plus=(1.0,) * 8,
            d_minus=(2.0,) * 8,
            gain_b=(0.5,) * 8,
            theta_b=(0.25,) * 8,
            clock=float(horizon),
            tick_id=horizon,
            history_epoch=horizon,
            mutation_seq=horizon,
            eligibility_ring=b_records,
            eligibility_horizon=horizon,
            clock_regressions=0,
        ),
        BrainCState(
            v=(0.25,) * N_NEURONS,
            adaptation=(0.5,) * N_NEURONS,
            filtered=(0.75,) * N_NEURONS,
            weights=TOPOLOGY.initial_weights,
            eligibility_ring=c_records,
            eligibility_horizon=horizon,
        ),
    )


def _adversarial_bundle(horizon: int) -> Any:
    """Return a deterministic, valid high-entropy bundle for cold-size gating."""
    from sylanne_core.compute.brain_c import (
        N_AXES,
        N_BASIS,
        N_EDGES,
        N_NEURONS,
        BrainCState,
        CEligibilityRecord,
    )
    from sylanne_core.compute.brain_codec import BrainBundle
    from sylanne_core.compute.brain_state import BEligibilityRecord, BrainState

    rng = random.Random(2718)
    b_records = tuple(
        BEligibilityRecord(tick, float(tick), tuple(rng.random() for _ in range(N_AXES)))
        for tick in range(1, horizon + 1)
    )
    c_records = tuple(
        CEligibilityRecord(
            tick,
            float(tick),
            tuple(8.0 * rng.random() for _ in range(N_EDGES)),
            tuple(rng.random() for _ in range(N_BASIS)),
        )
        for tick in range(1, horizon + 1)
    )
    return BrainBundle(
        BrainState(
            generation=7,
            lineage_id="33333333-3333-4333-8333-333333333333",
            e=tuple(2.0 * rng.random() - 1.0 for _ in range(N_AXES)),
            d_plus=tuple(32.0 * rng.random() for _ in range(N_AXES)),
            d_minus=tuple(32.0 * rng.random() for _ in range(N_AXES)),
            gain_b=tuple(0.05 + 0.95 * rng.random() for _ in range(N_AXES)),
            theta_b=tuple(0.95 * rng.random() for _ in range(N_AXES)),
            clock=float(horizon),
            tick_id=horizon,
            history_epoch=horizon,
            mutation_seq=horizon,
            eligibility_ring=b_records,
            eligibility_horizon=horizon,
            clock_regressions=3,
        ),
        BrainCState(
            v=tuple(4.0 * rng.random() - 2.0 for _ in range(N_NEURONS)),
            adaptation=tuple(rng.random() for _ in range(N_NEURONS)),
            filtered=tuple(rng.random() for _ in range(N_NEURONS)),
            weights=tuple(2.0 * rng.random() - 1.0 for _ in range(N_EDGES)),
            w_out=tuple(2.0 * rng.random() - 1.0 for _ in range(N_BASIS * N_AXES)),
            eligibility_ring=c_records,
            eligibility_horizon=horizon,
        ),
    )


def measure_state_resources(*, session_count: int = 1_000, horizon: int = 32) -> dict[str, int]:
    if isinstance(session_count, bool) or not isinstance(session_count, int) or session_count <= 0:
        raise ValueError("session_count must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 32:
        raise ValueError("horizon must be an integer in [1,32]")
    from sylanne_core.compute.brain_c import BrainCState
    from sylanne_core.compute.brain_codec import (
        MAX_STATE_BLOB_BYTES,
        MAX_STATE_PAYLOAD_BYTES,
        BrainBundle,
        inspect_brain_bundle_encoding,
    )
    from sylanne_core.compute.brain_state import BrainState

    if MAX_STATE_BLOB_BYTES != COLD_BLOB_LIMIT or MAX_STATE_PAYLOAD_BYTES != COLD_PAYLOAD_LIMIT:
        raise RuntimeError("benchmark and codec state-size limits disagree")

    template = _max_bundle(horizon)
    adversarial = _adversarial_bundle(horizon)
    fresh = BrainBundle(
        BrainState.fresh(
            lineage_id="22222222-2222-4222-8222-222222222222",
            feedback_horizon=horizon,
        ),
        BrainCState.fresh(feedback_horizon=horizon),
    )
    fresh_metrics = inspect_brain_bundle_encoding(fresh)
    representative_metrics = inspect_brain_bundle_encoding(template)
    adversarial_metrics = inspect_brain_bundle_encoding(adversarial)
    cold_blob_bytes = max(representative_metrics.blob_bytes, adversarial_metrics.blob_bytes)

    gc.collect()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    b_states = [template.b.copy() for _ in range(session_count)]
    after_b, _ = tracemalloc.get_traced_memory()
    c_states = [template.c.copy() for _ in range(session_count)]
    after_c, peak = tracemalloc.get_traced_memory()
    b_hot = max(0, round((after_b - baseline) / session_count))
    c_hot = max(0, round((after_c - after_b) / session_count))
    total_delta = max(0, after_c - baseline)
    del b_states, c_states
    if not already_tracing:
        tracemalloc.stop()
    return {
        "sessions": session_count,
        "horizon": horizon,
        "b_hot_bytes": b_hot,
        "c_hot_bytes": c_hot,
        "fresh_blob_bytes": fresh_metrics.blob_bytes,
        "representative_cold_payload_bytes": representative_metrics.payload_bytes,
        "representative_cold_payload_headroom_bytes": max(
            0, COLD_PAYLOAD_LIMIT - representative_metrics.payload_bytes
        ),
        "representative_cold_blob_bytes": representative_metrics.blob_bytes,
        "representative_cold_blob_headroom_bytes": max(
            0, COLD_BLOB_LIMIT - representative_metrics.blob_bytes
        ),
        "adversarial_cold_payload_bytes": adversarial_metrics.payload_bytes,
        "adversarial_cold_payload_headroom_bytes": max(
            0, COLD_PAYLOAD_LIMIT - adversarial_metrics.payload_bytes
        ),
        "adversarial_cold_blob_bytes": adversarial_metrics.blob_bytes,
        "adversarial_cold_blob_headroom_bytes": max(
            0, COLD_BLOB_LIMIT - adversarial_metrics.blob_bytes
        ),
        "cold_blob_bytes": cold_blob_bytes,
        "cold_blob_headroom_bytes": max(0, COLD_BLOB_LIMIT - cold_blob_bytes),
        "tracemalloc_delta_bytes": total_delta,
        "tracemalloc_peak_bytes": max(total_delta, peak - baseline),
    }


async def _unused_llm(_system: str, _user: str) -> str:
    return "unused"


def _brain_config(*, hot_session_limit: int) -> Any:
    from sylanne_core.config import BrainComputeConfig, SylanneConfig

    return SylanneConfig(
        assessor_enabled=False,
        brain_compute=BrainComputeConfig(
            enabled=True,
            c_enabled=True,
            sparse_routing=True,
            feedback_horizon=32,
            hot_session_limit=hot_session_limit,
        ),
    )


def _set_route(engine: Any, session_id: str, route: str) -> None:
    spine = engine._hosts[session_id].kernel.computation
    gate = getattr(spine, "_gate", None)
    if gate is None:
        gate = spine.gate
    thresholds = (1.0, 1.0) if route == "fast" else (0.0, 0.0)
    gate.set_route_thresholds(*thresholds)


def _last_route(engine: Any, session_id: str) -> str:
    spine = engine._hosts[session_id].kernel.computation
    diagnostics = spine.diagnostics()
    route = diagnostics.get("last_route")
    if route not in {"fast", "normal", "full"}:
        raise RuntimeError(f"Engine route diagnostics are unavailable: {route!r}")
    return str(route)


async def measure_engine_rss(
    data_dir: str | Path,
    *,
    host_count: int = 48,
    horizon: int = 32,
) -> dict[str, int]:
    if host_count <= 0 or horizon <= 0 or horizon > 32:
        raise ValueError("host_count and horizon must be positive; horizon <= 32")
    from sylanne_core import SylanneEngine

    gc.collect()
    before = process_rss_bytes()
    engine = SylanneEngine(
        Path(data_dir) / "engine-rss",
        _unused_llm,
        config=_brain_config(hot_session_limit=host_count),
    )
    await engine.start()
    maximum = 0
    try:
        for session_index in range(host_count):
            session_id = f"rss-{session_index:04d}"
            await engine.process(
                session_id,
                "",
                confidence=0.0,
                flags=[],
                now=1.0,
                event_id=f"rss-{session_index}-1",
            )
            _set_route(engine, session_id, "fast")
            maximum = max(maximum, len(engine._hosts))
        for tick in range(2, horizon + 1):
            for session_index in range(host_count):
                session_id = f"rss-{session_index:04d}"
                await engine.process(
                    session_id,
                    "",
                    confidence=0.0,
                    flags=[],
                    now=float(tick),
                    event_id=f"rss-{session_index}-{tick}",
                )
                maximum = max(maximum, len(engine._hosts))
        after = process_rss_bytes()
        return {
            "host_count": host_count,
            "horizon": horizon,
            "engine_48_host_rss_bytes": max(0, after - before),
            "max_hosts": maximum,
            "resident_hosts": len(engine._hosts),
        }
    finally:
        await engine.shutdown()


def _appraisal(index: int) -> tuple[float, ...]:
    return tuple(math.sin((index + 1) * (axis + 1) * 0.017) for axis in range(8))


def _measure_compute(*, route: str, samples: int, warmup_samples: int) -> list[float]:
    from sylanne_core.compute.brain_c import BrainCState, evolve_c_event
    from sylanne_core.compute.brain_compute import BrainComputeCore, BrainEvent
    from sylanne_core.compute.brain_state import EventAllocation

    brain = BrainComputeCore.fresh(
        lineage_id=(
            "33333333-3333-4333-8333-333333333333"
            if route == "fast"
            else "44444444-4444-4444-8444-444444444444"
        ),
        feedback_horizon=32,
    )
    c_state = BrainCState.fresh(feedback_horizon=32)
    retained: list[float] = []
    for index in range(samples + warmup_samples):
        appraisal = _appraisal(index)
        state = brain.state
        start = time.perf_counter_ns()
        c_candidate = evolve_c_event(
            c_state,
            appraisal,
            route=route,
            tick_id=state.tick_id + 1,
            created_at=float(state.tick_id + 1),
            delta_t=1.0,
        )
        event = BrainEvent(
            event_id=f"compute-{route}-{index}",
            assessment=appraisal,
            hdc=appraisal,
            wound_sum=_ZERO8,
            surprise=0.5,
            perception_acuity=1.0,
            proposal_c=c_candidate.proposal,
        )
        allocation = EventAllocation(
            generation=state.generation,
            lineage_id=state.lineage_id,
            tick_id=state.tick_id + 1,
            history_epoch=state.history_epoch + 1,
            mutation_seq=state.mutation_seq + 1,
        )
        candidate = brain.prepare_event(
            event,
            allocation=allocation,
            trusted_now=float(state.tick_id + 1),
            alpha_c=0.0,
        )
        brain.commit(candidate)
        c_state = c_candidate.state
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        if index >= warmup_samples:
            retained.append(elapsed)
    return retained


def _evolve_allocated(allocated: Any, *, event_id: str, index: int) -> tuple[Any, Any]:
    from sylanne_core.compute.brain_c import evolve_c_event
    from sylanne_core.compute.brain_codec import BrainBundle
    from sylanne_core.compute.brain_compute import BrainComputeCore, BrainEvent
    from sylanne_core.compute.brain_store import StoredReceipt

    old = allocated.bundle
    allocation = allocated.allocation
    appraisal = _appraisal(index)
    created_at = old.b.clock + 1.0
    c_candidate = evolve_c_event(
        old.c,
        appraisal,
        route="full",
        tick_id=allocation.tick_id,
        created_at=created_at,
        delta_t=1.0,
    )
    core = BrainComputeCore(old.b)
    event = BrainEvent(
        event_id=event_id,
        assessment=appraisal,
        hdc=appraisal,
        wound_sum=_ZERO8,
        surprise=0.5,
        perception_acuity=1.0,
        proposal_c=c_candidate.proposal,
    )
    candidate = core.prepare_event(
        event,
        allocation=allocation,
        trusted_now=created_at,
        alpha_c=0.0,
    )
    next_b = core.commit(candidate)
    bundle = BrainBundle(next_b, c_candidate.state)
    receipt = StoredReceipt(
        kind="event",
        status="applied",
        generation=next_b.generation,
        tick_id=next_b.tick_id,
        history_epoch=next_b.history_epoch,
        mutation_seq=next_b.mutation_seq,
    )
    return bundle, receipt


def _measure_store(
    data_dir: Path,
    *,
    samples: int,
    warmup_samples: int,
) -> tuple[list[float], list[float]]:
    from sylanne_core.compute.brain_store import (
        BrainStateStore,
        EventAllocated,
        EventCommit,
        event_id_digest,
        session_digest,
    )

    store = BrainStateStore.start(data_dir, feedback_horizon=32)
    session_key = session_digest("benchmark-store")
    commits: list[float] = []
    loads: list[float] = []
    try:
        for index in range(samples + warmup_samples):
            event_id = f"store-{index}"
            event_key = event_id_digest(event_id)
            allocated = store.preflight_allocate(session_key, event_key)
            if not isinstance(allocated, EventAllocated):
                raise RuntimeError("benchmark event unexpectedly duplicated")
            bundle, receipt = _evolve_allocated(allocated, event_id=event_id, index=index)
            commit = EventCommit(allocated=allocated, bundle=bundle, receipt=receipt)
            start = time.perf_counter_ns()
            store.commit_event(session_key, event_key, commit)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            if index >= warmup_samples:
                commits.append(elapsed)
        for _ in range(samples + warmup_samples):
            start = time.perf_counter_ns()
            store.load(session_key)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            if len(loads) >= warmup_samples:
                loads.append(elapsed)
            else:
                loads.append(elapsed)
        loads = loads[warmup_samples:]
        return commits, loads
    finally:
        store.close()


async def _measure_engine_mode(
    data_dir: Path,
    *,
    route: str,
    samples: int,
    warmup_samples: int,
    host_count: int,
    producers: int,
) -> tuple[list[float], dict[str, float | int], dict[str, int]]:
    from sylanne_core import SylanneEngine

    engine = SylanneEngine(
        data_dir,
        _unused_llm,
        config=_brain_config(hot_session_limit=host_count),
    )
    await engine.start()
    try:
        # PredictiveCodingGate.route() caps routing at "normal" for the first 15
        # events per host (cold-start guard: the predictor is uncalibrated on a
        # short surprise history). The full route therefore cannot be measured
        # until each host's gate is primed, so send enough non-blank warmup events
        # per host to clear that guard BEFORE pinning the requested route
        # thresholds. Fast returns before the guard, but priming it too keeps the
        # measured gate warm and realistic.
        route_warmup_events = 16
        for session_index in range(host_count):
            session_id = f"{route}-{session_index:04d}"
            initial_text = "fast route warmup" if route == "fast" else "full route warmup"
            for warm_index in range(route_warmup_events):
                await engine.process(
                    session_id,
                    initial_text,
                    confidence=0.5,
                    flags=[] if route == "fast" else ["hurt", "boundary"],
                    now=1.0 + warm_index,
                    event_id=f"{route}-{session_index}-warmup-{warm_index}",
                )
            _set_route(engine, session_id, route)

        total = samples + warmup_samples
        queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
        enqueued_at: dict[int, int] = {}
        for index in range(total):
            timestamp = time.perf_counter_ns()
            enqueued_at[index] = timestamp
            queue.put_nowait((index, timestamp))
        maximum_depth = queue.qsize()
        for _ in range(producers):
            queue.put_nowait(None)
        retained: list[float] = []
        waits: list[float] = []
        maximum_hosts = len(engine._hosts)
        overflow_waits = 0

        async def worker() -> None:
            nonlocal maximum_hosts, overflow_waits
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    index, queued_ns = item
                    waits.append((time.perf_counter_ns() - queued_ns) / 1_000_000.0)
                    session_id = f"{route}-{index % host_count:04d}"
                    if session_id not in engine._hosts and len(engine._hosts) >= host_count:
                        overflow_waits += 1
                    # Fast route must send NON-blank text: blank input hits the
                    # spine's is_blank "skip" early-return (neutral event, no route
                    # label), leaving diagnostics().last_route at its 'resonance'
                    # sentinel. A low-surprise non-blank event exercises the real
                    # sparse fast route so last_route == "fast" (thresholds forced
                    # by _set_route make the tier deterministic regardless of text).
                    text = "fast route event" if route == "fast" else f"full route event {index}"
                    start = time.perf_counter_ns()
                    await engine.process(
                        session_id,
                        text,
                        confidence=0.5,
                        flags=[] if route == "fast" else ["hurt", "boundary"],
                        now=float(index + 2),
                        event_id=f"{route}-event-{index}",
                    )
                    elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
                    actual_route = _last_route(engine, session_id)
                    if actual_route != route:
                        raise RuntimeError(
                            f"requested diagnostic route {route!r}, got {actual_route!r}"
                        )
                    if index >= warmup_samples:
                        retained.append(elapsed)
                    maximum_hosts = max(maximum_hosts, len(engine._hosts))
                finally:
                    queue.task_done()

        started = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(producers)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        await queue.join()
        duration = max(time.perf_counter() - started, 1e-9)
        throughput = {
            "producers": producers,
            "operations_per_second": samples / duration,
            "duration_seconds": duration,
        }
        overflow = {"max_hosts": maximum_hosts, "wait_count": overflow_waits}
        return (
            retained,
            {
                **throughput,
                "max_depth": maximum_depth,
                "wait_p99_ms": percentile_summary(waits)["p99"],
            },
            overflow,
        )
    finally:
        await engine.shutdown()


def _number_at(document: Mapping[str, object], *path: str) -> float:
    current: object = document
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"benchmark report is missing {'.'.join(path)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"benchmark report field {'.'.join(path)} must be numeric")
    converted = float(current)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"benchmark report field {'.'.join(path)} must be finite")
    return converted


def _integer_at(document: Mapping[str, object], *path: str) -> int:
    current: object = document
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"benchmark report is missing {'.'.join(path)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise ValueError(
            f"benchmark report field {'.'.join(path)} must be an exact nonnegative integer"
        )
    return current


def evaluate_gates(report: Mapping[str, object]) -> dict[str, object]:
    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("benchmark report requires environment")
    target_verified = _is_target_environment(environment)
    environment["target_verified"] = target_verified
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        return {
            "target": "2c2g",
            "status": "unverified",
            "failures": [f"schema_version must equal {REPORT_SCHEMA_VERSION}"],
        }
    verification_failures: list[str] = []
    budget_failures: list[str] = []
    samples = _integer_at(report, "samples")
    if samples < MIN_GATE_SAMPLES:
        verification_failures.append("samples must be at least 10000")
    warmup_samples = _integer_at(report, "warmup_samples")
    if warmup_samples < 128:
        verification_failures.append("warmup_samples must be at least 128")
    if report.get("assessor") != "disabled":
        verification_failures.append("assessor must be disabled")
    if report.get("backend") != "builtin":
        verification_failures.append("backend must be builtin")
    producers = _integer_at(report, "throughput", "producers")
    if producers != 2:
        verification_failures.append("throughput.producers must equal 2")
    if _number_at(report, "throughput", "operations_per_second") <= 0.0:
        verification_failures.append("throughput.operations_per_second must be positive")
    _integer_at(report, "queue", "max_depth")
    _number_at(report, "queue", "wait_p99_ms")
    if _integer_at(report, "overflow", "max_hosts") > 56:
        budget_failures.append("overflow.max_hosts exceeds 56")
    _integer_at(report, "overflow", "wait_count")

    for field, limit in LATENCY_LIMITS_MS.items():
        p99 = _number_at(report, field, "p99")
        if p99 <= 0.0:
            verification_failures.append(f"{field}.p99 must be positive")
        elif p99 > limit:
            budget_failures.append(f"{field}.p99 exceeds {limit:g}ms")
    if _number_at(report, "cold_load_ms", "p99") <= 0.0:
        verification_failures.append("cold_load_ms.p99 must be positive")
    if _integer_at(report, "memory", "engine_hot_hosts") != 48:
        verification_failures.append("memory.engine_hot_hosts must equal 48")
    if _integer_at(report, "memory", "tracemalloc_sessions") != 1_000:
        verification_failures.append("memory.tracemalloc_sessions must equal 1000")
    engine_rss = _integer_at(report, "memory", "engine_48_host_rss_bytes")
    if engine_rss <= 0:
        verification_failures.append("memory.engine_48_host_rss_bytes must be positive")
    elif engine_rss > ENGINE_RSS_LIMIT:
        budget_failures.append("memory.engine_48_host_rss_bytes exceeds 500MiB")
    b_hot = _integer_at(report, "memory", "b_hot_bytes")
    if b_hot <= 0:
        verification_failures.append("memory.b_hot_bytes must be positive")
    elif b_hot >= B_HOT_LIMIT:
        budget_failures.append("memory.b_hot_bytes is not below 16KiB")
    c_hot = _integer_at(report, "memory", "c_hot_bytes")
    if c_hot <= 0:
        verification_failures.append("memory.c_hot_bytes must be positive")
    elif c_hot >= C_HOT_LIMIT:
        budget_failures.append("memory.c_hot_bytes is not below 64KiB")
    profile_blobs: list[int] = []
    for profile in ("representative", "adversarial"):
        prefix = f"{profile}_cold"
        payload = _integer_at(report, "memory", f"{prefix}_payload_bytes")
        payload_headroom = _integer_at(report, "memory", f"{prefix}_payload_headroom_bytes")
        blob = _integer_at(report, "memory", f"{prefix}_blob_bytes")
        blob_headroom = _integer_at(report, "memory", f"{prefix}_blob_headroom_bytes")
        if payload <= 0:
            verification_failures.append(f"memory.{prefix}_payload_bytes must be positive")
        if blob <= 0:
            verification_failures.append(f"memory.{prefix}_blob_bytes must be positive")
        if payload_headroom != max(0.0, COLD_PAYLOAD_LIMIT - payload):
            verification_failures.append(f"memory.{prefix}_payload_headroom_bytes is inconsistent")
        if blob_headroom != max(0.0, COLD_BLOB_LIMIT - blob):
            verification_failures.append(f"memory.{prefix}_blob_headroom_bytes is inconsistent")
        if payload >= COLD_PAYLOAD_LIMIT:
            budget_failures.append(f"memory.{prefix}_payload_bytes is not below 64KiB")
        if blob >= COLD_BLOB_LIMIT:
            budget_failures.append(f"memory.{prefix}_blob_bytes is not below 32KiB")
        profile_blobs.append(blob)

    cold_blob = _integer_at(report, "memory", "cold_blob_bytes")
    cold_blob_headroom = _integer_at(report, "memory", "cold_blob_headroom_bytes")
    if cold_blob != max(profile_blobs):
        verification_failures.append("memory.cold_blob_bytes is inconsistent with cold profiles")
    if cold_blob_headroom != max(0.0, COLD_BLOB_LIMIT - cold_blob):
        verification_failures.append("memory.cold_blob_headroom_bytes is inconsistent")
    fresh_blob = _integer_at(report, "memory", "fresh_blob_bytes")
    if fresh_blob <= 0:
        verification_failures.append("memory.fresh_blob_bytes must be positive")
    elif fresh_blob >= COLD_BLOB_LIMIT:
        budget_failures.append("memory.fresh_blob_bytes is not below 32KiB")

    failures = verification_failures + budget_failures
    if not target_verified or verification_failures:
        status = "unverified"
    elif budget_failures:
        status = "failed"
    else:
        status = "pass"
    return {"target": "2c2g", "status": status, "failures": failures}


def _run_benchmark_in_directory(
    root: Path,
    *,
    samples: int,
    warmup_samples: int,
    host_count: int,
    producers: int,
) -> dict[str, object]:
    state_resources = measure_state_resources(session_count=1_000, horizon=32)
    compute_fast = _measure_compute(route="fast", samples=samples, warmup_samples=warmup_samples)
    compute_full = _measure_compute(route="full", samples=samples, warmup_samples=warmup_samples)
    sqlite_full, cold_load = _measure_store(
        root / "store",
        samples=samples,
        warmup_samples=warmup_samples,
    )
    engine_fast, fast_profile, fast_overflow = asyncio.run(
        _measure_engine_mode(
            root / "engine-fast",
            route="fast",
            samples=samples,
            warmup_samples=warmup_samples,
            host_count=host_count,
            producers=producers,
        )
    )
    engine_full, full_profile, full_overflow = asyncio.run(
        _measure_engine_mode(
            root / "engine-full",
            route="full",
            samples=samples,
            warmup_samples=warmup_samples,
            host_count=host_count,
            producers=producers,
        )
    )
    rss = asyncio.run(measure_engine_rss(root, host_count=host_count, horizon=32))
    throughput = {
        "producers": producers,
        "operations_per_second": float(full_profile["operations_per_second"]),
    }
    queue = {
        "max_depth": max(int(fast_profile["max_depth"]), int(full_profile["max_depth"])),
        "wait_p99_ms": max(
            float(fast_profile["wait_p99_ms"]),
            float(full_profile["wait_p99_ms"]),
        ),
    }
    overflow = {
        "max_hosts": max(fast_overflow["max_hosts"], full_overflow["max_hosts"]),
        "wait_count": fast_overflow["wait_count"] + full_overflow["wait_count"],
    }
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "samples": samples,
        "warmup_samples": warmup_samples,
        "assessor": "disabled",
        "backend": "builtin",
        "route_control": "diagnostic_threshold_override_with_route_confirmation",
        "compute_fast_ms": percentile_summary(compute_fast),
        "compute_full_ms": percentile_summary(compute_full),
        "sqlite_full_ms": percentile_summary(sqlite_full),
        "engine_e2e_fast_ms": percentile_summary(engine_fast),
        "engine_e2e_full_ms": percentile_summary(engine_full),
        "cold_load_ms": percentile_summary(cold_load),
        "cold_load_kind": "store_session_restore",
        "environment": detect_environment(),
        "throughput": throughput,
        "queue": queue,
        "overflow": overflow,
        "memory": {
            "engine_48_host_rss_bytes": rss["engine_48_host_rss_bytes"],
            "engine_hot_hosts": rss["resident_hosts"],
            "b_hot_bytes": state_resources["b_hot_bytes"],
            "c_hot_bytes": state_resources["c_hot_bytes"],
            "fresh_blob_bytes": state_resources["fresh_blob_bytes"],
            "representative_cold_payload_bytes": state_resources[
                "representative_cold_payload_bytes"
            ],
            "representative_cold_payload_headroom_bytes": state_resources[
                "representative_cold_payload_headroom_bytes"
            ],
            "representative_cold_blob_bytes": state_resources["representative_cold_blob_bytes"],
            "representative_cold_blob_headroom_bytes": state_resources[
                "representative_cold_blob_headroom_bytes"
            ],
            "adversarial_cold_payload_bytes": state_resources["adversarial_cold_payload_bytes"],
            "adversarial_cold_payload_headroom_bytes": state_resources[
                "adversarial_cold_payload_headroom_bytes"
            ],
            "adversarial_cold_blob_bytes": state_resources["adversarial_cold_blob_bytes"],
            "adversarial_cold_blob_headroom_bytes": state_resources[
                "adversarial_cold_blob_headroom_bytes"
            ],
            "cold_blob_bytes": state_resources["cold_blob_bytes"],
            "cold_blob_headroom_bytes": state_resources["cold_blob_headroom_bytes"],
            "tracemalloc_sessions": state_resources["sessions"],
        },
    }
    report["gates"] = evaluate_gates(report)
    return report


def run_benchmark(
    *,
    samples: int,
    warmup_samples: int = 128,
    host_count: int = 48,
    producers: int = 2,
    data_dir: str | Path | None = None,
) -> dict[str, object]:
    for name, value, minimum in (
        ("samples", samples, 1),
        ("warmup_samples", warmup_samples, 0),
        ("host_count", host_count, 1),
        ("producers", producers, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if data_dir is None:
        with tempfile.TemporaryDirectory(prefix="sylanne-brain-benchmark-") as temporary:
            return _run_benchmark_in_directory(
                Path(temporary),
                samples=samples,
                warmup_samples=warmup_samples,
                host_count=host_count,
                producers=producers,
            )
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return _run_benchmark_in_directory(
        root,
        samples=samples,
        warmup_samples=warmup_samples,
        host_count=host_count,
        producers=producers,
    )


def write_report(report: Mapping[str, object], output_path: str | Path | None) -> None:
    if output_path is None:
        raise ValueError("an explicit output path is required")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(report),
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=MIN_GATE_SAMPLES)
    parser.add_argument("--warmup-samples", type=int, default=128)
    parser.add_argument("--hosts", type=int, default=48)
    parser.add_argument("--producers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_benchmark(
        samples=arguments.samples,
        warmup_samples=arguments.warmup_samples,
        host_count=arguments.hosts,
        producers=arguments.producers,
        data_dir=arguments.data_dir,
    )
    write_report(report, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
