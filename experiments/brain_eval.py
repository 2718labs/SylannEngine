"""Deterministic offline evaluation for the v26 brain-compute path.

The harness consumes opt-in numeric JSONL only. It never reads dialogue text and
never substitutes synthetic measurements for a real-corpus promotion result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sylanne_core.compute import brain_c  # noqa: E402
from sylanne_core.compute.brain_compute import BrainComputeCore, BrainEvent  # noqa: E402
from sylanne_core.compute.brain_state import (  # noqa: E402
    BrainState,
    EventAllocation,
    FeedbackAllocation,
)

CORPUS_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 2
EVALUATOR_SEMANTICS_VERSION = 2
N_AXES = 8
MIN_SESSIONS = 30
MIN_TARGET_TICKS = 1_000
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
FIXED_SPLIT_SEEDS = (2718, 31415, 16180, 57721, 14142)
ADAPTER_NAMES = (
    "current_v26",
    "ema_arx",
    "pel",
    "b_only",
    "b_plus_c",
    "feedback_shuffled",
    "eligibility_disabled",
    "matched_compute_continuous",
)
_CORPUS_FIELDS = frozenset(
    {
        "schema_version",
        "session_id",
        "tick_id",
        "features",
        "target",
        "feedback_target_tick",
        "feedback_value",
    }
)
_ZERO8 = (0.0,) * N_AXES
_FEEDBACK_HORIZON = 32
_FEEDBACK_TTL_SECONDS = 7_200.0


@dataclass(frozen=True, slots=True)
class NumericRecord:
    schema_version: int
    session_id: str
    tick_id: int
    features: tuple[float, ...]
    target: tuple[float, ...]
    feedback_target_tick: int
    feedback_value: float

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CORPUS_SCHEMA_VERSION}")
        _identifier(self.session_id)
        _positive_int("tick_id", self.tick_id)
        _axes("features", self.features)
        _axes("target", self.target)
        if (
            isinstance(self.feedback_target_tick, bool)
            or not isinstance(self.feedback_target_tick, int)
            or not 0 <= self.feedback_target_tick <= self.tick_id
        ):
            raise ValueError("feedback_target_tick must be an integer in [0,tick_id]")
        _bounded_float("feedback_value", self.feedback_value)


@dataclass(frozen=True, slots=True)
class SessionSplit:
    seed: int
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
        raise ValueError("session_id must be nonempty UTF-8 capped at 256 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("session_id must not contain control characters")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if not -1.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be in [-1,1]")
    return converted


def _axes(name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != N_AXES:
        raise ValueError(f"{name} must contain exactly {N_AXES} numeric axes")
    return tuple(_bounded_float(f"{name}[{index}]", value) for index, value in enumerate(values))


def _json_constant(value: str) -> None:
    raise ValueError(f"JSON numeric value {value} must be finite")


def _record_from_document(document: object, *, line_number: int) -> NumericRecord:
    if not isinstance(document, dict) or set(document) != _CORPUS_FIELDS:
        raise ValueError(f"line {line_number}: record fields do not match schema v1")
    version = document["schema_version"]
    tick = document["tick_id"]
    feedback_tick = document["feedback_target_tick"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"line {line_number}: schema_version must be an integer")
    return NumericRecord(
        schema_version=version,
        session_id=_identifier(document["session_id"]),
        tick_id=_positive_int("tick_id", tick),
        features=_axes("features", document["features"]),
        target=_axes("target", document["target"]),
        feedback_target_tick=(
            feedback_tick
            if isinstance(feedback_tick, int) and not isinstance(feedback_tick, bool)
            else -1
        ),
        feedback_value=_bounded_float("feedback_value", document["feedback_value"]),
    )


def load_corpus(path: str | Path) -> list[NumericRecord]:
    corpus_path = Path(path)
    records: list[NumericRecord] = []
    seen: set[tuple[str, int]] = set()
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                document = json.loads(raw_line, parse_constant=_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"line {line_number}: invalid finite JSON: {error}") from error
            record = _record_from_document(document, line_number=line_number)
            key = (record.session_id, record.tick_id)
            if key in seen:
                raise ValueError(f"line {line_number}: duplicate session/tick")
            seen.add(key)
            records.append(record)
    return sorted(records, key=lambda item: (item.session_id, item.tick_id))


def session_block_splits(
    records: Sequence[NumericRecord] | Iterable[NumericRecord],
    *,
    seeds: Sequence[int] = FIXED_SPLIT_SEEDS,
    test_fraction: float = 0.2,
) -> tuple[SessionSplit, ...]:
    materialized = tuple(records)
    sessions = sorted({record.session_id for record in materialized})
    if len(sessions) < 2:
        raise ValueError("session-blocked splitting requires at least two sessions")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    fold_count = round(1.0 / test_fraction)
    if fold_count < 2 or not math.isclose(
        test_fraction,
        1.0 / fold_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("test_fraction must be the reciprocal of an integer fold count")
    validated_seeds = tuple(seeds)
    if len(validated_seeds) != fold_count:
        raise ValueError("one deterministic seed is required for every fold")
    if len(sessions) < fold_count:
        raise ValueError("session-blocked splitting requires at least one session per fold")
    for seed in validated_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("split seeds must be non-boolean integers")

    base_size, extra = divmod(len(sessions), fold_count)
    fold_sizes = [base_size + (1 if index < extra else 0) for index in range(fold_count)]
    remaining = set(sessions)
    splits: list[SessionSplit] = []
    for seed, fold_size in zip(validated_seeds, fold_sizes, strict=True):
        shuffled = sorted(remaining)
        random.Random(seed).shuffle(shuffled)
        test = tuple(sorted(shuffled[:fold_size]))
        remaining.difference_update(test)
        train = tuple(sorted(set(sessions) - set(test)))
        splits.append(SessionSplit(seed=int(seed), train_sessions=train, test_sessions=test))
    if remaining:  # pragma: no cover - fold_sizes partition every session
        raise AssertionError("exhaustive fold construction left sessions unassigned")
    return tuple(splits)


def paired_session_differences(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> tuple[float, ...]:
    if set(baseline) != set(candidate):
        raise ValueError("paired MAE requires the same sessions")
    return tuple(float(baseline[key]) - float(candidate[key]) for key in sorted(baseline))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def bootstrap_ci(
    differences: Sequence[float] | Iterable[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 2718,
) -> tuple[float, float]:
    values = tuple(float(value) for value in differences)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap differences must be nonempty and finite")
    if resamples < DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("resamples must be at least 10000")
    generator = random.Random(seed)
    size = len(values)
    means = [
        fmean(values[generator.randrange(size)] for _ in range(size)) for _ in range(resamples)
    ]
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def promotion_decision(
    *,
    baseline_mae: float,
    candidate_mae: float,
    improvement_ci: tuple[float, float],
    sessions: int,
    target_ticks: int,
) -> dict[str, object]:
    if sessions < MIN_SESSIONS or target_ticks < MIN_TARGET_TICKS:
        return {
            "status": "refused",
            "reason": "insufficient_data",
            "absolute_lower_bound": None,
            "relative_lower_bound": None,
        }
    values = (baseline_mae, candidate_mae, *improvement_ci)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return {
            "status": "refused",
            "reason": "invalid_metrics",
            "absolute_lower_bound": None,
            "relative_lower_bound": None,
        }
    lower = improvement_ci[0]
    relative = lower / baseline_mae if baseline_mae > 0.0 else 0.0
    passed = candidate_mae < baseline_mae and lower > 0.02 and relative > 0.05
    return {
        "status": "pass" if passed else "refused",
        "reason": "thresholds_met" if passed else "promotion_threshold_not_met",
        "absolute_lower_bound": lower,
        "relative_lower_bound": relative,
    }


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _surprise(features: Sequence[float], previous: Sequence[float]) -> float:
    return min(1.0, fmean(abs(features[index] - previous[index]) for index in range(N_AXES)))


def _group_records(records: Iterable[NumericRecord]) -> dict[str, list[NumericRecord]]:
    grouped: dict[str, list[NumericRecord]] = defaultdict(list)
    for record in records:
        grouped[record.session_id].append(record)
    for session_records in grouped.values():
        session_records.sort(key=lambda item: item.tick_id)
    return dict(grouped)


def _shuffled_feedback_targets(records: Sequence[NumericRecord], seed: int) -> list[int]:
    """Choose deterministic, valid wrong targets from events seen by each arrival."""
    generator = random.Random(seed)
    available: list[int] = []
    shuffled: list[int] = []
    for record in records:
        available.append(record.tick_id)
        if len(available) > _FEEDBACK_HORIZON:
            del available[:-_FEEDBACK_HORIZON]
        alternatives = [tick for tick in available if tick != record.feedback_target_tick]
        shuffled.append(
            alternatives[generator.randrange(len(alternatives))]
            if alternatives
            else record.feedback_target_tick
        )
    return shuffled


def _without_b_eligibility(state: BrainState) -> BrainState:
    return BrainState(
        generation=state.generation,
        lineage_id=state.lineage_id,
        e=state.e,
        d_plus=state.d_plus,
        d_minus=state.d_minus,
        gain_b=state.gain_b,
        theta_b=state.theta_b,
        clock=state.clock,
        tick_id=state.tick_id,
        history_epoch=state.history_epoch,
        mutation_seq=state.mutation_seq,
        eligibility_ring=(),
        eligibility_horizon=state.eligibility_horizon,
        clock_regressions=state.clock_regressions,
    )


def _without_c_eligibility(state: brain_c.BrainCState) -> brain_c.BrainCState:
    return brain_c.BrainCState(
        v=state.v,
        adaptation=state.adaptation,
        filtered=state.filtered,
        weights=state.weights,
        w_out=state.w_out,
        eligibility_ring=(),
        eligibility_horizon=state.eligibility_horizon,
    )


def _feedback_allocation(
    state: BrainState,
    *,
    target_tick: int,
    value: float,
    confidence: float,
) -> FeedbackAllocation | None:
    eligible = any(record.tick_id == target_tick for record in state.eligibility_records)
    if not eligible or value == 0.0 or confidence == 0.0:
        return None
    return FeedbackAllocation(
        generation=state.generation,
        lineage_id=state.lineage_id,
        target_tick=target_tick,
        expected_mutation_seq=state.mutation_seq,
        next_mutation_seq=state.mutation_seq + 1,
    )


def _predict_session(
    records: Sequence[NumericRecord],
    *,
    model_name: str,
    parameter: float,
    seed: int,
) -> list[tuple[float, ...]]:
    predictions: list[tuple[float, ...]] = []
    previous = [0.0] * N_AXES
    previous_features = [0.0] * N_AXES
    continuous = [0.0] * N_AXES
    pending_continuous_feedback = 0.0
    feedback_targets = (
        _shuffled_feedback_targets(records, seed)
        if model_name == "feedback_shuffled"
        else [record.feedback_target_tick for record in records]
    )
    tick_map: dict[int, int] = {}
    pel = None
    brain: BrainComputeCore | None = None
    c_state: brain_c.BrainCState | None = None

    if model_name == "pel":
        from sylanne_core.compute.pel_core import PELCore

        pel = PELCore.from_personality({})
    if model_name in {"b_only", "b_plus_c", "feedback_shuffled", "eligibility_disabled"}:
        lineage = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sylanne-eval:{records[0].session_id}"))
        brain = BrainComputeCore.fresh(
            lineage_id=lineage,
            feedback_horizon=_FEEDBACK_HORIZON,
        )
        if model_name != "b_only":
            c_state = brain_c.BrainCState.fresh(feedback_horizon=_FEEDBACK_HORIZON)

    for index, record in enumerate(records):
        if model_name == "current_v26":
            predicted = tuple(record.features)
        elif model_name == "ema_arx":
            alpha = 0.1 + 0.8 * parameter
            trend = 0.25 * parameter
            predicted = tuple(
                _clip(
                    alpha * record.features[axis]
                    + (1.0 - alpha) * previous[axis]
                    + trend * (record.features[axis] - previous_features[axis])
                )
                for axis in range(N_AXES)
            )
        elif model_name == "pel":
            assert pel is not None
            output, _free_energy = pel.step(
                list(record.features),
                _surprise(record.features, previous),
                a_vec=list(record.features),
                confidence=parameter,
            )
            predicted = tuple(_clip(value) for value in output)
        elif model_name in {
            "b_only",
            "b_plus_c",
            "feedback_shuffled",
            "eligibility_disabled",
        }:
            assert brain is not None
            state = brain.state
            appraisal = tuple(record.features)
            proposal = _ZERO8
            if c_state is not None:
                c_candidate = brain_c.evolve_c_event(
                    c_state,
                    appraisal,
                    route="full",
                    tick_id=state.tick_id + 1,
                    created_at=float(state.tick_id + 1),
                    delta_t=1.0,
                )
                c_state = c_candidate.state
                proposal = c_candidate.proposal
            event = BrainEvent(
                event_id=f"{record.session_id}:{record.tick_id}",
                assessment=appraisal,
                hdc=appraisal,
                wound_sum=_ZERO8,
                surprise=_surprise(appraisal, state.e),
                perception_acuity=1.0,
                proposal_c=proposal,
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
                alpha_c=(0.0 if model_name == "b_only" else min(0.1, 0.1 * parameter)),
            )
            committed = brain.commit(candidate)
            predicted = tuple(committed.e)
            tick_map[record.tick_id] = committed.tick_id

            if model_name == "eligibility_disabled":
                brain = BrainComputeCore(_without_b_eligibility(committed))
                if c_state is not None:
                    c_state = _without_c_eligibility(c_state)

            feedback_target = tick_map.get(feedback_targets[index], 0)
            confidence = max(0.0, min(1.0, float(parameter)))
            feedback_state = brain.state
            b_feedback = brain.prepare_feedback(
                target_tick=feedback_target,
                value=record.feedback_value,
                confidence=confidence,
                trusted_now=feedback_state.clock,
                feedback_ttl_seconds=_FEEDBACK_TTL_SECONDS,
                allocation=_feedback_allocation(
                    feedback_state,
                    target_tick=feedback_target,
                    value=record.feedback_value,
                    confidence=confidence,
                ),
            )
            brain.commit(b_feedback)
            if c_state is not None:
                c_feedback = brain_c.evolve_c_feedback(
                    c_state,
                    target_tick=feedback_target,
                    state_tick=brain.state.tick_id,
                    value=record.feedback_value,
                    confidence=confidence,
                    trusted_now=brain.state.clock,
                    state_clock=brain.state.clock,
                    feedback_ttl_seconds=_FEEDBACK_TTL_SECONDS,
                )
                c_state = c_feedback.state
        elif model_name == "matched_compute_continuous":
            drive = [
                _clip(
                    record.features[axis]
                    + (0.2 * parameter * pending_continuous_feedback if axis == 0 else 0.0)
                )
                for axis in range(N_AXES)
            ]
            for _ in range(4):
                continuous = [
                    _clip(0.75 * continuous[axis] + 0.25 * drive[axis]) for axis in range(N_AXES)
                ]
            predicted = tuple(continuous)
            pending_continuous_feedback = record.feedback_value
        else:
            raise ValueError(f"unknown evaluator adapter {model_name!r}")

        predictions.append(predicted)
        previous = list(predicted)
        previous_features = list(record.features)
    return predictions


def _session_mae(
    grouped: Mapping[str, Sequence[NumericRecord]],
    *,
    model_name: str,
    parameter: float,
    seed: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for session_id in sorted(grouped):
        records = grouped[session_id]
        predictions = _predict_session(
            records,
            model_name=model_name,
            parameter=parameter,
            seed=seed,
        )
        errors = [
            abs(prediction[axis] - record.target[axis])
            for record, prediction in zip(records, predictions, strict=True)
            for axis in range(N_AXES)
        ]
        result[session_id] = fmean(errors)
    return result


def _candidate_parameters(budget: int) -> tuple[float, ...]:
    _positive_int("tuning_budget", budget)
    return tuple((index + 1) / (budget + 1) for index in range(budget))


def adapter_registry(*, tuning_budget: int) -> dict[str, dict[str, object]]:
    """Describe every preregistered adapter without executing an evaluation."""
    candidates = list(_candidate_parameters(tuning_budget))
    actual_core = {"pel", "b_only", "b_plus_c", "feedback_shuffled", "eligibility_disabled"}
    return {
        name: {
            "tuning_budget": tuning_budget,
            "candidate_parameters": list(candidates),
            "adapter_kind": "repository_core" if name in actual_core else "offline_numeric",
        }
        for name in ADAPTER_NAMES
    }


def _best_parameter(
    records: Sequence[NumericRecord],
    sessions: Sequence[str],
    *,
    model_name: str,
    budget: int,
    seed: int,
) -> float:
    selected = set(sessions)
    grouped = _group_records(record for record in records if record.session_id in selected)
    ranked: list[tuple[float, float]] = []
    for parameter in _candidate_parameters(budget):
        scores = _session_mae(grouped, model_name=model_name, parameter=parameter, seed=seed)
        ranked.append((fmean(scores.values()), parameter))
    return min(ranked)[1]


def evaluate_corpus(
    records: Sequence[NumericRecord] | Iterable[NumericRecord],
    *,
    tuning_budget: int = 8,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    materialized = sorted(tuple(records), key=lambda item: (item.session_id, item.tick_id))
    record_keys = {(record.session_id, record.tick_id) for record in materialized}
    if len(record_keys) != len(materialized):
        raise ValueError("evaluation records contain duplicate session/tick")
    session_count = len({record.session_id for record in materialized})
    target_ticks = len(record_keys)
    if session_count < MIN_SESSIONS or target_ticks < MIN_TARGET_TICKS:
        return {
            "status": "insufficient_data",
            "sessions": session_count,
            "target_ticks": target_ticks,
            "minimum_sessions": MIN_SESSIONS,
            "minimum_target_ticks": MIN_TARGET_TICKS,
            "models": {},
            "promotion": {
                "status": "refused",
                "reason": "insufficient_data",
                "absolute_lower_bound": None,
                "relative_lower_bound": None,
            },
        }

    splits = session_block_splits(materialized)
    records_per_session = Counter(record.session_id for record in materialized)
    heldout_assignments = Counter(
        session_id for split in splits for session_id in split.test_sessions
    )
    heldout_sessions = set(heldout_assignments)
    heldout_target_ticks = sum(records_per_session[session_id] for session_id in heldout_sessions)
    complete_heldout_coverage = heldout_sessions == set(records_per_session) and all(
        count == 1 for count in heldout_assignments.values()
    )
    fold_assignments = [
        {
            "fold": fold_index,
            "seed": split.seed,
            "test_sessions": list(split.test_sessions),
            "target_ticks": sum(
                records_per_session[session_id] for session_id in split.test_sessions
            ),
        }
        for fold_index, split in enumerate(splits)
    ]
    heldout_coverage = {
        "sessions": len(heldout_sessions),
        "target_ticks": heldout_target_ticks,
        "total_sessions": session_count,
        "total_target_ticks": target_ticks,
        "complete": complete_heldout_coverage,
    }
    per_model_observations: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ADAPTER_NAMES
    }
    chosen_parameters: dict[str, list[float]] = {name: [] for name in ADAPTER_NAMES}
    for split in splits:
        test_set = set(split.test_sessions)
        test_grouped = _group_records(
            record for record in materialized if record.session_id in test_set
        )
        for name in ADAPTER_NAMES:
            parameter = _best_parameter(
                materialized,
                split.train_sessions,
                model_name=name,
                budget=tuning_budget,
                seed=split.seed,
            )
            chosen_parameters[name].append(parameter)
            scores = _session_mae(
                test_grouped,
                model_name=name,
                parameter=parameter,
                seed=split.seed,
            )
            for session_id, score in scores.items():
                per_model_observations[name][session_id].append(score)

    models: dict[str, dict[str, object]] = {}
    averaged: dict[str, dict[str, float]] = {}
    registry = adapter_registry(tuning_budget=tuning_budget)
    for name in ADAPTER_NAMES:
        session_scores = {
            session_id: fmean(values)
            for session_id, values in sorted(per_model_observations[name].items())
        }
        averaged[name] = session_scores
        models[name] = {
            "mae": fmean(session_scores.values()),
            "per_session_mae": session_scores,
            "tuning_budget": tuning_budget,
            "selected_parameters": chosen_parameters[name],
            "adapter_kind": registry[name]["adapter_kind"],
        }

    continuous_names = (
        "current_v26",
        "ema_arx",
        "pel",
        "b_only",
        "matched_compute_continuous",
    )

    def model_mae(name: str) -> float:
        value = models[name]["mae"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"model {name} has a nonnumeric MAE")
        return float(value)

    strongest = min(continuous_names, key=model_mae)
    differences = paired_session_differences(averaged[strongest], averaged["b_plus_c"])
    interval = bootstrap_ci(
        differences,
        resamples=bootstrap_resamples,
        seed=FIXED_SPLIT_SEEDS[0],
    )
    decision = promotion_decision(
        baseline_mae=model_mae(strongest),
        candidate_mae=model_mae("b_plus_c"),
        improvement_ci=interval,
        sessions=len(heldout_sessions),
        target_ticks=heldout_target_ticks,
    )
    if not complete_heldout_coverage:
        decision = {
            "status": "refused",
            "reason": "incomplete_heldout_coverage",
            "absolute_lower_bound": None,
            "relative_lower_bound": None,
        }
    decision.update(
        {
            "baseline": strongest,
            "candidate": "b_plus_c",
            "improvement_ci95": list(interval),
        }
    )
    return {
        "status": "evaluated",
        "sessions": session_count,
        "target_ticks": target_ticks,
        "split_seeds": list(FIXED_SPLIT_SEEDS),
        "split_unit": "session",
        "tuning_scope": "training_sessions_only",
        "heldout_coverage": heldout_coverage,
        "fold_assignments": fold_assignments,
        "models": models,
        "promotion": decision,
    }


def synthetic_corpus(
    *,
    session_count: int = MIN_SESSIONS,
    ticks_per_session: int = 34,
    seed: int = 2718,
) -> list[NumericRecord]:
    _positive_int("session_count", session_count)
    _positive_int("ticks_per_session", ticks_per_session)
    generator = random.Random(seed)
    records: list[NumericRecord] = []
    for session in range(session_count):
        phase = generator.randrange(2)
        previous_label = 0.0
        for tick in range(1, ticks_per_session + 1):
            label = 1.0 if (tick + phase) % 2 == 0 else -1.0
            features = tuple(
                scale * label for scale in (0.8, -0.8, 0.6, -0.6, 0.4, -0.4, 0.2, -0.2)
            )
            target = (0.9 * label,) * N_AXES
            records.append(
                NumericRecord(
                    schema_version=CORPUS_SCHEMA_VERSION,
                    session_id=f"synthetic-{session:03d}",
                    tick_id=tick,
                    features=features,
                    target=target,
                    feedback_target_tick=tick - 1,
                    feedback_value=previous_label,
                )
            )
            previous_label = label
    return records


def synthetic_sanity(*, seed: int = 2718) -> dict[str, object]:
    records = synthetic_corpus(session_count=6, ticks_per_session=64, seed=seed)
    grouped = _group_records(records)
    target_scores = _session_mae(
        grouped,
        model_name="b_plus_c",
        parameter=1.0,
        seed=seed,
    )
    shuffled_scores = _session_mae(
        grouped,
        model_name="feedback_shuffled",
        parameter=1.0,
        seed=seed,
    )
    target_mae = fmean(target_scores.values())
    shuffled_mae = fmean(shuffled_scores.values())
    advantage = shuffled_mae - target_mae
    return {
        "status": "pass" if advantage > 0.02 else "failed",
        "target_mae": target_mae,
        "shuffled_mae": shuffled_mae,
        "absolute_advantage": advantage,
        "compared_adapters": ["b_plus_c", "feedback_shuffled"],
        "measurement": "model_prediction_mae",
        "shuffle_scope": "within_session",
        "sessions": len(grouped),
        "target_ticks": len(records),
        "seed": seed,
    }


def run_evaluation(
    *,
    corpus_path: str | Path | None,
    synthetic: bool,
    tuning_budget: int = 8,
) -> dict[str, object]:
    if corpus_path is None:
        real: dict[str, object] = {
            "status": "insufficient_data",
            "sessions": 0,
            "target_ticks": 0,
            "models": {},
            "promotion": {"status": "refused", "reason": "insufficient_data"},
        }
    else:
        real = evaluate_corpus(load_corpus(corpus_path), tuning_budget=tuning_budget)
    sanity: dict[str, object] | None = synthetic_sanity() if synthetic else None
    promotion_raw = real.get("promotion", {"status": "refused", "reason": "no_evaluation"})
    if not isinstance(promotion_raw, Mapping):
        raise ValueError("evaluation promotion must be a mapping")
    promotion = dict(promotion_raw)
    if sanity is None and promotion.get("status") == "pass":
        promotion = {"status": "refused", "reason": "synthetic_sanity_required"}
    elif sanity is not None and sanity["status"] != "pass":
        promotion = {"status": "refused", "reason": "synthetic_sanity_failed"}
    real_report = dict(real)
    real_report["value_gate_promotion"] = dict(promotion_raw)
    real_report["promotion"] = dict(promotion)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluator_semantics_version": EVALUATOR_SEMANTICS_VERSION,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "synthetic_sanity": sanity,
        "real_corpus": real_report,
        "promotion": promotion,
    }


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
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.write("\n")
        handle.flush()
    temporary.replace(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--tuning-budget", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_evaluation(
        corpus_path=arguments.corpus,
        synthetic=arguments.synthetic_sanity,
        tuning_budget=arguments.tuning_budget,
    )
    write_report(report, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
