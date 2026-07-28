from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from experiments.brain_eval import (
    ADAPTER_NAMES,
    CORPUS_SCHEMA_VERSION,
    FIXED_SPLIT_SEEDS,
    NumericRecord,
    _predict_session,
    _shuffled_feedback_targets,
    adapter_registry,
    bootstrap_ci,
    evaluate_corpus,
    load_corpus,
    paired_session_differences,
    promotion_decision,
    run_evaluation,
    session_block_splits,
    synthetic_sanity,
    write_report,
)

ZERO8 = (0.0,) * 8


def _record(session: str, tick: int, *, target: float = 0.0) -> NumericRecord:
    return NumericRecord(
        schema_version=CORPUS_SCHEMA_VERSION,
        session_id=session,
        tick_id=tick,
        features=(target / 2.0,) + ZERO8[1:],
        target=(target,) + ZERO8[1:],
        feedback_target_tick=max(0, tick - 1),
        feedback_value=max(-1.0, min(1.0, target)),
    )


def test_versioned_jsonl_loader_accepts_only_finite_numeric_records(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    document = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "session_id": "s1",
        "tick_id": 1,
        "features": [0.0] * 8,
        "target": [0.25] * 8,
        "feedback_target_tick": 0,
        "feedback_value": 0.5,
    }
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    assert load_corpus(path) == [
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id="s1",
            tick_id=1,
            features=ZERO8,
            target=(0.25,) * 8,
            feedback_target_tick=0,
            feedback_value=0.5,
        )
    ]

    document["target"][0] = math.nan
    path.write_text(json.dumps(document, allow_nan=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_corpus(path)


def test_loader_rejects_unknown_version_duplicate_tick_and_tick_splits(tmp_path: Path) -> None:
    base = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "session_id": "s1",
        "tick_id": 1,
        "features": [0.0] * 8,
        "target": [0.0] * 8,
        "feedback_target_tick": 0,
        "feedback_value": 0.0,
    }
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join((json.dumps(base), json.dumps(base))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path)

    base["schema_version"] = CORPUS_SCHEMA_VERSION + 1
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_corpus(path)


def test_session_block_splits_are_deterministic_disjoint_and_use_five_seeds() -> None:
    records = [_record(f"s{session}", tick) for session in range(10) for tick in range(1, 4)]

    first = session_block_splits(records)
    second = session_block_splits(list(reversed(records)))

    assert first == second
    assert tuple(split.seed for split in first) == FIXED_SPLIT_SEEDS
    held_out = [session_id for split in first for session_id in split.test_sessions]
    assert sorted(held_out) == sorted({record.session_id for record in records})
    assert len(held_out) == len(set(held_out))
    assert [len(split.test_sessions) for split in first] == [2, 2, 2, 2, 2]
    for split in first:
        assert set(split.train_sessions).isdisjoint(split.test_sessions)
        assert set(split.train_sessions) | set(split.test_sessions) == {
            f"s{session}" for session in range(10)
        }


def test_paired_session_differences_require_the_same_sessions() -> None:
    assert paired_session_differences(
        {"a": 0.4, "b": 0.3},
        {"a": 0.1, "b": 0.35},
    ) == pytest.approx((0.3, -0.05))
    with pytest.raises(ValueError, match="same sessions"):
        paired_session_differences({"a": 0.4}, {"b": 0.1})


def test_bootstrap_ci_is_seeded_and_uses_requested_10000_resamples() -> None:
    first = bootstrap_ci((0.1, 0.1, 0.1), resamples=10_000, seed=2718)
    second = bootstrap_ci((0.1, 0.1, 0.1), resamples=10_000, seed=2718)

    assert first == second == pytest.approx((0.1, 0.1))
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ci((0.1,), resamples=9_999, seed=2718)


@pytest.mark.parametrize(
    ("baseline", "candidate", "ci", "expected"),
    [
        (0.40, 0.35, (0.03, 0.07), "pass"),
        (0.40, 0.37, (0.019, 0.04), "refused"),
        (0.40, 0.37, (0.0201, 0.04), "pass"),
        (0.20, 0.17, (0.0101, 0.04), "refused"),
    ],
)
def test_promotion_requires_both_absolute_and_relative_lower_bounds(
    baseline: float,
    candidate: float,
    ci: tuple[float, float],
    expected: str,
) -> None:
    decision = promotion_decision(
        baseline_mae=baseline,
        candidate_mae=candidate,
        improvement_ci=ci,
        sessions=30,
        target_ticks=1_000,
    )

    assert decision["status"] == expected


def test_minimum_real_corpus_gate_refuses_metrics_and_promotion() -> None:
    too_few_sessions = [_record(f"s{i}", tick) for i in range(29) for tick in range(1, 36)]
    too_few_ticks = [_record(f"s{i}", tick) for i in range(30) for tick in range(1, 34)]

    for records in (too_few_sessions, too_few_ticks):
        report = evaluate_corpus(records, tuning_budget=2, bootstrap_resamples=10_000)
        assert report["status"] == "insufficient_data"
        assert report["models"] == {}
        assert report["promotion"]["status"] == "refused"


def test_direct_evaluation_rejects_duplicate_session_tick() -> None:
    records = [_record(f"s{i:02d}", tick) for i in range(30) for tick in range(1, 35)]
    records.append(records[0])

    with pytest.raises(ValueError, match="duplicate session/tick"):
        evaluate_corpus(records, tuning_budget=2, bootstrap_resamples=10_000)


def test_adapter_registry_has_every_required_ablation_with_equal_budget() -> None:
    registry = adapter_registry(tuning_budget=2)

    assert tuple(registry) == ADAPTER_NAMES
    assert set(ADAPTER_NAMES) == {
        "current_v26",
        "ema_arx",
        "pel",
        "b_only",
        "b_plus_c",
        "feedback_shuffled",
        "eligibility_disabled",
        "matched_compute_continuous",
    }
    assert {model["tuning_budget"] for model in registry.values()} == {2}
    assert {len(model["candidate_parameters"]) for model in registry.values()} == {2}


def _feedback_records(*, target_tick: int = 1) -> list[NumericRecord]:
    return [
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id="feedback-session",
            tick_id=1,
            features=(0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1),
            target=ZERO8,
            feedback_target_tick=0,
            feedback_value=0.0,
        ),
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id="feedback-session",
            tick_id=2,
            features=(0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.05),
            target=ZERO8,
            feedback_target_tick=target_tick,
            feedback_value=1.0,
        ),
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id="feedback-session",
            tick_id=3,
            features=(0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1),
            target=ZERO8,
            feedback_target_tick=2,
            feedback_value=0.0,
        ),
    ]


def test_b_plus_c_feedback_is_delayed_and_targets_the_recorded_tick() -> None:
    targeted = _predict_session(
        _feedback_records(target_tick=1),
        model_name="b_plus_c",
        parameter=1.0,
        seed=2718,
    )
    missed = _predict_session(
        _feedback_records(target_tick=0),
        model_name="b_plus_c",
        parameter=1.0,
        seed=2718,
    )

    assert targeted[:2] == missed[:2]
    assert targeted[2] != missed[2]


def test_current_feedback_is_not_synchronously_mixed_into_appraisal() -> None:
    positive = _feedback_records(target_tick=0)[:2]
    negative = [
        positive[0],
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id=positive[1].session_id,
            tick_id=positive[1].tick_id,
            features=positive[1].features,
            target=positive[1].target,
            feedback_target_tick=0,
            feedback_value=-1.0,
        ),
    ]

    positive_predictions = _predict_session(
        positive,
        model_name="b_plus_c",
        parameter=1.0,
        seed=2718,
    )
    negative_predictions = _predict_session(
        negative,
        model_name="b_plus_c",
        parameter=1.0,
        seed=2718,
    )

    assert positive_predictions == negative_predictions


def test_matched_continuous_applies_current_feedback_only_after_current_prediction() -> None:
    first = NumericRecord(
        schema_version=CORPUS_SCHEMA_VERSION,
        session_id="matched-feedback-session",
        tick_id=1,
        features=ZERO8,
        target=ZERO8,
        feedback_target_tick=0,
        feedback_value=0.0,
    )
    positive = [
        first,
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id=first.session_id,
            tick_id=2,
            features=ZERO8,
            target=ZERO8,
            feedback_target_tick=1,
            feedback_value=1.0,
        ),
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id=first.session_id,
            tick_id=3,
            features=ZERO8,
            target=ZERO8,
            feedback_target_tick=2,
            feedback_value=0.0,
        ),
    ]
    negative = [
        positive[0],
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id=positive[1].session_id,
            tick_id=positive[1].tick_id,
            features=positive[1].features,
            target=positive[1].target,
            feedback_target_tick=positive[1].feedback_target_tick,
            feedback_value=-1.0,
        ),
        positive[2],
    ]

    positive_predictions = _predict_session(
        positive,
        model_name="matched_compute_continuous",
        parameter=1.0,
        seed=2718,
    )
    negative_predictions = _predict_session(
        negative,
        model_name="matched_compute_continuous",
        parameter=1.0,
        seed=2718,
    )

    assert positive_predictions[:2] == negative_predictions[:2]
    assert positive_predictions[2] != negative_predictions[2]


def test_feedback_adapters_use_real_b_and_c_targeted_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    from sylanne_core.compute import brain_c
    from sylanne_core.compute.brain_compute import BrainComputeCore

    b_calls: list[tuple[int, float, int]] = []
    c_calls: list[tuple[int, float, int]] = []
    original_b = BrainComputeCore.prepare_feedback
    original_c = brain_c.evolve_c_feedback

    def spy_b(self: BrainComputeCore, **kwargs: object):
        b_calls.append(
            (
                int(kwargs["target_tick"]),
                float(kwargs["value"]),
                len(self.state.eligibility_records),
            )
        )
        return original_b(self, **kwargs)

    def spy_c(state: object, **kwargs: object):
        c_calls.append(
            (
                int(kwargs["target_tick"]),
                float(kwargs["value"]),
                len(state.eligibility_records),
            )
        )
        return original_c(state, **kwargs)

    monkeypatch.setattr(BrainComputeCore, "prepare_feedback", spy_b)
    monkeypatch.setattr(brain_c, "evolve_c_feedback", spy_c)
    records = _feedback_records(target_tick=1)

    _predict_session(records, model_name="b_plus_c", parameter=1.0, seed=2718)
    assert [call[0] for call in b_calls] == [0, 1, 2]
    assert [call[0] for call in c_calls] == [0, 1, 2]

    b_calls.clear()
    c_calls.clear()
    _predict_session(records, model_name="feedback_shuffled", parameter=1.0, seed=42)
    first_b = list(b_calls)
    first_c = list(c_calls)
    b_calls.clear()
    c_calls.clear()
    _predict_session(records, model_name="feedback_shuffled", parameter=1.0, seed=42)
    assert b_calls == first_b
    assert c_calls == first_c
    assert [call[0] for call in first_b] != [0, 1, 2]
    assert [call[0] for call in first_c] == [call[0] for call in first_b]

    b_calls.clear()
    c_calls.clear()
    _predict_session(records, model_name="eligibility_disabled", parameter=1.0, seed=2718)
    assert any(call[1] != 0.0 for call in b_calls)
    assert any(call[1] != 0.0 for call in c_calls)
    assert {call[2] for call in b_calls} == {0}
    assert {call[2] for call in c_calls} == {0}


def test_shuffled_targets_remain_inside_the_causal_eligibility_window() -> None:
    records = [
        NumericRecord(
            schema_version=CORPUS_SCHEMA_VERSION,
            session_id="long-feedback-session",
            tick_id=tick,
            features=ZERO8,
            target=ZERO8,
            feedback_target_tick=max(0, tick - 1),
            feedback_value=1.0,
        )
        for tick in range(1, 65)
    ]

    shuffled = _shuffled_feedback_targets(records, seed=42)

    for index, (record, target) in enumerate(zip(records, shuffled, strict=True)):
        retained_ticks = {item.tick_id for item in records[max(0, index - 31) : index + 1]}
        assert target in retained_ticks
        assert target != record.feedback_target_tick


def test_synthetic_target_advantage_beats_within_session_shuffle() -> None:
    first = synthetic_sanity(seed=42)
    second = synthetic_sanity(seed=42)

    assert first == second
    assert first["status"] == "pass"
    assert first["target_mae"] + 0.02 < first["shuffled_mae"]
    assert first["compared_adapters"] == ["b_plus_c", "feedback_shuffled"]
    assert first["measurement"] == "model_prediction_mae"
    assert first["shuffle_scope"] == "within_session"


def test_synthetic_cli_report_never_claims_real_corpus_promotion(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.json"

    report = run_evaluation(corpus_path=None, synthetic=True, tuning_budget=2)
    write_report(report, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 2
    assert loaded["evaluator_semantics_version"] == 2
    assert loaded["synthetic_sanity"]["status"] == "pass"
    assert loaded["real_corpus"]["status"] == "insufficient_data"
    assert loaded["promotion"]["status"] == "refused"


def test_real_corpus_promotion_requires_synthetic_sanity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from experiments import brain_eval

    monkeypatch.setattr(brain_eval, "load_corpus", lambda _path: [])
    monkeypatch.setattr(
        brain_eval,
        "evaluate_corpus",
        lambda _records, *, tuning_budget: {
            "status": "complete",
            "promotion": {"status": "pass", "reason": "all_value_gates_passed"},
        },
    )

    report = brain_eval.run_evaluation(
        corpus_path=tmp_path / "corpus.jsonl",
        synthetic=False,
        tuning_budget=2,
    )

    assert report["synthetic_sanity"] is None
    assert report["promotion"] == {
        "status": "refused",
        "reason": "synthetic_sanity_required",
    }
    assert report["real_corpus"]["promotion"] == report["promotion"]
    assert report["real_corpus"]["value_gate_promotion"] == {
        "status": "pass",
        "reason": "all_value_gates_passed",
    }


def test_evaluation_reports_complete_actual_heldout_coverage_and_fold_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import brain_eval

    records = [_record(f"s{i:02d}", tick) for i in range(30) for tick in range(1, 35)]

    monkeypatch.setattr(
        brain_eval,
        "_best_parameter",
        lambda _records, _sessions, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        brain_eval,
        "_session_mae",
        lambda grouped, **_kwargs: {session_id: 0.25 for session_id in sorted(grouped)},
    )

    report = brain_eval.evaluate_corpus(
        records,
        tuning_budget=2,
        bootstrap_resamples=10_000,
    )

    coverage = report["heldout_coverage"]
    assert coverage == {
        "sessions": 30,
        "target_ticks": 1_020,
        "total_sessions": 30,
        "total_target_ticks": 1_020,
        "complete": True,
    }
    assignments = report["fold_assignments"]
    assert [assignment["fold"] for assignment in assignments] == [0, 1, 2, 3, 4]
    assert [assignment["seed"] for assignment in assignments] == list(FIXED_SPLIT_SEEDS)
    assert sum(assignment["target_ticks"] for assignment in assignments) == 1_020
    held_out = [
        session_id for assignment in assignments for session_id in assignment["test_sessions"]
    ]
    assert sorted(held_out) == [f"s{i:02d}" for i in range(30)]
    assert len(held_out) == len(set(held_out))
    assert len(report["models"]["b_plus_c"]["per_session_mae"]) == 30


def test_promotion_gate_uses_actual_heldout_counts_and_refuses_incomplete_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import brain_eval

    records = [_record(f"s{i:02d}", tick) for i in range(30) for tick in range(1, 35)]
    incomplete_splits = brain_eval.session_block_splits(records)[:4]
    captured: dict[str, int] = {}

    monkeypatch.setattr(
        brain_eval,
        "session_block_splits",
        lambda _records: incomplete_splits,
    )
    monkeypatch.setattr(
        brain_eval,
        "_best_parameter",
        lambda _records, _sessions, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        brain_eval,
        "_session_mae",
        lambda grouped, **_kwargs: {session_id: 0.25 for session_id in sorted(grouped)},
    )

    def capture_promotion(**kwargs: object) -> dict[str, object]:
        captured["sessions"] = int(kwargs["sessions"])
        captured["target_ticks"] = int(kwargs["target_ticks"])
        return {
            "status": "pass",
            "reason": "thresholds_met",
            "absolute_lower_bound": 1.0,
            "relative_lower_bound": 1.0,
        }

    monkeypatch.setattr(brain_eval, "promotion_decision", capture_promotion)

    report = brain_eval.evaluate_corpus(
        records,
        tuning_budget=2,
        bootstrap_resamples=10_000,
    )

    assert captured == {"sessions": 24, "target_ticks": 816}
    assert report["heldout_coverage"]["complete"] is False
    assert report["promotion"]["status"] == "refused"
    assert report["promotion"]["reason"] == "incomplete_heldout_coverage"


def test_evaluation_report_requires_an_explicit_output_path() -> None:
    with pytest.raises(ValueError, match="explicit output"):
        write_report({"schema_version": 1}, None)


def test_direct_eval_script_bootstraps_repository_root_under_isolated_python() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "brain_eval.py"
    code = f"import runpy,sys; runpy.run_path({str(script)!r}); assert {str(root)!r} in sys.path"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=Path(tempfile.gettempdir()),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
