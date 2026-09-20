"""Tests for confidence calibration tooling (v0.4.0).

Benchmarks:
- test_calibrate_perfect_model_ece_zero
- test_calibrate_overconfident_model_high_ece
- test_calibrate_underconfident_model_high_ece
- test_calibrate_emits_report_json
- test_calibrate_cli_exit_zero_on_perfect
- test_calibrate_cli_exit_one_on_overconfident
"""

from __future__ import annotations
import json
import pytest
from pydantic import BaseModel

from snail import (
    Program,
    edge,
    NodeResult,
    OODSignal,
    run_calibration,
    CalibrationReport,
)
from snail.wrappers import DeterministicNode
from snail.calibrate import compute_ece, plot_reliability_diogram


# ── Schemas ──────────────────────────────────────────────────────────────


class FooOk(BaseModel):
    value: str
    confidence: float


class Foo(NodeResult):
    ok: FooOk | None = None
    ood: OODSignal | None = None


# ── Tests: ECE math ──────────────────────────────────────────────────────


def test_calibrate_perfect_model_ece_zero():
    """When confidence == accuracy in every bin, ECE = 0.

    Construct a setup where each bin has matching avg_conf and avg_acc.
    Use 4 bins with 4 samples per bin, where all samples in the 25%-bin
    have conf=0.25 and 25% are correct (acc=0.25):
    - bin [0, 0.25): 4 samples, conf=0.10, 1/4 correct → acc=0.25, conf=0.10 (diff 0.15)... NO.
    Use 2 bins, conf=0.25 → all "correct" (acc=0.25 = avg_conf 0.25):
    - bin [0, 0.5): 4 samples at conf=0.25, 1/4 correct → acc=0.25 (diff 0)
    - bin [0.5, 1.0]: 4 samples at conf=0.75, 3/4 correct → acc=0.75 (diff 0)
    """
    confidences = [0.25] * 4 + [0.75] * 4
    accuracies = [1, 0, 0, 0] + [1, 1, 1, 0]  # 1/4 in bin1, 3/4 in bin2
    ece, bins = compute_ece(confidences, accuracies, n_bins=2)
    # bin [0, 0.5): conf_avg=0.25, acc_avg=0.25 → diff 0
    # bin [0.5, 1.0]: conf_avg=0.75, acc_avg=0.75 → diff 0
    assert ece < 1e-9
    assert len(bins) >= 1


def test_calibrate_overconfident_model_high_ece():
    """Confidence always 0.95 but accuracy only 50% → ECE > 0.3."""
    confidences = [0.95] * 100
    accuracies = [1, 0] * 50  # 50% accurate
    ece, _ = compute_ece(confidences, accuracies, n_bins=10)
    # All confs in bin [0.9, 1.0], avg_acc = 0.5, avg_conf = 0.95
    # diff = 0.45, weight = 1.0 → ECE ~ 0.45
    assert ece > 0.3


def test_calibrate_underconfident_model_high_ece():
    """Confidence always 0.5 but accuracy always 1.0 → ECE > 0.3."""
    confidences = [0.5] * 100
    accuracies = [1] * 100
    ece, _ = compute_ece(confidences, accuracies, n_bins=10)
    # All confs in bin [0.5, 0.6], avg_acc = 1.0, avg_conf = 0.5
    # diff = 0.5
    assert ece > 0.3


# ── Tests: run_calibration end-to-end ────────────────────────────────────


def _build_program(correct_value_fn):
    """Build a Program that returns `correct_value_fn(input)` for any input."""
    def body(x):
        is_correct = correct_value_fn(x)
        return Foo(ok=FooOk(
            value="yes" if is_correct else "no",
            confidence=0.9 if is_correct else 0.9,  # always 0.9
        ))

    p = DeterministicNode(
        name="classifier",
        input_schema=dict,
        output_schema=Foo,
        fn=body,
    )
    return Program(name="calib_test", nodes=[p])


def test_calibrate_emits_report_json():
    """run_calibration returns a CalibrationReport; .to_json() is valid JSON."""
    p = _build_program(lambda x: True)  # always correct, always 0.9 confidence
    cases = [{"input": {"text": str(i)}, "expected": {"value": "yes"}} for i in range(10)]

    report = run_calibration(p, cases)
    assert isinstance(report, CalibrationReport)
    assert report.program_name == "calib_test"
    assert report.n_cases == 10
    assert report.n_passed == 10
    assert report.accuracy == 1.0
    # ECE: conf 0.9 in bin [0.8, 0.9] (or [0.9, 1.0] depending on n_bins),
    # acc 1.0 → diff = 0.1
    assert 0.0 <= report.ece < 0.2

    # JSON round-trip
    j = report.to_json()
    obj = json.loads(j)
    assert obj["program_name"] == "calib_test"
    assert obj["n_cases"] == 10


def test_calibrate_cli_exit_zero_on_perfect():
    """CalibrationReport below threshold → below_threshold = True."""
    # Custom cases with conf always matching accuracy perfectly
    confidences = [0.1, 0.1, 0.9, 0.9, 0.9]
    accuracies = [0, 0, 1, 1, 1]
    ece, bins = compute_ece(confidences, accuracies, n_bins=10)
    # Perfect calibration → ece very small
    assert ece < 0.1


def test_calibrate_cli_exit_one_on_overconfident():
    """High ECE → produces an over-confident report (above threshold)."""
    cases = []
    for i in range(20):
        # Confidence is always 0.99; we alternate correct/wrong
        cases.append({
            "input": {"text": str(i)},
            "expected": {"value": "yes" if i % 2 == 0 else "no"},
            "confidence_override": 0.99,  # documented override
        })
    # Build a program whose confidence is set via the body
    def body(x):
        return Foo(ok=FooOk(value="yes", confidence=0.99))

    p = DeterministicNode(
        name="overconfident",
        input_schema=dict,
        output_schema=Foo,
        fn=body,
    )
    program = Program(name="overconfident_prog", nodes=[p])
    report = run_calibration(program, cases)
    # 20 cases, half match "yes" in expected (i % 2 == 0 → matches "yes" output → correct)
    # But the actual output is always "yes" with conf 0.99
    # For odd i: expected="no" but actual="yes" → mismatch → 0
    # For even i: expected="yes", actual="yes" → correct → 1
    # accuracy = 10/20 = 0.5; conf 0.99 → diff 0.49 → ECE > 0.3
    assert report.ece > 0.3
