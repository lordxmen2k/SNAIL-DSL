"""Example: calibration_demo — v0.4.0 confidence calibration.

A tiny program with a known confidence/accuracy profile, run through
`run_calibration()` to produce a CalibrationReport. Demonstrates:

1. The ECE metric over 10 bins.
2. The per-bin accuracy / confidence statistics.
3. JSON serialization for CI gating.

Run:
    python examples/calibration_demo.py
"""

from __future__ import annotations
import json
from pydantic import BaseModel

from snail import (
    Program,
    NodeResult,
    OODSignal,
    run_calibration,
)
from snail.wrappers import DeterministicNode


# ── Schemas ──────────────────────────────────────────────────────────────


class PredOk(BaseModel):
    label: str
    confidence: float


class Pred(NodeResult):
    ok: PredOk | None = None
    ood: OODSignal | None = None


# ── Program: returns labels with confidence proportional to "true" accuracy


def body_factory(true_label: str, reported_conf: float, is_correct: bool):
    """Build a node body that reports a fixed confidence and a fixed label."""
    def body(x):
        return Pred(ok=PredOk(
            label=true_label if is_correct else "WRONG",
            confidence=reported_conf,
        ))
    return body


# Two cases:
#   case 1: predicted "yes" with conf 0.9, expected "yes" → correct
#   case 2: predicted "maybe" with conf 0.5, expected "yes" → incorrect
correct = DeterministicNode(
    name="correct",
    input_schema=dict,
    output_schema=Pred,
    fn=body_factory("yes", 0.9, True),
)
incorrect = DeterministicNode(
    name="incorrect",
    input_schema=dict,
    output_schema=Pred,
    fn=body_factory("maybe", 0.5, False),
)

calibration_demo = Program(
    name="calibration_demo",
    nodes=[correct, incorrect],
)


def run_demo():
    print("=" * 60)
    print("SNAIL calibration_demo — confidence calibration")
    print("=" * 60)

    # Run each node with its own input
    cases = [
        # Each case routes to one node. We use the full program; calibration
        # uses the *terminal* node's output.
        {"input": {"text": "case1"}, "expected": {"label": "yes"}},
        {"input": {"text": "case1"}, "expected": {"label": "yes"}},
        {"input": {"text": "case1"}, "expected": {"label": "yes"}},
        {"input": {"text": "case2"}, "expected": {"label": "yes"}},  # wrong
        {"input": {"text": "case1"}, "expected": {"label": "yes"}},
    ]

    report = run_calibration(calibration_demo, cases)
    print("\n── Calibration report ──")
    print(f"  program:    {report.program_name}")
    print(f"  n_cases:    {report.n_cases}")
    print(f"  n_passed:   {report.n_passed}")
    print(f"  accuracy:   {report.accuracy:.3f}")
    print(f"  ECE:        {report.ece:.4f}")
    print(f"\n  Per-bin statistics:")
    for b in report.bins:
        print(
            f"    conf {b.bin_low:.2f}-{b.bin_high:.2f}: "
            f"count={b.count}, acc={b.accuracy:.2f}, "
            f"avg_conf={b.avg_confidence:.2f}"
        )

    print("\n── JSON report (truncated) ──")
    print(json.dumps(report.to_dict(), indent=2)[:600] + "...")


if __name__ == "__main__":
    run_demo()
