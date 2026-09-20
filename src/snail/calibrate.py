"""Confidence calibration — ECE (Expected Calibration Error) and reliability diagrams.

`snail calibrate` runs the program's golden cases, plots predicted-confidence
vs actual-accuracy, and emits a machine-readable report. This is what makes
the OOD threshold trustable: the user can prove the threshold isn't wishful.

ECE definition (canonical):
    ECE = sum over bins (bin_count / total) * |accuracy_in_bin - avg_conf_in_bin|

A model with ECE = 0 is perfectly calibrated (confidence N means N%
accurate). A model with high ECE is over-confident or under-confident,
both of which cause downstream thresholds to misbehave.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class CalibrationBin:
    """One bin in a reliability diagram."""

    bin_low: float
    bin_high: float
    count: int
    accuracy: float
    avg_confidence: float


@dataclass
class CalibrationReport:
    """Result of running `snail calibrate` against a program."""

    program_name: str
    n_cases: int
    n_passed: int
    accuracy: float
    ece: float
    bins: list[CalibrationBin] = field(default_factory=list)
    threshold: float = 0.5
    below_threshold: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_name": self.program_name,
            "n_cases": self.n_cases,
            "n_passed": self.n_passed,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "bins": [asdict(b) for b in self.bins],
            "threshold": self.threshold,
            "below_threshold": self.below_threshold,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def compute_ece(
    confidences: list[float],
    accuracies: list[float],
    n_bins: int = 10,
) -> tuple[float, list[CalibrationBin]]:
    """Compute ECE and the per-bin statistics.

    Args:
        confidences: list of model-reported confidence scores (0..1).
        accuracies: list of 0/1 correctness indicators.
        n_bins: number of equal-width bins in [0, 1]. Default 10.

    Returns:
        (ece, bins) — ece is a float in [0, 1]; bins is a list of per-bin
        statistics suitable for plotting.
    """
    if len(confidences) != len(accuracies):
        raise ValueError(
            f"confidences ({len(confidences)}) and accuracies "
            f"({len(accuracies)}) must be the same length"
        )
    if not confidences:
        return 0.0, []

    bin_width = 1.0 / n_bins
    bin_edges = [i * bin_width for i in range(n_bins + 1)]
    bins: list[dict[str, Any]] = [
        {"low": bin_edges[i], "high": bin_edges[i + 1], "confs": [], "accs": []}
        for i in range(n_bins)
    ]

    for c, a in zip(confidences, accuracies):
        idx = min(int(c / bin_width), n_bins - 1)
        bins[idx]["confs"].append(c)
        bins[idx]["accs"].append(a)

    total = len(confidences)
    ece = 0.0
    out: list[CalibrationBin] = []
    for b in bins:
        if not b["confs"]:
            # Skip empty bins in ECE computation; record as zero-count.
            continue
        avg_conf = sum(b["confs"]) / len(b["confs"])
        avg_acc = sum(b["accs"]) / len(b["accs"])
        weight = len(b["confs"]) / total
        ece += weight * abs(avg_acc - avg_conf)
        out.append(
            CalibrationBin(
                bin_low=b["low"],
                bin_high=b["high"],
                count=len(b["confs"]),
                accuracy=avg_acc,
                avg_confidence=avg_conf,
            )
        )
    return ece, out


def run_calibration(
    program: Any,
    golden_cases: list[dict[str, Any]],
    *,
    confidence_attr: str = "confidence",
    expected_attr: str = "expected",
    n_bins: int = 10,
) -> CalibrationReport:
    """Run a program's golden cases and produce a CalibrationReport.

    Args:
        program: a SNAIL Program.
        golden_cases: list of dicts, each with shape
            `{"input": <program input>, "expected": <expected payload>,
              "confidence_min": <float>}`.
            The "expected" is matched against the program's output — a
            string match for non-dict outputs, a dict membership check
            for dicts.
        confidence_attr: which attribute on the OK payload carries
            confidence. Default "confidence".
        expected_attr: which key on the golden case carries the expected
            output. Default "expected".
        n_bins: number of bins for ECE. Default 10.

    Returns:
        CalibrationReport — call `.to_json()` or `.to_dict()` to serialize.
    """
    confidences: list[float] = []
    accuracies: list[int] = []

    for case in golden_cases:
        try:
            result = program.run(case["input"])
        except Exception:
            # Program crashed on this case. Treat as wrong with confidence 0.
            confidences.append(0.0)
            accuracies.append(0)
            continue

        # Find the terminal (last-fired) OK payload. If it's OK and we
        # can read a confidence, use that. Otherwise use 0.
        terminal = result.terminal
        if terminal is not None and terminal.is_ok and terminal.ok is not None:
            ok_payload = terminal.ok
            conf = getattr(ok_payload, confidence_attr, None)
            if conf is None and isinstance(ok_payload, dict):
                conf = ok_payload.get(confidence_attr)
            if conf is None:
                conf = 0.5  # unknown confidence → midpoint
            confidences.append(float(conf))
            # Accuracy check
            expected = case.get(expected_attr)
            actual = (
                ok_payload.model_dump()
                if hasattr(ok_payload, "model_dump")
                else (
                    ok_payload
                    if isinstance(ok_payload, dict)
                    else {"value": ok_payload}
                )
            )
            correct = _matches_expected(expected, actual)
            accuracies.append(1 if correct else 0)
        else:
            # No OK result (e.g., OOD throughout).
            # Use confidence = 0 if it was an OOD (it didn't answer).
            confidences.append(0.0)
            accuracies.append(0)

    ece, bins = compute_ece(confidences, accuracies, n_bins=n_bins)
    n_cases = len(golden_cases)
    n_passed = sum(accuracies)
    accuracy = n_passed / n_cases if n_cases else 0.0

    return CalibrationReport(
        program_name=program.name,
        n_cases=n_cases,
        n_passed=n_passed,
        accuracy=accuracy,
        ece=ece,
        bins=bins,
    )


def _matches_expected(expected: Any, actual: Any) -> bool:
    """Heuristic match: dict subset check if both are dicts; else ==."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(actual.get(k) == v for k, v in expected.items())
    return expected == actual


def plot_reliability_diogram(
    report: CalibrationReport,
    output_path: str | Path,
) -> Path:
    """Save a reliability-diagram PNG for the report.

    Requires matplotlib. If not installed, raises ImportError with a clear
    message.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plot_reliability_diogram. "
            "Install with `pip install snail-dsl[calibrate]`."
        ) from e

    fig, ax = plt.subplots(figsize=(7, 6))
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    # Bin points + bars
    if report.bins:
        mids = [(b.bin_low + b.bin_high) / 2 for b in report.bins]
        accs = [b.accuracy for b in report.bins]
        counts = [b.count for b in report.bins]
        ax.bar(mids, accs, width=1.0 / (len(report.bins) + 1), alpha=0.5,
               color="steelblue", label="Bin accuracy")
        ax.plot(mids, accs, "o-", color="steelblue")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(
        f"Reliability Diagram — {report.program_name}\n"
        f"ECE = {report.ece:.4f} | "
        f"n = {report.n_cases} | "
        f"acc = {report.accuracy:.3f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
