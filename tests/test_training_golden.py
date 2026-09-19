"""Tests for write_golden and verify_golden.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path

from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal, Program, node
from snail.training import verify_golden, write_golden


class In(BaseModel):
    x: int


class Ok(BaseModel):
    y: int


class Out(NodeResult):
    ok: Ok | None = None
    ood: OODSignal | None = None


@node(name="double", input_schema=In, output_schema=Out, distribution="t")
def double(ctx: NodeContext, weights, payload: In):
    return Out(ok=Ok(y=payload.x * 2))


prog = Program(name="p", nodes=[double])


def test_write_and_verify_golden_passes(tmp_path: Path):
    write_golden("double", tmp_path, {"input": {"x": 5}, "expected": {"y": 10}})
    failures = verify_golden(prog, tmp_path)
    assert failures == []


def test_verify_golden_reports_failure(tmp_path: Path):
    write_golden("double", tmp_path, {"input": {"x": 5}, "expected": {"y": 999}})
    failures = verify_golden(prog, tmp_path)
    assert len(failures) == 1
    assert failures[0].reason == "payload mismatch"


def test_verify_golden_supports_min_max_bounds(tmp_path: Path):
    write_golden("double", tmp_path, {"input": {"x": 5}, "expected": {"y_min": 9, "y_max": 11}})
    failures = verify_golden(prog, tmp_path)
    assert failures == []  # 10 is in [9, 11]

    write_golden("double", tmp_path, {"input": {"x": 5}, "expected": {"y_min": 100}})
    failures = verify_golden(prog, tmp_path)
    assert len(failures) == 1


def test_verify_golden_empty_dir_returns_no_failures(tmp_path: Path):
    assert verify_golden(prog, tmp_path) == []
