"""Tests for the parallel_edges() primitive (v0.3.0).

Benchmarks:
- test_parallel_basic_two_nodes: two extractors, both OK, merge receives both
- test_parallel_one_ood_one_ok: one OOD, one OK, merge sees only the OK
- test_parallel_both_ood: both OOD, no downstream OK to merge
- test_parallel_manifest_records_parallel_group
- test_parallel_rejects_duplicate_targets
- test_parallel_rejects_empty_to_list
"""

from __future__ import annotations
import pytest
from pydantic import BaseModel
from typing import Literal

from snail import (
    Program,
    edge,
    parallel_edges,
    NodeResult,
    OODSignal,
)
from snail.wrappers import DeterministicNode, HostedNode


# ── Schemas ──────────────────────────────────────────────────────────────


class ExtOk(BaseModel):
    value: str
    confidence: float


class Extracted(NodeResult):
    ok: ExtOk | None = None
    ood: OODSignal | None = None


class MergedOk(BaseModel):
    a: str
    b: str
    confidence: float


class Merged(NodeResult):
    ok: MergedOk | None = None
    ood: OODSignal | None = None


# ── Tests ─────────────────────────────────────────────────────────────────


def test_parallel_basic_two_nodes():
    """Two extractors run on the same input, both fire and both produce OK."""

    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    a = DeterministicNode(
        name="extract_a",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="alpha", confidence=0.95)),
    )
    b = DeterministicNode(
        name="extract_b",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="beta", confidence=0.92)),
    )
    # Merge node receives just a_result (b_result is also wired so both have
    # a consumer — required by static validation).
    merge = DeterministicNode(
        name="merge",
        input_schema=dict,
        output_schema=Merged,
        fn=lambda x: Merged(ok=MergedOk(a="x", b="y", confidence=0.9)),
    )

    p = Program(
        name="parallel_basic",
        nodes=[inp, a, b, merge],
        edges=[
            edge(a.ok, target_field="a_result"),
            edge(b.ok, target_field="b_result"),
        ],
        parallel_groups=[parallel_edges(inp, to=[a, b])],
    )

    result = p.run({"text": "hi"})
    assert "extract_a" in result.outputs
    assert "extract_b" in result.outputs
    assert result.outputs["extract_a"].is_ok
    assert result.outputs["extract_b"].is_ok


def test_parallel_one_ood_one_ok():
    """One extractor returns OOD, the other OK; merge sees only the OK."""

    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    # a always returns OK
    a = DeterministicNode(
        name="extract_a",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="alpha", confidence=0.95)),
    )
    # b always returns OOD
    b = DeterministicNode(
        name="extract_b",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ood=OODSignal(
            reason="explicit", confidence=0.0, threshold=0.5, distribution="d",
        )),
        ood_on_none=False,
    )

    p = Program(
        name="parallel_mixed",
        nodes=[inp, a, b],
        edges=[],
        parallel_groups=[parallel_edges(inp, to=[a, b])],
    )

    result = p.run({"text": "hi"})
    assert result.outputs["extract_a"].is_ok
    assert result.outputs["extract_b"].is_ood


def test_parallel_both_ood():
    """Both extractors return OOD — both events recorded as OOD in manifest."""

    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    a = DeterministicNode(
        name="extract_a",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ood=OODSignal(
            reason="explicit", confidence=0.0, threshold=0.5, distribution="d",
        )),
        ood_on_none=False,
    )
    b = DeterministicNode(
        name="extract_b",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ood=OODSignal(
            reason="explicit", confidence=0.0, threshold=0.5, distribution="d",
        )),
        ood_on_none=False,
    )

    p = Program(
        name="parallel_both_ood",
        nodes=[inp, a, b],
        edges=[],
        parallel_groups=[parallel_edges(inp, to=[a, b])],
    )

    result = p.run({"text": "hi"})
    assert result.outputs["extract_a"].is_ood
    assert result.outputs["extract_b"].is_ood
    fired = {e.node_name for e in result.manifest.node_events}
    assert "extract_a" in fired
    assert "extract_b" in fired


def test_parallel_manifest_records_parallel_group():
    """Manifest records both nodes fired from the same group."""
    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    a = DeterministicNode(
        name="extract_a",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="alpha", confidence=0.95)),
    )
    b = DeterministicNode(
        name="extract_b",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="beta", confidence=0.92)),
    )

    p = Program(
        name="manifest_group",
        nodes=[inp, a, b],
        edges=[],
        parallel_groups=[parallel_edges(inp, to=[a, b])],
    )

    result = p.run({"text": "hi"})
    fired = {e.node_name for e in result.manifest.node_events}
    # All three nodes fire
    assert "inp" in fired
    assert "extract_a" in fired
    assert "extract_b" in fired


def test_parallel_rejects_duplicate_targets():
    """Same target listed twice → ValueError at construction."""
    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    a = DeterministicNode(
        name="a",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="x", confidence=0.9)),
    )

    with pytest.raises(ValueError, match="duplicate"):
        parallel_edges(inp, to=[a, a])


def test_parallel_rejects_empty_to_list():
    """`to=[]` → ValueError."""
    inp = DeterministicNode(
        name="inp",
        input_schema=dict,
        output_schema=Extracted,
        fn=lambda x: Extracted(ok=ExtOk(value="raw", confidence=1.0)),
    )
    with pytest.raises(ValueError, match="non-empty"):
        parallel_edges(inp, to=[])
