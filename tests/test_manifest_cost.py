"""Tests for cost/token accounting in Manifest (v0.3.0).

Benchmarks:
- test_manifest_records_zero_cost_for_stub
- test_manifest_aggregates_summary
- test_manifest_aggregates_tokens
- test_manifest_records_cost_usd_per_node
- test_manifest_top_level_summary_block
- test_manifest_round_trips_json_with_cost_fields
"""

from __future__ import annotations
import json
import pytest
from pydantic import BaseModel
from typing import Literal

from snail import (
    Program,
    edge,
    NodeResult,
    OODSignal,
)
from snail.wrappers import HostedNode, DeterministicNode
from snail.providers.pricing import (
    ANTHROPIC_PRICING,
    OPENAI_PRICING,
    cost_usd,
)


# ── Schemas ──────────────────────────────────────────────────────────────


class FooOk(BaseModel):
    value: str
    confidence: float


class Foo(NodeResult):
    ok: FooOk | None = None
    ood: OODSignal | None = None


# ── Tests: Manifest cost/token fields ─────────────────────────────────────


def test_manifest_records_zero_cost_for_stub():
    """Stub provider → cost_usd = 0, tokens = 0."""
    foo = HostedNode(
        name="foo",
        input_schema=dict,
        output_schema=Foo,
        distribution="d",
        endpoint="anthropic://claude-haiku",
        prompt_template="x",
        api_key_env="ANTHROPIC_API_KEY",
        provider="stub",
        cost_tier="small",
    )

    p = Program(name="zero_cost", nodes=[foo])
    result = p.run({"x": "y"})
    assert len(result.manifest.node_events) == 1
    ev = result.manifest.node_events[0]
    assert ev.cost_usd == 0.0
    assert ev.tokens_in == 0
    assert ev.tokens_out == 0


def test_manifest_aggregates_summary():
    """Multi-node run → summary.total_cost_usd = sum of per-node costs."""
    foo1 = DeterministicNode(
        name="foo1",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="a", confidence=0.9)),
    )
    foo2 = DeterministicNode(
        name="foo2",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="b", confidence=0.9)),
    )

    p = Program(
        name="agg",
        nodes=[foo1, foo2],
        edges=[edge(foo1.ok, target_field="input")],
    )
    result = p.run({"text": "hi"})
    s = result.manifest.summary
    # Deterministic nodes contribute 0 cost.
    assert s.total_cost_usd == 0.0
    assert s.nodes_fired == 2


def test_manifest_aggregates_tokens():
    """Multi-node run → summary.total_tokens_in/out = sum."""
    foo1 = HostedNode(
        name="foo1",
        input_schema=dict,
        output_schema=Foo,
        distribution="d",
        endpoint="anthropic://claude-haiku",
        prompt_template="x",
        api_key_env="ANTHROPIC_API_KEY",
        provider="stub",
        cost_tier="small",
    )
    foo2 = DeterministicNode(
        name="foo2",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="b", confidence=0.9)),
    )

    p = Program(
        name="tokens_agg",
        nodes=[foo1, foo2],
        edges=[edge(foo1.ok, target_field="input")],
    )
    result = p.run({"text": "hi"})
    s = result.manifest.summary
    # stub provider → 0 tokens
    assert s.total_tokens_in == 0
    assert s.total_tokens_out == 0
    assert s.nodes_fired == 2


def test_manifest_records_cost_usd_per_node():
    """Each node event has cost_usd field (default 0)."""
    foo = DeterministicNode(
        name="foo",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="x", confidence=0.9)),
    )
    p = Program(name="per_node_cost", nodes=[foo])
    result = p.run({"text": "hi"})
    for ev in result.manifest.node_events:
        assert hasattr(ev, "cost_usd")
        assert ev.cost_usd == 0.0
        assert hasattr(ev, "tokens_in")
        assert hasattr(ev, "tokens_out")
        assert hasattr(ev, "model_id")


def test_manifest_top_level_summary_block():
    """Manifest.to_dict() includes a 'summary' block."""
    foo = DeterministicNode(
        name="foo",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="x", confidence=0.9)),
    )
    p = Program(name="summary_block", nodes=[foo])
    result = p.run({"text": "hi"})
    d = result.manifest.to_dict()
    assert "summary" in d
    assert "total_cost_usd" in d["summary"]
    assert "total_tokens_in" in d["summary"]
    assert "total_tokens_out" in d["summary"]
    assert "nodes_fired" in d["summary"]
    assert "escalations_triggered" in d["summary"]


def test_manifest_round_trips_json_with_cost_fields():
    """Manifest.to_json() produces JSON that includes cost fields."""
    foo = DeterministicNode(
        name="foo",
        input_schema=dict,
        output_schema=Foo,
        fn=lambda x: Foo(ok=FooOk(value="x", confidence=0.9)),
    )
    p = Program(name="json_rt", nodes=[foo])
    result = p.run({"text": "hi"})
    raw = result.manifest.to_json()
    obj = json.loads(raw)
    assert "summary" in obj
    for ev in obj["node_events"]:
        assert "cost_usd" in ev
        assert "tokens_in" in ev
        assert "tokens_out" in ev


# ── Tests: pricing module ─────────────────────────────────────────────────


def test_anthropic_cost_for_claude_haiku():
    """claude-haiku-4-5: $0.001/1k in, $0.005/1k out. 1000 in + 200 out = $0.002."""
    cost = cost_usd("anthropic", "claude-haiku-4-5", tokens_in=1000, tokens_out=200)
    expected = (1000 / 1000) * 0.001 + (200 / 1000) * 0.005
    assert abs(cost - expected) < 1e-9


def test_anthropic_cost_for_claude_opus():
    """claude-opus-5 is the most expensive tier; verify rate table works."""
    cost = cost_usd("anthropic", "claude-opus-5", tokens_in=1000, tokens_out=200)
    in_rate, out_rate = ANTHROPIC_PRICING["claude-opus-5"]
    expected = in_rate + 0.2 * out_rate
    assert abs(cost - expected) < 1e-9


def test_user_override_pricing():
    """cost_per_1k_input/output overrides the table."""
    cost = cost_usd(
        "anthropic",
        "claude-opus-5",
        tokens_in=1000,
        tokens_out=200,
        override_input=0.5,
        override_output=2.0,
    )
    expected = 0.5 + 0.2 * 2.0
    assert abs(cost - expected) < 1e-9


def test_unknown_model_returns_zero_cost():
    """Unknown model → cost = 0 (don't make up numbers)."""
    cost = cost_usd("anthropic", "claude-mystery-99", tokens_in=1000, tokens_out=200)
    assert cost == 0.0


def test_ollama_returns_zero_cost():
    """Local provider → cost = 0."""
    cost = cost_usd("ollama", "llama3", tokens_in=1000000, tokens_out=500000)
    assert cost == 0.0
