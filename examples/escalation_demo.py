"""Example: escalation_demo — small-model-first with OOD escalation.

Demonstrates the v0.3.0 `escalate()` primitive. A cheap tier-1 model
(Claude Haiku) classifies intent; if it returns OOD (confidence below
threshold), a heavier tier-2 model (Claude Sonnet) takes over.

The same `Program` runs on stub by default — no API key needed.

Run:
    python examples/escalation_demo.py

With real providers:
    export ANTHROPIC_API_KEY=sk-...
    # In code, change `provider="stub"` to `provider="anthropic"`.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail import (
    Program,
    edge,
    escalate,
    NodeResult,
    OODSignal,
)
from snail.wrappers import HostedNode, DeterministicNode


# ── Schemas ──────────────────────────────────────────────────────────────


class IntentOk(BaseModel):
    intent: Literal["billing", "technical", "other"]
    confidence: float


class Intent(NodeResult):
    ok: IntentOk | None = None
    ood: OODSignal | None = None


class FinalOk(BaseModel):
    intent: str
    routed_to_tier: Literal["tier1", "tier2"]
    confidence: float


class Final(NodeResult):
    ok: FinalOk | None = None
    ood: OODSignal | None = None


# ── Nodes ───────────────────────────────────────────────────────────────


classify_t1 = HostedNode(
    name="classify_t1",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3_tier1",
    endpoint="anthropic://claude-haiku-4-5",
    prompt_template="Classify the intent of: {text}",
    api_key_env="ANTHROPIC_API_KEY",
    provider="stub",
    cost_tier="small",
    confidence_threshold=0.7,
)

classify_t2 = HostedNode(
    name="classify_t2",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3_tier2",
    endpoint="anthropic://claude-sonnet-5",
    prompt_template="Classify the intent of: {text}",
    api_key_env="ANTHROPIC_API_KEY",
    provider="stub",
    cost_tier="large",
    confidence_threshold=0.85,
)

record_tier = DeterministicNode(
    name="record_tier",
    input_schema=dict,
    output_schema=Final,
    fn=lambda x: Final(ok=FinalOk(
        intent="(from t1)",
        routed_to_tier="tier1",
        confidence=0.9,
    )),
    cost_tier="small",
)


# ── Program ─────────────────────────────────────────────────────────────


escalation_demo = Program(
    name="escalation_demo",
    nodes=[classify_t1, classify_t2, record_tier],
    edges=[
        # t1 OK routes straight to record_tier
        edge(classify_t1.ok, target_field="t1_result"),
        # t2 OK also routes to record_tier (different field)
        edge(classify_t2.ok, target_field="t2_result"),
        # t1 OOD escalates to t2 (one-liner)
    ],
    escalations=[escalate(classify_t1.ood, to=classify_t2)],
)


def run_demo():
    print("=" * 60)
    print("SNAIL escalation_demo — small-model-first with OOD escalation")
    print("=" * 60)

    result = escalation_demo.run({"text": "I want a refund for order #12345"})

    print("\n── Manifest summary ──")
    print(f"  nodes fired       : {result.manifest.summary.nodes_fired}")
    print(f"  total tokens in   : {result.manifest.summary.total_tokens_in}")
    print(f"  total tokens out  : {result.manifest.summary.total_tokens_out}")
    print(f"  total cost (USD)  : ${result.manifest.summary.total_cost_usd:.6f}")

    print("\n── Per-node events ──")
    for ev in result.manifest.node_events:
        print(
            f"  {ev.node_name:20s} → {ev.variant:3s} "
            f"({ev.latency_ms:.1f}ms, ${ev.cost_usd:.6f})"
        )


if __name__ == "__main__":
    run_demo()
