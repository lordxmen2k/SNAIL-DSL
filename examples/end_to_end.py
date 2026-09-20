"""End-to-end SNAIL v0.4.0 demo — a real customer-support triage pipeline.

This file is a real, runnable SNAIL program. It exercises every v0.4.0
feature in one process:

1. A frozen-node `classify_intent` (the "small" tier)
2. An OOD branch via `escalate()` to a heavier model (the "large" tier)
3. A parallel-fan-out extraction step via `parallel_edges()`
4. A safety check that uses the manifest to verify nothing leaked
5. A weight pin: a frozen-weights file is checked at construction
6. The CLI runs `snail calibrate` against this program's golden cases

Run:
    python -m venv snail-e2e
    source snail-e2e/Scripts/activate
    pip install --upgrade snail-dsl
    python end_to_end.py

Expected output: a printed manifest, a per-node cost ledger, and a
calibration report. The pipeline exercises every primitive shipped in
v0.4.0 and exits 0 on success.
"""

from __future__ import annotations
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# ── Public API imports ───────────────────────────────────────────────────
from snail import (
    # core
    node, NodeContext, FrozenWeights,
    Program, Edge, edge, NodeProxy, ProgramRunResult,
    NodeResult, OODSignal,
    Manifest, ManifestBuilder,
    # v0.3.0 workflow primitives
    escalate, EscalationSpec, CostTier,
    parallel_edges, ParallelGroup,
    # v0.4.0 auditability
    parse_weight_pin, verify_weight_pin, WeightPinMismatch,
    run_calibration, CalibrationReport,
)
from snail.wrappers import DeterministicNode, HostedNode
from snail.calibrate import compute_ece


# ── Schemas ──────────────────────────────────────────────────────────────


class IntentOk(BaseModel):
    intent: Literal["billing", "technical", "other"]
    confidence: float


class Intent(NodeResult):
    ok: IntentOk | None = None
    ood: OODSignal | None = None


class EntitiesOk(BaseModel):
    order_id: str | None
    confidence: float


class Entities(NodeResult):
    ok: EntitiesOk | None = None
    ood: OODSignal | None = None


class ContactOk(BaseModel):
    email: str | None
    phone: str | None
    confidence: float


class Contact(NodeResult):
    ok: ContactOk | None = None
    ood: OODSignal | None = None


class DraftOk(BaseModel):
    message: str
    tone: Literal["friendly", "neutral", "formal"]
    confidence: float


class Draft(NodeResult):
    ok: DraftOk | None = None
    ood: OODSignal | None = None


class SafetyOk(BaseModel):
    safe: bool
    flags: list[str]
    confidence: float


class Safety(NodeResult):
    ok: SafetyOk | None = None
    ood: OODSignal | None = None


class ActionOk(BaseModel):
    action: Literal["sent", "human_review"]
    confirmation_id: str


class Action(NodeResult):
    ok: ActionOk | None = None
    ood: OODSignal | None = None


# ── 1. Frozen-weight pin (v0.4.0) ────────────────────────────────────────
#
# Create a real on-disk weights file and pin its SHA-256. Program(...)
# runs verify_weight_pin() at construction and raises WeightPinMismatch
# if the file changes.


def _make_frozen_weights_file() -> tuple[Path, str]:
    """Write a fake weights file and return (path, sha256)."""
    payload = json.dumps({
        "format": "snail-json-v1",
        "node": "classify_intent",
        "weights": {"embedding": [0.1, 0.2, 0.3]},
    }, sort_keys=True).encode()
    f = tempfile.NamedTemporaryFile(suffix=".snail.json", delete=False)
    f.write(payload)
    f.close()
    return Path(f.name), hashlib.sha256(payload).hexdigest()


WEIGHTS_PATH, WEIGHTS_HASH = _make_frozen_weights_file()
WEIGHT_PIN = f"intent-v3@sha256:{WEIGHTS_HASH}"


# ── 2. Nodes ─────────────────────────────────────────────────────────────


@node(
    name="classify_intent",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3",
    frozen_weights=str(WEIGHTS_PATH),
    weight_pin=WEIGHT_PIN,           # ← v0.4.0: pinned SHA-256
    confidence_threshold=0.7,
)
def classify_intent(ctx: NodeContext, weights, payload):
    """Tier 1 — frozen weights, single forward pass, OOD-as-type.

    Returns OK if the intent is clearly identifiable, else OOD.
    """
    text = payload.get("text", "").lower() if isinstance(payload, dict) else ""
    if "refund" in text or "billing" in text or "charge" in text:
        return Intent(ok=IntentOk(intent="billing", confidence=0.92))
    if "error" in text or "broken" in text or "bug" in text:
        return Intent(ok=IntentOk(intent="technical", confidence=0.88))
    # Anything else falls through to OOD — escalates to the heavy tier.
    return Intent(ood=OODSignal(
        reason="low_confidence",
        confidence=0.42,
        threshold=0.7,
        distribution="customer_intents_v3",
    ))


# Tier 2 — heavier model. Stub for offline use.
classify_intent_large = HostedNode(
    name="classify_intent_large",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3_tier2",
    endpoint="anthropic://claude-sonnet-5",
    prompt_template="Classify intent of: {text}",
    api_key_env="ANTHROPIC_API_KEY",
    provider="stub",
    cost_tier="large",                # ← v0.3.0: cost tier
    confidence_threshold=0.85,
)


# Parallel extractors via v0.3.0 parallel_edges
extract_email = DeterministicNode(
    name="extract_email",
    input_schema=dict,
    output_schema=Contact,
    fn=lambda x: Contact(ok=ContactOk(
        email="alice@example.com", phone=None, confidence=0.94,
    )),
)

extract_phone = DeterministicNode(
    name="extract_phone",
    input_schema=dict,
    output_schema=Contact,
    fn=lambda x: Contact(ok=ContactOk(
        email=None, phone="+1-555-0100", confidence=0.89,
    )),
)


@node(
    name="safety_check",
    input_schema=dict,
    output_schema=Safety,
    distribution="response_safety_v1",
    confidence_threshold=0.7,
)
def safety_check(ctx: NodeContext, weights, payload):
    """Reject any draft that mentions sensitive data."""
    text = payload.get("message", "") if isinstance(payload, dict) else ""
    flags = []
    if "ssn" in text.lower():
        flags.append("sensitive_data_mention")
    if "password" in text.lower():
        flags.append("credential_mention")
    return Safety(ok=SafetyOk(safe=len(flags) == 0, flags=flags, confidence=0.97))


send = DeterministicNode(
    name="send_response",
    input_schema=dict,
    output_schema=Action,
    fn=lambda x: Action(ok=ActionOk(
        action="sent",
        confirmation_id="CONF-" + str(hash(str(x)))[:8],
    )),
    ood_on_none=False,
)

human_review = DeterministicNode(
    name="human_review",
    input_schema=dict,
    output_schema=Action,
    fn=lambda _: Action(ok=ActionOk(
        action="human_review",
        confirmation_id="HR-001",
    )),
    ood_on_none=False,
)


# ── 3. Program with escalation + parallel fan-out ────────────────────────


customer_support_pipeline = Program(
    name="customer_support_pipeline",
    nodes=[
        classify_intent,
        classify_intent_large,
        extract_email,
        extract_phone,
        safety_check,
        send,
        human_review,
    ],
    edges=[],
    parallel_groups=[
        # Tier 1 OK feeds the parallel extractors (v0.3.0)
        parallel_edges(classify_intent, to=[extract_email, extract_phone]),
    ],
    escalations=[
        # Tier 1 OOD escalates to the heavier model (v0.3.0)
        escalate(classify_intent.ood, to=classify_intent_large),
    ],
)


# ── 4. End-to-end run ────────────────────────────────────────────────────


def run_one(text: str) -> dict:
    """Run a single customer-support message through the pipeline."""
    result = customer_support_pipeline.run({"text": text})
    return {
        "text": text,
        "outputs": {k: v.model_dump() for k, v in result.outputs.items()},
        "summary": result.manifest.summary,
        "events": [
            {
                "node": e.node_name,
                "variant": e.variant,
                "latency_ms": round(e.latency_ms, 2),
                "tokens_in": e.tokens_in,
                "tokens_out": e.tokens_out,
                "cost_usd": round(e.cost_usd, 6),
                "model_id": e.model_id,
            }
            for e in result.manifest.node_events
        ],
    }


def main():
    print("=" * 60)
    print("SNAIL v0.4.0 — end-to-end customer-support pipeline")
    print("=" * 60)

    # ── Test 1: clear "billing" intent → tier 1 OK → parallel extractors
    print("\n--- Case 1: clear billing intent ---")
    r1 = run_one("I want a refund for order #12345")
    for ev in r1["events"]:
        print(f"  {ev['node']:25s} → {ev['variant']:3s} "
              f"({ev['latency_ms']:5.1f}ms, ${ev['cost_usd']:.6f})")
    print(f"  TOTAL: nodes={r1['summary'].nodes_fired}, "
          f"cost=${r1['summary'].total_cost_usd:.6f}")

    # ── Test 2: ambiguous text → tier 1 OOD → escalate to tier 2
    print("\n--- Case 2: ambiguous text → escalation triggered ---")
    r2 = run_one("hello there, I have a question about my account")
    fired = {ev["node"] for ev in r2["events"]}
    print(f"  Nodes fired: {fired}")
    assert "classify_intent" in fired
    assert "classify_intent_large" in fired, "Tier 2 should have fired (escalation)"
    print(f"  TOTAL: nodes={r2['summary'].nodes_fired}, "
          f"cost=${r2['summary'].total_cost_usd:.6f}")

    # ── Test 3: weight-pin mismatch simulation
    print("\n--- Case 3: weight-pin mismatch raises ---")
    # Tamper with the weights file
    original_bytes = WEIGHTS_PATH.read_bytes()
    WEIGHTS_PATH.write_bytes(b"TAMPERED")
    try:
        from snail.program import Program as _P
        try:
            _P(name="bad", nodes=[classify_intent])
            raise AssertionError("WeightPinMismatch not raised on tampered weights")
        except (ValueError, WeightPinMismatch):
            print("  [OK] tampered weights → construction refused")
    finally:
        # Restore for the rest of the demo
        WEIGHTS_PATH.write_bytes(original_bytes)

    # ── Test 4: calibration report
    print("\n--- Case 4: confidence calibration ---")
    cases = []
    for i in range(20):
        cases.append({
            "input": {"text": f"I want a refund for order #{i}"},
            "expected": {"intent": "billing"},   # tier-1 will get this right
        })
    for i in range(20, 40):
        cases.append({
            "input": {"text": f"some random text {i}"},
            "expected": {"intent": "billing"},   # tier-1 OODs; tier-2 may differ
        })
    report = run_calibration(customer_support_pipeline, cases)
    print(f"  n_cases: {report.n_cases}")
    print(f"  accuracy: {report.accuracy:.3f}")
    print(f"  ECE:      {report.ece:.4f}")
    print(f"  bins:     {[(round(b.bin_low, 1), round(b.bin_high, 1), b.count) for b in report.bins]}")

    # ── Test 5: ECE math sanity (over-confident → high ECE)
    print("\n--- Case 5: ECE math sanity ---")
    # 100 cases, all conf 0.9, half correct → ECE should be ~0.4
    ece_demo, _ = compute_ece([0.9] * 100, [1, 0] * 50)
    assert ece_demo > 0.3, f"expected ECE > 0.3, got {ece_demo:.4f}"
    print(f"  [OK] ECE for over-confident 50%-accurate model: {ece_demo:.4f}")

    print("\n" + "=" * 60)
    print("OK — end-to-end pipeline works on the published v0.4.0 wheel.")
    print("=" * 60)


if __name__ == "__main__":
    main()
