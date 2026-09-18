"""Example: customer_support_v3 — a SNAIL program for a customer support agent.

The LLM (Claude) is just one node in this graph. It has a locked prompt
template, a declared output schema, and an OOD branch. It cannot decide
to skip the safety check or send the email directly — those edges don't exist.

Run:
    python examples/customer_support_v3.py
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail import node, Program, edge, NodeContext, NodeResult, OODSignal
from snail.wrappers import DeterministicNode, HostedNode


# ── Type definitions ────────────────────────────────────────────────────

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


class AccountOk(BaseModel):
    customer_id: str
    status: Literal["active", "suspended", "unknown"]
    confidence: float


class Account(NodeResult):
    ok: AccountOk | None = None
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
    action: Literal["sent", "queued", "human_review"]
    confirmation_id: str


class Action(NodeResult):
    ok: ActionOk | None = None
    ood: OODSignal | None = None


# ── Nodes ───────────────────────────────────────────────────────────────

@node(
    name="classify_intent",
    input_schema=dict,
    output_schema=Intent,
    distribution="customer_intents_v3",
    frozen_weights="weights/classify_intent_v3.snail.json",
    confidence_threshold=0.7,
)
def classify_intent(ctx, weights, message):
    """Classify the user's intent. Frozen weights, single forward pass."""
    text = message.get("text", "").lower()
    if "refund" in text or "billing" in text or "charge" in text:
        return Intent(ok=IntentOk(intent="billing", confidence=0.92))
    if "error" in text or "broken" in text or "bug" in text:
        return Intent(ok=IntentOk(intent="technical", confidence=0.88))
    return Intent(ok=IntentOk(intent="other", confidence=0.65))


@node(
    name="extract_entities",
    input_schema=dict,
    output_schema=Entities,
    distribution="customer_entities_v3",
    frozen_weights="weights/extract_entities_v3.snail.json",
    confidence_threshold=0.7,
)
def extract_entities(ctx, weights, intent_result):
    """Pull order_id and similar structured fields out of the message."""
    # In production: call weights.raw["model"].forward(intent_result.ok.message)
    return Entities(ok=EntitiesOk(order_id="ORD-12345", confidence=0.91))


@node(
    name="check_account_status",
    input_schema=dict,
    output_schema=Account,
    distribution="account_status_v3",
    frozen_weights="weights/check_account_status_v3.snail.json",
    confidence_threshold=0.6,
)
def check_account_status(ctx, weights, entities):
    """Look up the customer's account status. Stub for the example."""
    return Account(ok=AccountOk(
        customer_id="CUST-987",
        status="active",
        confidence=0.95,
    ))


# Stub HostedNode — the user wires real provider client in v0.2.0.
draft_response = HostedNode(
    name="draft_response",
    input_schema=dict,
    output_schema=Draft,
    distribution="customer_response_tones_v1",
    endpoint="anthropic://claude-sonnet-4-5",
    prompt_template="You are a customer support agent. Tone: friendly. "
                    "Account: {account}. Intent: {intent}. Entities: {entities}. "
                    "Write a brief, helpful reply. Output JSON with: "
                    "message (str), tone (friendly|neutral|formal).",
    api_key_env="ANTHROPIC_API_KEY",
    confidence_threshold=0.6,
)


@node(
    name="safety_check",
    input_schema=dict,
    output_schema=Safety,
    distribution="response_safety_v1",
    frozen_weights="weights/safety_check_v1.snail.json",
    confidence_threshold=0.7,
)
def safety_check(ctx, weights, draft):
    """Run the draft through the safety classifier. Always called — no LLM skip.

    `draft` here is the OK payload (a DraftOk instance), not the full Draft
    result. The runtime unwraps the variant before passing downstream.
    """
    msg = draft.message if hasattr(draft, "message") else str(draft)
    flags = []
    if "ssn" in msg.lower() or "password" in msg.lower():
        flags.append("sensitive_data_mention")
    return Safety(ok=SafetyOk(
        safe=len(flags) == 0,
        flags=flags,
        confidence=0.97,
    ))


send_response = DeterministicNode(
    name="send_response",
    input_schema=dict,
    output_schema=Action,
    fn=lambda draft: Action(ok=ActionOk(
        action="sent",
        confirmation_id="CONF-" + str(getattr(draft, "message", draft))[:8],
    )),
    ood_on_none=False,
    distribution="deterministic",
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
    distribution="deterministic",
)


# ── Program ─────────────────────────────────────────────────────────────

customer_support_v3 = Program(
    name="customer_support_v3",
    nodes=[
        classify_intent,
        extract_entities,
        check_account_status,
        draft_response,
        safety_check,
        send_response,
        human_review,
    ],
    edges=[
        # OK paths
        edge(classify_intent.ok, target_field="message"),
        edge(extract_entities.ok, target_field="entities_result"),
        edge(check_account_status.ok, target_field="account"),
        edge(draft_response.ok, target_field="draft"),
        edge(safety_check.ok, target_field="draft"),

        # OOD paths — anything uncertain goes to human review
        edge(classify_intent.ood),
        edge(extract_entities.ood),
        edge(check_account_status.ood),
        edge(draft_response.ood),
        edge(safety_check.ood),
    ],
)


def run_demo():
    print("=" * 60)
    print("SNAIL customer_support_v3 — end-to-end demo")
    print("=" * 60)

    # The LLM node is a stub because we don't have a real API key in this demo.
    # We bypass HostedNode by directly invoking the rest of the graph with
    # a pre-computed draft. To do this, we use a small "demo mode" that
    # replaces draft_response with a deterministic stub.

    # Replace the LLM with a deterministic stub for the demo
    # (real users would keep the HostedNode).
    draft_stub = DeterministicNode(
        name="draft_response_stub",
        input_schema=dict,
        output_schema=Draft,
        fn=lambda ctx_dict: Draft(ok=DraftOk(
            message="Hi! I see your refund request for order ORD-12345. "
                    "I'll process it now and email you a confirmation.",
            tone="friendly",
            confidence=0.93,
        )),
        ood_on_none=False,
        distribution="demo_stub",
    )

    # Build a demo program with the stub
    demo_program = Program(
        name="customer_support_v3_demo",
        nodes=[
            classify_intent,
            extract_entities,
            check_account_status,
            draft_stub,
            safety_check,
            send_response,
            human_review,
        ],
        edges=[
            edge(classify_intent.ok, target_field="message"),
            edge(extract_entities.ok, target_field="entities_result"),
            edge(check_account_status.ok, target_field="account"),
            edge(draft_stub.ok, target_field="draft"),
            edge(safety_check.ok, target_field="draft"),
            edge(classify_intent.ood),
            edge(extract_entities.ood),
            edge(check_account_status.ood),
            edge(draft_stub.ood),
            edge(safety_check.ood),
        ],
    )

    result = demo_program.run({"text": "I want a refund for order ORD-12345, please."})

    print("\n── Manifest ──")
    print(result.manifest.to_json())

    print("\n── Outputs ──")
    for name, out in result.outputs.items():
        variant = "OOD" if out.is_ood else "OK"
        print(f"  {name:25s} → {variant}")

    print("\n── Final action ──")
    if result.terminal and result.terminal.is_ok:
        print(f"  {result.terminal.ok}")
    elif result.terminal:
        print(f"  Routed to human review: {result.terminal.ood}")

    return result


if __name__ == "__main__":
    run_demo()
