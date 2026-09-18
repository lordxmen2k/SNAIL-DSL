"""Example 1: Invoice processing pipeline — pure SNAIL with frozen nodes.

Single forward pass through extract → classify → route → output.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from snail import node, Program, edge, NodeContext, NodeResult, OODSignal
from snail.wrappers import DeterministicNode


class InvoiceOk(BaseModel):
    total: float
    currency: str
    confidence: float


class Invoice(NodeResult):
    ok: InvoiceOk | None = None
    ood: OODSignal | None = None


class RiskOk(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    confidence: float


class Risk(NodeResult):
    ok: RiskOk | None = None
    ood: OODSignal | None = None


class DecisionOk(BaseModel):
    action: Literal["approve", "manual_review", "reject"]
    reason: str


class Decision(NodeResult):
    ok: DecisionOk | None = None
    ood: OODSignal | None = None


@node(
    name="extract_total",
    input_schema=dict,
    output_schema=Invoice,
    distribution="synthetic_invoices_v2",
    frozen_weights="",
    confidence_threshold=0.7,
)
def extract_total(ctx, weights, image):
    text = image.get("text", "")
    if not text:
        return Invoice(ok=InvoiceOk(total=0.0, currency="USD", confidence=0.4))
    return Invoice(ok=InvoiceOk(total=1247.50, currency="USD", confidence=0.92))


@node(
    name="classify_risk",
    input_schema=dict,
    output_schema=Risk,
    distribution="risk_v1",
    frozen_weights="",
    confidence_threshold=0.7,
)
def classify_risk(ctx, weights, invoice):
    total = invoice.total if hasattr(invoice, "total") else 0
    if total > 10000:
        return Risk(ok=RiskOk(risk_level="high", confidence=0.95))
    if total > 1000:
        return Risk(ok=RiskOk(risk_level="medium", confidence=0.88))
    return Risk(ok=RiskOk(risk_level="low", confidence=0.93))


decide = DeterministicNode(
    name="decide",
    input_schema=dict,
    output_schema=Decision,
    fn=lambda risk: Decision(ok=DecisionOk(
        action="approve" if getattr(risk, "risk_level", "low") == "low" else "manual_review",
        reason="risk_threshold",
    )),
    ood_on_none=False,
    distribution="deterministic",
)


human_review = DeterministicNode(
    name="human_review",
    input_schema=dict,
    output_schema=Decision,
    fn=lambda _: Decision(ok=DecisionOk(action="manual_review", reason="ood_routed")),
    ood_on_none=False,
    distribution="deterministic",
)


invoice_pipeline = Program(
    name="invoice_pipeline_v1",
    nodes=[extract_total, classify_risk, decide, human_review],
    edges=[
        edge(extract_total.ok),
        edge(classify_risk.ok),
        edge(extract_total.ood),
        edge(classify_risk.ood),
    ],
)


if __name__ == "__main__":
    result = invoice_pipeline.run({"text": "Invoice #12345 — Total: $1,247.50 USD"})
    print("Manifest:", result.manifest.to_json(indent=None))
    print("Outputs:")
    for n, out in result.outputs.items():
        v = "OOD" if out.is_ood else "OK"
        print(f"  {n:20s} → {v}")
    print("Final:", result.terminal.ok if result.terminal.is_ok else result.terminal.ood)
