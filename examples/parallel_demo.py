"""Example: parallel_demo — fan-out extraction with parallel_edges().

Three extractors run on the same input. Their OK results feed downstream
as separate fields.

Demonstrates the v0.3.0 `parallel_edges()` primitive.

Run:
    python examples/parallel_demo.py
"""

from __future__ import annotations
from pydantic import BaseModel

from snail import (
    Program,
    edge,
    parallel_edges,
    NodeResult,
    OODSignal,
)
from snail.wrappers import DeterministicNode


# ── Schemas ──────────────────────────────────────────────────────────────


class ExtractedOk(BaseModel):
    value: str
    confidence: float


class Extracted(NodeResult):
    ok: ExtractedOk | None = None
    ood: OODSignal | None = None


class ContactOk(BaseModel):
    email: str
    phone: str
    address: str
    confidence: float


class Contact(NodeResult):
    ok: ContactOk | None = None
    ood: OODSignal | None = None


# ── Nodes ───────────────────────────────────────────────────────────────


raw = DeterministicNode(
    name="raw",
    input_schema=dict,
    output_schema=Extracted,
    fn=lambda x: Extracted(ok=ExtractedOk(value="raw_text", confidence=1.0)),
)

extract_email = DeterministicNode(
    name="extract_email",
    input_schema=dict,
    output_schema=Extracted,
    fn=lambda x: Extracted(ok=ExtractedOk(value="alice@example.com", confidence=0.96)),
)

extract_phone = DeterministicNode(
    name="extract_phone",
    input_schema=dict,
    output_schema=Extracted,
    fn=lambda x: Extracted(ok=ExtractedOk(value="+1-555-0100", confidence=0.91)),
)

extract_address = DeterministicNode(
    name="extract_address",
    input_schema=dict,
    output_schema=Extracted,
    fn=lambda x: Extracted(ok=ExtractedOk(value="123 Main St", confidence=0.88)),
)

merge = DeterministicNode(
    name="merge",
    input_schema=dict,
    output_schema=Contact,
    fn=lambda x: Contact(ok=ContactOk(
        email="alice@example.com",
        phone="+1-555-0100",
        address="123 Main St",
        confidence=0.9,
    )),
)


# ── Program ─────────────────────────────────────────────────────────────


parallel_demo = Program(
    name="parallel_demo",
    nodes=[raw, extract_email, extract_phone, extract_address, merge],
    edges=[
        # Each parallel extractor feeds the merge as a separate field.
        edge(extract_email.ok, target_field="email_result"),
        edge(extract_phone.ok, target_field="phone_result"),
        edge(extract_address.ok, target_field="address_result"),
    ],
    parallel_groups=[
        parallel_edges(raw, to=[extract_email, extract_phone, extract_address]),
    ],
)


def run_demo():
    print("=" * 60)
    print("SNAIL parallel_demo — three extractors on the same input")
    print("=" * 60)

    result = parallel_demo.run({"text": "Alice lives at 123 Main St, phone +1-555-0100."})

    print("\n── Manifest summary ──")
    print(f"  nodes fired: {result.manifest.summary.nodes_fired}")

    print("\n── Per-node events ──")
    for ev in result.manifest.node_events:
        print(f"  {ev.node_name:25s} → {ev.variant:3s}")


if __name__ == "__main__":
    run_demo()
