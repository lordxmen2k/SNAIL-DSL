"""Golden test example: extract_invoice_total node.

This is the canonical "first node" — shows the full discipline:
frozen weights, declared distribution, schema-locked output, OOD signaling.

Golden tests pin the expected behavior of a trained node. When you
retrain and ship new weights, run these to confirm nothing drifted.
"""

from pathlib import Path
from pydantic import BaseModel
import pytest

from snail import node, NodeContext, NodeResult, OODSignal


class InvoiceTotalOk(BaseModel):
    total: float
    currency: str
    confidence: float


class InvoiceTotal(NodeResult):
    ok: InvoiceTotalOk | None = None
    ood: OODSignal | None = None


@pytest.fixture
def fake_weights(tmp_path):
    """Write a tiny JSON sidecar weights file for testing."""
    p = tmp_path / "extract_invoice_total_v3.snail.json"
    p.write_text(
        '{"model_id": "extract_invoice_total_v3", '
        '"trained_on": "synthetic_invoices_v2", '
        '"weights": [0.1, 0.2, 0.3]}'
    )
    return p


def _build_node(weights_path):
    """Build the node with a given weights path."""

    def body(ctx, weights, image):
        # Stub forward pass — in production this would call
        # weights.raw["model"].forward(image).
        return InvoiceTotal(
            ok=InvoiceTotalOk(
                total=1247.50,
                currency="USD",
                confidence=0.92,
            )
        )

    return node(
        name="extract_invoice_total",
        input_schema=dict,
        output_schema=InvoiceTotal,
        distribution="synthetic_invoices_v2",
        frozen_weights=weights_path,
        confidence_threshold=0.7,
    )(body)


def test_golden_invoice_total_happy_path(fake_weights):
    """OK case: high confidence → result.ok populated, result.ood None."""
    fn = _build_node(fake_weights)
    ctx = NodeContext(
        program_name="t", run_id="g1",
        node_name="extract_invoice_total", metadata={},
    )
    result = fn(ctx, {"image_path": "fixtures/invoice_001.png"})

    assert result.is_ok
    assert result.ok.currency == "USD"
    assert abs(result.ok.total - 1247.50) < 0.01
    assert result.ok.confidence > 0.7


def test_golden_invoice_total_ood_when_low_confidence(fake_weights):
    """OOD case: confidence below threshold flips to OOD."""

    def body(ctx, weights, image):
        return InvoiceTotal(
            ok=InvoiceTotalOk(total=0.0, currency="USD", confidence=0.3)
        )

    fn = node(
        name="extract_invoice_total_lowconf",
        input_schema=dict,
        output_schema=InvoiceTotal,
        distribution="synthetic_invoices_v2",
        frozen_weights=fake_weights,
        confidence_threshold=0.7,
    )(body)

    ctx = NodeContext(
        program_name="t", run_id="g2",
        node_name="extract_invoice_total_lowconf", metadata={},
    )
    result = fn(ctx, {"image_path": "garbled.png"})

    assert result.is_ood
    assert result.ood.reason == "low_confidence"
    assert result.ood.threshold == 0.7
    assert result.ood.confidence == 0.3


def test_golden_invoice_total_explicit_ood(fake_weights):
    """Explicit OOD return passes through unchanged."""

    def body(ctx, weights, image):
        return InvoiceTotal(
            ood=OODSignal(
                reason="explicit",
                confidence=0.0,
                threshold=0.7,
                distribution="synthetic_invoices_v2",
            )
        )

    fn = node(
        name="extract_invoice_total_explicit",
        input_schema=dict,
        output_schema=InvoiceTotal,
        distribution="synthetic_invoices_v2",
        frozen_weights=fake_weights,
    )(body)

    ctx = NodeContext(
        program_name="t", run_id="g3",
        node_name="extract_invoice_total_explicit", metadata={},
    )
    result = fn(ctx, {"image_path": "anything.png"})

    assert result.is_ood
    assert result.ood.reason == "explicit"
    assert result.ood.distribution == "synthetic_invoices_v2"


def test_golden_invoice_total_weights_metadata(fake_weights):
    """The frozen weights metadata should be accessible to the node body."""

    captured = {}

    def body(ctx, weights, image):
        captured["model_id"] = weights.raw.get("model_id")
        captured["trained_on"] = weights.raw.get("trained_on")
        return InvoiceTotal(
            ok=InvoiceTotalOk(total=1.0, currency="USD", confidence=1.0)
        )

    fn = node(
        name="metadata_check",
        input_schema=dict,
        output_schema=InvoiceTotal,
        distribution="synthetic_invoices_v2",
        frozen_weights=fake_weights,
    )(body)

    ctx = NodeContext(
        program_name="t", run_id="g4",
        node_name="metadata_check", metadata={},
    )
    fn(ctx, {"image_path": "x.png"})

    assert captured["model_id"] == "extract_invoice_total_v3"
    assert captured["trained_on"] == "synthetic_invoices_v2"
