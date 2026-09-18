# SNAIL — Single Node Activated Inference Layer

A Python DSL for composing **frozen, single-pass neural primitives** into **statically-typed dataflow programs**.

If an MCP tool and a markdown skill had a baby, and you told the LLM it wasn't allowed to interpret the recipe — it was just one of many nodes in the recipe — you'd get SNAIL.

## Why

Most AI models are trained to be comprehensive. SNAIL trains tiny, single-task nodes — **activated exactly once**, **locked output**, **frozen forever** — and composes them into deterministic graphs.

A SNAIL program is a **recipe that's more resilient than a `.md` spec** and **more deterministic than an MCP tool call**. Every step is a typed function call, not a prompt the LLM might re-interpret. Every run emits a manifest. Every node knows what it doesn't know (OOD is a first-class type).

## Install

```bash
pip install snail-dsl
```

## Quick start

```python
from snail import node, Program, edge
from pydantic import BaseModel

# Declare the type contract for each node's output.
class InvoiceTotal(BaseModel):
    ok: dict | None = None       # { "total": float, "currency": str, "confidence": float }
    ood: dict | None = None      # { "reason": str, "confidence": float, "threshold": float }

@node(
    name="extract_invoice_total",
    input_schema=dict,            # replace with a real pydantic model
    output_schema=InvoiceTotal,
    distribution="synthetic_invoices_v2",
    frozen_weights="weights/extract_invoice_total_v3.snail",
    confidence_threshold=0.7,
)
def extract_invoice_total(ctx, weights, image):
    raw = weights.model.forward(image["tensor"])
    return InvoiceTotal(ok={
        "total": raw["total"],
        "currency": raw["currency"],
        "confidence": raw["confidence"],
    })

# Compose into a typed DAG.
invoice_pipeline = Program(
    name="invoice_pipeline_v1",
    nodes=[extract_invoice_total],
    edges=[],
)

result = invoice_pipeline.run({"image_path": "scan.png"})
print(result.manifest)
```

The decorator enforces:

- Single forward pass per call — no retries, no second thoughts.
- Weights loaded once, frozen for the lifetime of the program.
- Output type-checked against `output_schema`; mismatches become OOD.
- Confidence below `confidence_threshold` is silently flipped to OOD.
- Output is locked (immutable) before being handed back to the caller.

## The four primitives

| Primitive | What it does |
|---|---|
| `@node` | Decorator that wraps a Python function as a frozen, single-pass, OOD-aware node. |
| `Program` | Container that builds a typed DAG of nodes. Validates at construction time. |
| `edge()` | Builder for typed field-to-field connections between nodes. |
| `manifest` | Structured per-run log: which nodes fired, what they got, what they returned. |

## Architecture

- **Nodes** are frozen, single forward pass, locked output.
- **Composition** is a static DAG declared in Python.
- **Training** is 100% synthetic, offline.
- **OOD** is a first-class type (Ok / OOD discriminated union).
- **Every node ships with golden tests** (`pytest`).
- **Every run emits a manifest.**

## Wrapping external models

External models (HuggingFace, hosted APIs, pure functions) become nodes through wrappers. They never get called directly.

```python
from snail.wrappers import ExternalLocalNode, HostedNode, DeterministicNode

# Pin a local model to exact weights hash
classify_layout = ExternalLocalNode(
    name="classify_layout",
    input_schema=InvoiceImage,
    output_schema=LayoutClass,
    distribution="rvl_cdip_subset",
    call=lambda img: yolo_model.predict(img.path),
    weight_pin="yolov8n@sha256:abc123...",
)

# Wrap a hosted API
summarize = HostedNode(
    name="summarize",
    input_schema=InvoiceText,
    output_schema=InvoiceSummary,
    endpoint="anthropic://claude-sonnet-4-5",
    prompt_template="Summarize: {text}",
    api_key_env="ANTHROPIC_API_KEY",
)

# Or just a pure function
validate_email = DeterministicNode(
    name="validate_email",
    input_schema=EmailField,
    output_schema=EmailValidated,
    fn=lambda x: x if "@" in x.value else None,
    ood_on_none=True,
)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Golden tests live in `tests/golden/`. The lint rule (catches direct `import openai`, `import anthropic`, etc. outside wrappers) is enforced via the `snail.lint` module.

## License

Apache License 2.0. Copyright 2026 Tico Internet LLC.

## Status

v0.1.0 — alpha. The core primitives (`@node`, `Program`, `edge()`, wrappers, manifest, lint) are working. The training pipeline for producing `.snail` weights is not yet included — that ships in v0.2.0.
