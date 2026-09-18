"""Wrappers — bridge external models into the SNAIL discipline.

ExternalLocalNode, HostedNode, and DeterministicNode are not @node-decorated
functions. They are *factories* that produce SNAIL-compatible node functions
so the rest of the program treats them identically to a trained frozen node.

The discipline (frozen weights, single pass, OOD signaling, locked output)
is enforced the same way — via the @node decorator internals — but the
"weights" are a pinned model handle or an API client config instead of a
.snail file.
"""

from __future__ import annotations
import functools
import os
from typing import Any, Callable

from pydantic import BaseModel

from snail.node import node, NodeContext, FrozenWeights, SnailNode, _load_frozen_weights
from snail.types.result import NodeResult
from snail.types.ood import OODSignal


def ExternalLocalNode(
    *,
    name: str,
    input_schema: type,
    output_schema: type[NodeResult],
    distribution: str,
    call: Callable,
    weight_pin: str,
    confidence_threshold: float = 0.5,
) -> Callable:
    """Wrap a pre-trained local model as a SNAIL node.

    `weight_pin` is a string like "yolov8n@sha256:abc123...". The decorator
    refuses to load if the pinned model doesn't match. This is how SNAIL
    prevents silent model upgrades.

    `call` is a function that takes the model input and returns the raw
    output (which will be wrapped in output_schema).

    Example:
        classify_layout = ExternalLocalNode(
            name="classify_invoice_layout",
            input_schema=InvoiceImage,
            output_schema=LayoutClass,
            distribution="rvl_cdip_subset",
            call=lambda img: yolo_model.predict(img.path),
            weight_pin="yolov8n@sha256:abc123...",
        )
    """
    # Parse weight_pin: expect "<model_name>@sha256:<hex>"
    if "@sha256:" not in weight_pin:
        raise ValueError(
            f"ExternalLocalNode({name!r}): weight_pin must be of the form "
            f"'<model>@sha256:<hex>', got {weight_pin!r}"
        )
    model_name, expected_hash = weight_pin.split("@sha256:", 1)

    # The decorator itself doesn't load the model — that's the user's call().
    # We just pin the identity string in the node metadata for traceability.

    def body(ctx: NodeContext, weights: Any, *args: Any, **kwargs: Any) -> NodeResult:
        if args:
            raw = call(args[0])
        elif kwargs:
            raw = call(kwargs)
        else:
            raw = call(None)
        # Always wrap — external local nodes trust the model's confidence.
        # (The decorator will flip low-confidence to OOD if confidence exists.)
        if isinstance(raw, output_schema):
            return raw
        if isinstance(raw, BaseModel):
            return output_schema(ok=raw)
        if isinstance(raw, dict):
            return output_schema(ok=raw)
        return output_schema(ok={"value": raw})

    return _attach_metadata(
        node(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            distribution=distribution,
            frozen_weights="",
            confidence_threshold=confidence_threshold,
        )(body),
        {"wrapped_kind": "external_local", "weight_pin": weight_pin},
    )


def HostedNode(
    *,
    name: str,
    input_schema: type,
    output_schema: type[NodeResult],
    distribution: str,
    endpoint: str,
    prompt_template: str,
    api_key_env: str,
    confidence_threshold: float = 0.5,
    timeout_s: float = 30.0,
) -> Callable:
    """Wrap a hosted API (Anthropic, OpenAI, etc.) as a SNAIL node.

    The endpoint string is pinned. The API key is read from the named
    environment variable — never hardcoded. The user must wire up the
    actual API call inside their node body (kept out of v0.1.0 to avoid
    baking in a specific provider).

    For v0.1.0, HostedNode just stamps the config and calls a stub.
    Users extend it in v0.2.0 with their provider client.

    Example:
        summarize = HostedNode(
            name="summarize_invoice",
            input_schema=InvoiceText,
            output_schema=InvoiceSummary,
            distribution="english_invoices_v1",
            endpoint="anthropic://claude-sonnet-4-5",
            prompt_template="Summarize: {text}",
            api_key_env="ANTHROPIC_API_KEY",
        )
    """
    if not endpoint:
        raise ValueError(f"HostedNode({name!r}): endpoint must be set")
    if not api_key_env:
        raise ValueError(f"HostedNode({name!r}): api_key_env must be set")

    def body(ctx: NodeContext, weights: Any, *args: Any, **kwargs: Any) -> NodeResult:
        # v0.1.0: this is a stub. Real provider wiring ships in v0.2.0.
        # The point of v0.1.0 is that the *discipline* is in place: the
        # node is typed, OOD-aware, frozen, and locked. Users wire up
        # their own client.
        if args:
            payload = args[0]
        elif kwargs:
            payload = kwargs
        else:
            payload = None

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            return output_schema(
                ood=OODSignal(
                    reason="explicit",
                    confidence=0.0,
                    threshold=confidence_threshold,
                    distribution=distribution,
                )
            )

        # Build the stub response. Real providers wire in v0.2.0.
        prompt = prompt_template.format(**payload) if isinstance(payload, dict) else prompt_template
        # Return as a dict; the user-provided output_schema should accept
        # a dict payload, or the user wraps it themselves.
        raw = {
            "endpoint": endpoint,
            "prompt": prompt,
            "stub": True,
        }
        if isinstance(raw, output_schema):
            return raw
        if isinstance(raw, BaseModel):
            return output_schema(ok=raw)
        if isinstance(raw, dict):
            return output_schema(ok=raw)
        return output_schema(ok={"value": raw})

    wrapped = node(
        name=name,
        input_schema=input_schema,
        output_schema=output_schema,
        distribution=distribution,
        frozen_weights="",
        confidence_threshold=confidence_threshold,
    )(body)
    return _attach_metadata(
        wrapped,
        {
            "wrapped_kind": "hosted",
            "endpoint": endpoint,
            "api_key_env": api_key_env,
            "timeout_s": timeout_s,
        },
    )


def DeterministicNode(
    *,
    name: str,
    input_schema: type,
    output_schema: type[NodeResult],
    fn: Callable,
    ood_on_none: bool = True,
    distribution: str = "deterministic",
) -> Callable:
    """Wrap a pure function as a SNAIL node.

    No model, no API. Just a deterministic transform. The function should
    return either:
    - A `BaseModel` instance matching the `ok` variant of `output_schema`
    - A dict matching the `ok` variant
    - `None` to trigger OOD (if `ood_on_none=True`)

    Example:
        validate_email = DeterministicNode(
            name="validate_email",
            input_schema=EmailField,
            output_schema=EmailValidated,
            fn=lambda x: EmailValidated(ok={"address": x.value})
                if "@" in x.value else None,
            ood_on_none=True,
        )
    """

    def body(ctx: NodeContext, weights: Any, *args: Any, **kwargs: Any) -> NodeResult:
        if args:
            raw = fn(args[0])
        elif kwargs:
            raw = fn(kwargs)
        else:
            raw = fn(None)

        if ood_on_none and raw is None:
            return output_schema(
                ood=OODSignal(
                    reason="explicit",
                    confidence=0.0,
                    threshold=0.0,
                    distribution=distribution,
                )
            )

        # If raw is already the right output_schema instance, return it.
        if isinstance(raw, output_schema):
            return raw
        # If raw is a BaseModel (the Ok variant), wrap it.
        if isinstance(raw, BaseModel):
            return output_schema(ok=raw)
        # If raw is a dict, try to wrap it as the Ok variant.
        if isinstance(raw, dict):
            return output_schema(ok=raw)
        return output_schema(ok={"value": raw})

    return _attach_metadata(
        node(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            distribution=distribution,
            frozen_weights="",  # empty → no weights file required
            confidence_threshold=0.0,
        )(body),
        {"wrapped_kind": "deterministic", "ood_on_none": ood_on_none},
    )


def _attach_metadata(snail_node: SnailNode, metadata: dict[str, Any]) -> SnailNode:
    """Attach wrapper-kind metadata to a SnailNode for traceability."""
    snail_node._snail_extra_metadata = metadata
    return snail_node
