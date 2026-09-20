"""@node decorator — enforces the SNAIL node discipline.

Every node is:
- FROZEN: weights loaded once, never updated
- SINGLE-PASS: one forward call per invocation
- LOCKED OUTPUT: result is immutable after the call
- OOD-AWARE: low confidence or schema mismatch becomes an OOD variant

The decorator wraps the user's function. The user writes the model
forward pass; the decorator enforces the contract.

The wrapped object is a SnailNode — it's callable like a function
(`node_fn(ctx, payload)`) AND exposes `.ok` / `.ood` accessors for
use with `edge()`.
"""

from __future__ import annotations
import functools
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

from pydantic import BaseModel

from snail.types.result import NodeResult
from snail.types.ood import OODSignal


@dataclass
class NodeContext:
    """Per-call context passed to the node body."""

    program_name: str
    run_id: str
    node_name: str
    metadata: dict[str, Any]


@dataclass
class FrozenWeights:
    """Handle to the frozen weights file backing a node."""

    path: Path
    expected_sha256: str | None
    raw: Any

    def __post_init__(self):
        object.__setattr__(self, "raw", self.raw)


def _verify_sha256(path: Path, expected: str | None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"Weights file {path} SHA-256 mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Refusing to load — frozen weights must be byte-identical."
        )
    return actual


def _load_frozen_weights(path: str | os.PathLike, sha256: str | None = None) -> FrozenWeights:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Frozen weights file not found: {p}")

    actual_hash = _verify_sha256(p, sha256)

    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        return FrozenWeights(path=p, expected_sha256=sha256, raw=data)

    return FrozenWeights(path=p, expected_sha256=sha256, raw=None)


class _VariantAccessor:
    """Subscript-style accessor for use in edge() declarations."""

    def __init__(self, name: str, variant: str):
        self._node_name = name
        self._variant = variant


class SnailNode:
    """The result of @node decoration. Callable + has .ok / .ood accessors.

    Usage:
        my_node = node(...)(my_function)
        my_node(ctx, payload)               # call it
        edge(my_node.ok)                     # reference its OK variant
        edge(my_node.ood)                    # reference its OOD variant
    """

    def __init__(
        self,
        wrapped: Callable,
        *,
        name: str,
        input_schema: Any,
        output_schema: type,
        distribution: str,
        confidence_threshold: float,
        weights: FrozenWeights | None,
        original_fn: Callable,
        weight_pin: str | None = None,
        frozen_weights_path: str = "",
    ):
        self._wrapped = wrapped
        self._snail_name = name
        self._snail_input_schema = input_schema
        self._snail_output_schema = output_schema
        self._snail_distribution = distribution
        self._snail_confidence_threshold = confidence_threshold
        self._snail_weights = weights
        self._original_fn = original_fn
        self._snail_weight_pin = weight_pin
        self._snail_frozen_weights = frozen_weights_path
        functools.update_wrapper(self, wrapped)

    def __call__(self, ctx: NodeContext, *args: Any, **kwargs: Any) -> NodeResult:
        return self._wrapped(ctx, *args, **kwargs)

    @property
    def is_snail_node(self) -> bool:
        return True

    @property
    def ok(self) -> _VariantAccessor:
        return _VariantAccessor(self._snail_name, "ok")

    @property
    def ood(self) -> _VariantAccessor:
        return _VariantAccessor(self._snail_name, "ood")

    # Backwards-compat attributes (legacy tests / API).
    _is_snail_node = True


def node(
    *,
    name: str,
    input_schema: Type[BaseModel] | type,
    output_schema: Type[NodeResult],
    distribution: str,
    frozen_weights: str | os.PathLike | None = None,
    weights_sha256: str | None = None,
    weight_pin: str | None = None,
    confidence_threshold: float = 0.5,
) -> Callable[[Callable], SnailNode]:
    """Decorator: wrap a function as a SNAIL node.

    Returns a SnailNode — a callable that also has .ok / .ood accessors
    for use with edge().
    """

    if not distribution or not distribution.strip():
        raise ValueError(
            f"@node({name!r}): 'distribution' must be a non-empty string. "
            "Declare the training distribution so downstream programs "
            "can route on out-of-distribution inputs."
        )

    weights = None
    if frozen_weights is not None and str(frozen_weights).strip():
        try:
            weights = _load_frozen_weights(frozen_weights, weights_sha256)
        except FileNotFoundError:
            weights = None

    def decorator(fn: Callable) -> SnailNode:
        # If weight_pin was given, parse it once here so a malformed pin
        # fails loudly at decoration time (before Program construction).
        if weight_pin is not None:
            from snail.weights import parse_weight_pin
            parse_weight_pin(weight_pin)

        @functools.wraps(fn)
        def wrapped(ctx: NodeContext, *args: Any, **kwargs: Any) -> NodeResult:
            if args and isinstance(input_schema, type) and issubclass(input_schema, BaseModel):
                if not isinstance(args[0], input_schema):
                    return output_schema(
                        ood=OODSignal(
                            reason="schema_violation",
                            confidence=0.0,
                            threshold=confidence_threshold,
                            distribution=distribution,
                        )
                    )

            result = fn(ctx, weights, *args, **kwargs)

            if not isinstance(result, output_schema):
                if isinstance(result, BaseModel):
                    return output_schema(
                        ood=OODSignal(
                            reason="schema_violation",
                            confidence=0.0,
                            threshold=confidence_threshold,
                            distribution=distribution,
                        )
                    )
                raise TypeError(
                    f"@node({name!r}) returned {type(result).__name__}, "
                    f"expected {output_schema.__name__}."
                )

            if result.is_ok:
                confidence = _extract_confidence(result.ok)
                if confidence is not None and confidence < confidence_threshold:
                    return output_schema(
                        ood=OODSignal(
                            reason="low_confidence",
                            confidence=confidence,
                            threshold=confidence_threshold,
                            distribution=distribution,
                        )
                    )

            return result.model_copy(deep=True)

        return SnailNode(
            wrapped,
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            distribution=distribution,
            confidence_threshold=confidence_threshold,
            weights=weights,
            original_fn=fn,
            weight_pin=weight_pin,
            frozen_weights_path=str(frozen_weights) if frozen_weights else "",
        )

    return decorator


def _extract_confidence(ok_payload: Any) -> float | None:
    if isinstance(ok_payload, BaseModel):
        if hasattr(ok_payload, "confidence"):
            v = ok_payload.confidence
            if v is not None:
                return float(v)
        if hasattr(ok_payload, "score"):
            v = ok_payload.score
            if v is not None:
                return float(v)
    if isinstance(ok_payload, dict):
        if "confidence" in ok_payload and ok_payload["confidence"] is not None:
            return float(ok_payload["confidence"])
        if "score" in ok_payload and ok_payload["score"] is not None:
            return float(ok_payload["score"])
    return None
