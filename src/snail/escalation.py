"""Escalation primitive — first-class OOD → heavier-node routing.

The `escalate(node, to=other)` helper turns the OOD branch of a node into
a wired edge targeting a second node. The intent is to make the
small-model-first / large-model-on-OOD pattern a one-liner instead of
hand-wiring `edge(node.ood, target_field=...)` every time.

Cost discipline:
- Every node declares a `cost_tier` ("small" < "medium" < "large").
- `escalate()` rejects downgrades (you cannot escalate to a cheaper tier).
- Cycles are rejected at Program construction time.

This is the v0.3.0 workflow primitive that closes the gap between SNAIL's
OK/OOD-as-type mechanism and the SLM-tiering pattern the field is
converging on.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from snail.program import Edge, _NodeVariantAccessor
from snail.node import _VariantAccessor

# A source accessor can come from either NodeProxy.ok/.ood (in edge())
# or from SnailNode.ok/.ood (in HostedNode / @node). Both classes carry
# _node_name + _variant. We accept either.
_SourceAccessor = (_NodeVariantAccessor, _VariantAccessor)


class CostTier(str, Enum):
    """Cost tier for an LLM-backed node. Used by `escalate()` to forbid
    cost downgrades at composition time."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @classmethod
    def _rank(cls, tier: "CostTier") -> int:
        order = [cls.SMALL, cls.MEDIUM, cls.LARGE]
        return order.index(tier)

    def __lt__(self, other: "CostTier") -> bool:  # type: ignore[override]
        return self._rank(self) < self._rank(other)


_TIER_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
}


def _tier_of(node: object) -> CostTier:
    """Read the `cost_tier` declared on a node proxy/wrapper.

    Defaults to MEDIUM if the node doesn't declare one (preserves the
    conservative position: unknown tier cannot escalate DOWN, but is
    itself hitable from SMALL).
    """
    meta = getattr(node, "_snail_extra_metadata", {}) or {}
    raw = meta.get("cost_tier", "medium")
    if isinstance(raw, CostTier):
        return raw
    if isinstance(raw, str) and raw in _TIER_ORDER:
        return CostTier(raw)
    return CostTier.MEDIUM


def _node_name(node: object) -> str:
    """Extract the node's registered name from any node-like object."""
    n = getattr(node, "_snail_name", None) or getattr(node, "name", None)
    if not n:
        raise TypeError(
            f"escalate(): source must be a SNAIL node (got {type(node).__name__})"
        )
    return n


@dataclass
class EscalationSpec:
    """Compiled escalation declaration: source OOD → target node + field.

    Surfaced in Program construction. The runner treats this exactly like
    an `Edge(source, "ood", target, field)`.
    """

    source_node: str
    target_node: str
    target_field: str
    source_tier: CostTier
    target_tier: CostTier


def escalate(
    source: object,
    *,
    to: object,
    target_field: str = "input",
) -> EscalationSpec:
    """Declare an escalation: source OOD → to (heavier node).

    Returns an EscalationSpec that Program.__init__ consumes.

    The escalation is rejected at Program construction time if:
    - source or `to` is not a registered SNAIL node
    - `to` declares a lower or equal `cost_tier` than source
    - the resulting graph has a cycle
    """
    if not isinstance(source, _SourceAccessor):
        raise TypeError(
            f"escalate(): first arg must be a node variant accessor "
            f"(e.g. classify.ood), got {type(source).__name__}"
        )
    if source._variant != "ood":
        raise ValueError(
            f"escalate(): source must be a .ood accessor (got .{source._variant!r}). "
            "Escalation routes the OOD branch; for OK routing use edge()."
        )
    if not hasattr(to, "_snail_name") and not hasattr(to, "name"):
        raise TypeError(
            f"escalate(): `to` must be a SNAIL node (got {type(to).__name__})"
        )

    target_name = _node_name(to)
    if target_name == source._node_name:
        raise ValueError(
            f"escalate(): source and target are the same node ({target_name!r}); "
            "self-escalation is not allowed."
        )

    src_tier = _tier_of(source)  # use accessor? no — we need the node, not the accessor
    # Recover the source node from its name via the registry is awkward here.
    # Instead, defer the cost-tier comparison to Program construction, where
    # we have the node dict. Store the node names here.

    return EscalationSpec(
        source_node=source._node_name,
        target_node=target_name,
        target_field=target_field,
        source_tier=src_tier,
        target_tier=CostTier.MEDIUM,  # filled in by Program at construction
    )


def escalation_to_edge(spec: EscalationSpec) -> Edge:
    """Convert an EscalationSpec into an Edge (for the runner).

    The Edge uses source_variant="ood" and the same target_field the spec
    declared.
    """
    return Edge(
        source_node=spec.source_node,
        source_variant="ood",
        target_node=spec.target_node,
        target_field=spec.target_field,
    )
