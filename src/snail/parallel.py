"""Parallel fan-out primitive — multiple nodes receive the same input.

`parallel_edges(inputs=node, to=[list_of_nodes])` declares that all
nodes in `to` receive the output of `inputs` as their input. They run
in topological order, but the framework treats the group as a unit:
the manifest records the parallel_group identifier, the runner executes
the members sequentially (asyncio fan-out is a v0.4.0 candidate).

This is the third workflow primitive in the v0.3.0 release, completing
the chaining + routing + parallelization set the v2.0 book claims SNAIL
has.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any

from snail.program import Edge, _NodeVariantAccessor
from snail.node import _VariantAccessor

_SourceAccessor = (_NodeVariantAccessor, _VariantAccessor)


@dataclass
class ParallelGroup:
    """Declared fan-out: `inputs` → `to` (multiple nodes, same input)."""

    inputs_node: str
    target_nodes: list[str]
    target_field: str = "input"
    group_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_edges(self) -> list[Edge]:
        """Convert this group into a list of Edge objects."""
        return [
            Edge(
                source_node=self.inputs_node,
                source_variant="ok",
                target_node=t,
                target_field=self.target_field,
            )
            for t in self.target_nodes
        ]


def parallel_edges(
    inputs: object,
    *,
    to: list[object],
    target_field: str = "input",
) -> ParallelGroup:
    """Declare a parallel fan-out.

    Args:
        inputs: A SNAIL node (or its `.ok` accessor) that produces the
            shared input for the parallel group.
        to: A list of nodes that will all receive the same input.
        target_field: The input field name on each target node.

    Returns:
        ParallelGroup — consumed by Program at construction.
    """
    # Accept either `node` or `node.ok` as the input source.
    if isinstance(inputs, _SourceAccessor):
        if inputs._variant != "ok":
            raise ValueError(
                "parallel_edges(): inputs must be a node or a .ok accessor "
                f"(got .{inputs._variant!r})"
            )
        inputs_name = inputs._node_name
    elif hasattr(inputs, "_snail_name"):
        inputs_name = inputs._snail_name
    elif hasattr(inputs, "name"):
        inputs_name = inputs.name
    else:
        raise TypeError(
            f"parallel_edges(): inputs must be a SNAIL node "
            f"(got {type(inputs).__name__})"
        )

    if not to:
        raise ValueError("parallel_edges(): `to` must be a non-empty list")

    target_names: list[str] = []
    for t in to:
        if hasattr(t, "_snail_name"):
            target_names.append(t._snail_name)
        elif hasattr(t, "name"):
            target_names.append(t.name)
        else:
            raise TypeError(
                f"parallel_edges(): target must be a SNAIL node "
                f"(got {type(t).__name__})"
            )

    if len(set(target_names)) != len(target_names):
        raise ValueError(
            "parallel_edges(): `to` must not contain duplicate nodes"
        )

    return ParallelGroup(
        inputs_node=inputs_name,
        target_nodes=target_names,
        target_field=target_field,
    )
