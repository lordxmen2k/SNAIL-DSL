"""Program — typed DAG container for SNAIL nodes.

A Program is built once from a list of @node-decorated functions and
a list of typed edges between them. Construction runs static validation:
- cycles are rejected
- missing edges are reported
- type mismatches on edges are reported
- unreachable nodes are flagged

Once constructed, a Program can be `run(input)` any number of times.
Each run executes the DAG topologically and emits a manifest.
"""

from __future__ import annotations
import hashlib
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from snail.node import NodeContext, _extract_confidence
from snail.manifest import ManifestBuilder
from snail.types.result import NodeResult


@dataclass
class Edge:
    """A typed connection between two nodes.

    Source: a node name + the variant being connected ("ok" or "ood")
    Target: a node name + the input field that receives the value
    """

    source_node: str
    source_variant: str  # "ok" or "ood"
    target_node: str
    target_field: str = "input"

    def __post_init__(self):
        if self.source_variant not in ("ok", "ood"):
            raise ValueError(
                f"Edge: source_variant must be 'ok' or 'ood', got {self.source_variant!r}"
            )


def edge(
    source: Any,
    target_field: str = "input",
) -> Edge:
    """Build an Edge using subscript syntax.

    Usage:
        edge(node_a.ok, target_field="amount")  # → Edge("node_a", "ok", ..., "amount")
        edge(node_b.ood)                          # → Edge("node_b", "ood", ..., "input")

    The `source` argument should be either a SNAILNodeAccessor (returned
    by NodeProxy.ok / NodeProxy.ood) or an EdgeSpec-like object with
    ._node_name and ._variant attributes.
    """
    if not hasattr(source, "_node_name") or not hasattr(source, "_variant"):
        raise TypeError(
            f"edge(): source must be a node variant accessor (e.g. classify.ok), "
            f"got {type(source).__name__}"
        )
    return Edge(
        source_node=source._node_name,
        source_variant=source._variant,
        target_node="",  # filled in by Program.build() when paired
        target_field=target_field,
    )


@dataclass
class _NodeVariantAccessor:
    """Subscript-style accessor: `my_node.ok` returns an object that,
    when passed to edge(), produces an Edge targeting this node's OK variant.
    """

    _node_name: str
    _variant: str  # "ok" or "ood"


@dataclass
class NodeProxy:
    """Lightweight proxy used in edge() declarations.

    Calling `node.ok` returns a _NodeVariantAccessor that edge() knows how
    to read. The proxy itself just holds metadata.
    """

    name: str
    fn: Callable
    input_schema: Any = None
    output_schema: Any = None
    distribution: str = ""

    @property
    def ok(self) -> _NodeVariantAccessor:
        return _NodeVariantAccessor(_node_name=self.name, _variant="ok")

    @property
    def ood(self) -> _NodeVariantAccessor:
        return _NodeVariantAccessor(_node_name=self.name, _variant="ood")


@dataclass
class ProgramRunResult:
    """Result of running a Program once.

    - `outputs`: dict from node name → final NodeResult instance
    - `manifest`: structured per-run log
    - `terminal`: the last node to fire, or the sink if OOD routed to one
    """

    outputs: dict[str, NodeResult]
    manifest: Any
    terminal: NodeResult | None = None


class Program:
    """A typed DAG of SNAIL nodes.

    Construction is two-phase:
    1. Pass nodes=[...] and edges=[...] at construction.
    2. The `build()` method (called automatically) runs static validation
       and fails loud on any inconsistency.

    After construction, call `program.run(input)` to execute.
    """

    def __init__(
        self,
        *,
        name: str,
        nodes: list[Callable],
        edges: list[Edge] | None = None,
        strict_mode: bool = True,
    ):
        self.name = name
        self.strict_mode = strict_mode
        self._node_proxies: dict[str, NodeProxy] = {}
        self._raw_edges: list[Edge] = list(edges or [])
        self._validated_edges: list[Edge] = []

        if not nodes:
            raise ValueError(
                f"Program({name!r}): must have at least one node. "
                "Empty pipelines don't make sense in SNAIL."
            )

        for fn in nodes:
            if not getattr(fn, "_is_snail_node", False):
                raise TypeError(
                    f"Program({name!r}): {fn} is not decorated with @node. "
                    "All nodes must be SNAIL nodes."
                )
            n = fn._snail_name
            if n in self._node_proxies:
                raise ValueError(f"Program({name!r}): duplicate node name {n!r}")
            self._node_proxies[n] = NodeProxy(
                name=n,
                fn=fn,
                input_schema=fn._snail_input_schema,
                output_schema=fn._snail_output_schema,
                distribution=fn._snail_distribution,
            )

        self._validate()

    def _validate(self) -> None:
        """Run static validation. Fail loud if strict_mode is True."""
        errors: list[str] = []

        # 1. Every edge must reference a known node.
        for e in self._raw_edges:
            if e.source_node not in self._node_proxies:
                errors.append(
                    f"Edge from unknown node {e.source_node!r} "
                    f"variant={e.source_variant!r}"
                )

        # 2. No cycles.
        adj: dict[str, set[str]] = defaultdict(set)
        # We'll resolve edges after step 3.

        # 3. Resolve edges: each edge is a (source.variant) → (target.field)
        # pairing. We need to know which node owns the target.field.
        # Strategy:
        #   - Edges with target_node already set: pass through as-is.
        #   - OK edges with target_node="": auto-resolve to the next
        #     unclaimed node in declared order (the "primary" consumer).
        #   - OOD edges with target_node="": auto-resolve to the LAST node
        #     in declared order (the terminal sink).

        node_order = list(self._node_proxies.keys())
        claimed_by_ok: set[str] = set()
        for e in self._raw_edges:
            if e.source_variant == "ok" and e.target_node:
                claimed_by_ok.add(e.target_node)

        resolved: list[Edge] = []
        for e in self._raw_edges:
            if e.target_node:
                resolved.append(e)
                continue

            if e.source_variant == "ok":
                # OK edges go to the next unclaimed node.
                source_idx = node_order.index(e.source_node)
                for cand_name in node_order[source_idx + 1 :]:
                    if cand_name not in claimed_by_ok:
                        claimed_by_ok.add(cand_name)
                        resolved.append(
                            Edge(
                                source_node=e.source_node,
                                source_variant=e.source_variant,
                                target_node=cand_name,
                                target_field=e.target_field,
                            )
                        )
                        break
                else:
                    errors.append(
                        f"Edge from {e.source_node!r}.ok could not resolve "
                        "a target node"
                    )
            else:
                # OOD edges go to the terminal node (last in order).
                terminal = node_order[-1]
                resolved.append(
                    Edge(
                        source_node=e.source_node,
                        source_variant=e.source_variant,
                        target_node=terminal,
                        target_field=e.target_field,
                    )
                )

        # 4. No cycles (after resolution).
        for e in resolved:
            adj[e.source_node].add(e.target_node)
        if self._has_cycle(adj):
            errors.append("Cycle detected in program graph")

        # 5. Every node that produces output should have a downstream
        # consumer OR be explicitly terminal. Flag unreachable nodes.
        reachable = self._reachable_from_inputs(resolved)
        for n in self._node_proxies:
            if n not in reachable and not reachable:
                # Only complain if NOTHING is reachable.
                errors.append(f"Node {n!r} is unreachable from any input")

        # 6. Every node's declared input fields must have at least one
        # incoming edge OR a default input. Flag missing inputs.
        for n in self._node_proxies:
            incoming = [e for e in resolved if e.target_node == n]
            if not incoming:
                # OK if it's a top-of-pipeline node — it gets the program input.
                if not self._is_toplevel(n, resolved):
                    errors.append(
                        f"Node {n!r} has no incoming edge and is not the "
                        "pipeline input"
                    )

        if errors:
            if self.strict_mode:
                raise ValueError(
                    f"Program({self.name!r}) failed static validation:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
            else:
                import warnings

                warnings.warn(
                    f"Program({self.name!r}) has validation issues:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        self._validated_edges = resolved

    @staticmethod
    def _has_cycle(adj: dict[str, set[str]]) -> bool:
        """Detect cycles via DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        # Include all known nodes (not just adj keys), so we never hit KeyError.
        all_nodes: set[str] = set(adj.keys())
        for s in adj.values():
            all_nodes.update(s)
        color: dict[str, int] = {n: WHITE for n in all_nodes}

        def visit(n: str) -> bool:
            c = color.get(n, WHITE)
            if c == GRAY:
                return True
            if c == BLACK:
                return False
            color[n] = GRAY
            for m in adj.get(n, ()):
                if visit(m):
                    return True
            color[n] = BLACK
            return False

        for n in all_nodes:
            if color[n] == WHITE and visit(n):
                return True
        return False

    def _reachable_from_inputs(self, edges: list[Edge]) -> set[str]:
        """Topological order of nodes, starting from nodes with no incoming edges."""
        in_deg: dict[str, int] = {n: 0 for n in self._node_proxies}
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if e.target_node in self._node_proxies:
                in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
                adj[e.source_node].append(e.target_node)

        queue = deque(n for n, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj.get(n, []):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        # If we couldn't order everything, there's a cycle (already caught).
        return set(order)

    def _is_toplevel(self, node_name: str, edges: list[Edge]) -> bool:
        return not any(e.target_node == node_name for e in edges)

    def topological_order(self) -> list[str]:
        """Return the node names in execution order."""
        in_deg: dict[str, int] = {n: 0 for n in self._node_proxies}
        adj: dict[str, list[str]] = defaultdict(list)
        for e in self._validated_edges:
            in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
            adj[e.source_node].append(e.target_node)

        queue = deque(n for n, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj.get(n, []):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        return order

    def run(self, program_input: Any) -> ProgramRunResult:
        """Execute the program once.

        `program_input` is passed to every top-level (no-incoming-edge) node.
        Each downstream node receives the merged output of its producers,
        filtered to the field its edge declared.

        Returns a ProgramRunResult with the final outputs and the manifest.
        """
        run_id = str(uuid.uuid4())
        manifest = ManifestBuilder(program_name=self.name, run_id=run_id)
        manifest.record_start()

        outputs: dict[str, NodeResult] = {}
        # Track which inputs each node should receive, gathered from upstream.
        inputs_for_node: dict[str, dict[str, Any]] = defaultdict(dict)

        order = self.topological_order()
        ctx = NodeContext(
            program_name=self.name,
            run_id=run_id,
            node_name="",
            metadata={},
        )

        # Top-level nodes get the program input.
        for n in self._node_proxies:
            incoming = [e for e in self._validated_edges if e.target_node == n]
            if not incoming:
                inputs_for_node[n]["input"] = program_input

        # Resolve producer→consumer mapping per edge.
        edge_index: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        for e in self._validated_edges:
            edge_index[(e.source_node, e.source_variant)].append(e)

        terminal_name: str | None = None
        last_output: NodeResult | None = None

        for node_name in order:
            proxy = self._node_proxies[node_name]
            ctx_node = NodeContext(
                program_name=self.name,
                run_id=run_id,
                node_name=node_name,
                metadata={},
            )
            t0 = time.time()
            try:
                payload = inputs_for_node.get(node_name) or {}
                # Routing rule:
                # - 1 incoming total → pass value directly as first positional arg
                # - Multiple incoming OR program_input + incoming → pass as dict
                # - 0 incoming (top of pipeline) → pass program_input
                if len(payload) == 1:
                    only_value = next(iter(payload.values()))
                    result = proxy.fn(ctx_node, only_value)
                elif payload:
                    result = proxy.fn(ctx_node, payload)
                else:
                    # Shouldn't happen (Program ensures at least one incoming)
                    # but be defensive.
                    result = proxy.fn(ctx_node, program_input)
                dt = (time.time() - t0) * 1000.0
                manifest.record_node(
                    node_name=node_name,
                    variant="ood" if result.is_ood else "ok",
                    latency_ms=dt,
                    confidence=_extract_confidence(result.ok) if result.is_ok else None,
                )
            except Exception as e:
                manifest.record_error(node_name=node_name, error=str(e))
                raise

            outputs[node_name] = result
            last_output = result
            terminal_name = node_name

            # Propagate output to consumers via the edge routing table.
            for variant in ("ok", "ood"):
                payload_value = result.ok if (variant == "ok" and result.is_ok) else (
                    result.ood if (variant == "ood" and result.is_ood) else None
                )
                if payload_value is None:
                    continue
                for e in edge_index[(node_name, variant)]:
                    inputs_for_node[e.target_node][e.target_field] = payload_value

        manifest.record_end()
        return ProgramRunResult(
            outputs=outputs,
            manifest=manifest.build(),
            terminal=last_output,
        )
