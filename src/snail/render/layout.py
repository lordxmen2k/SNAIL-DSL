"""Layered DAG layout algorithm.

Sugiyama-style left-to-right layered layout. Nodes are grouped by
their topological "layer" (longest path from any source). Within each
layer, nodes are spaced vertically so they never overlap.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Box:
    """A node's bounding box in the rendered SVG."""

    x: float
    y: float
    w: float
    h: float


@dataclass
class EdgeLayout:
    """An edge with its routed waypoints."""

    source: str
    target: str
    variant: str  # "ok" or "ood"
    points: list[tuple[float, float]]


@dataclass
class LayoutResult:
    """The full layout for a Program."""

    nodes: dict[str, Box]
    edges: list[EdgeLayout]
    width: float
    height: float


_NODE_W = 200.0
_NODE_H = 80.0
_LAYER_GAP = 100.0
_ROW_GAP = 30.0
_PADDING = 40.0


def layout_program(program: object) -> LayoutResult:
    """Lay out a Program's nodes in a layered DAG (left-to-right).

    Reads:
      - program._node_proxies: dict[name, NodeProxy]
      - program._validated_edges: list[Edge]
    """
    proxy_names: list[str] = list(program._node_proxies.keys())  # type: ignore[attr-defined]
    in_deg: dict[str, int] = {n: 0 for n in proxy_names}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in program._validated_edges:  # type: ignore[attr-defined]
        in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
        adj[e.source_node].append(e.target_node)

    # Compute layer (longest path from any source)
    layer: dict[str, int] = {n: 0 for n in proxy_names}
    queue: deque[str] = deque(n for n in proxy_names if in_deg[n] == 0)
    while queue:
        n = queue.popleft()
        for m in adj[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            in_deg[m] -= 1
            if in_deg[m] == 0:
                queue.append(m)

    # Group nodes by layer, preserve declaration order within each layer
    layers: dict[int, list[str]] = defaultdict(list)
    for n in proxy_names:
        layers[layer[n]].append(n)

    # Compute positions
    max_rows = max((len(v) for v in layers.values()), default=0)
    layer_count = (max(layers.keys()) + 1) if layers else 1
    width = _PADDING * 2 + layer_count * _NODE_W + (layer_count - 1) * _LAYER_GAP
    height = _PADDING * 2 + max_rows * _NODE_H + max(0, max_rows - 1) * _ROW_GAP

    nodes: dict[str, Box] = {}
    for lyr_idx, names in layers.items():
        col_x = _PADDING + lyr_idx * (_NODE_W + _LAYER_GAP)
        col_height = len(names) * _NODE_H + max(0, len(names) - 1) * _ROW_GAP
        col_y_start = _PADDING + (height - _PADDING * 2 - col_height) / 2
        for row_idx, name in enumerate(names):
            nodes[name] = Box(
                x=col_x,
                y=col_y_start + row_idx * (_NODE_H + _ROW_GAP),
                w=_NODE_W,
                h=_NODE_H,
            )

    # Build edges with simple orthogonal routing (right side of source -> left side of target)
    edges: list[EdgeLayout] = []
    for e in program._validated_edges:  # type: ignore[attr-defined]
        if e.source_node in nodes and e.target_node in nodes:
            sb = nodes[e.source_node]
            tb = nodes[e.target_node]
            sx = sb.x + sb.w
            sy = sb.y + sb.h / 2
            tx = tb.x
            ty = tb.y + tb.h / 2
            mx = (sx + tx) / 2
            edges.append(
                EdgeLayout(
                    source=e.source_node,
                    target=e.target_node,
                    variant=e.source_variant,
                    points=[(sx, sy), (mx, sy), (mx, ty), (tx, ty)],
                )
            )

    return LayoutResult(nodes=nodes, edges=edges, width=width, height=height)
