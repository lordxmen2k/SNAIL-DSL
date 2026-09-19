"""Tests for DAG layout.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal, Program, edge, node
from snail.render.layout import layout_program


class In(BaseModel):
    x: int


class Ok(BaseModel):
    y: int


class Out(NodeResult):
    ok: Ok | None = None
    ood: OODSignal | None = None


@node(name="a", input_schema=In, output_schema=Out, distribution="t")
def a(ctx, w, p):
    return Out(ok=Ok(y=p.x))


@node(name="b", input_schema=In, output_schema=Out, distribution="t")
def b(ctx, w, p):
    return Out(ok=Ok(y=p.x))


@node(name="c", input_schema=In, output_schema=Out, distribution="t")
def c(ctx, w, p):
    return Out(ok=Ok(y=p.x))


prog = Program(name="p", nodes=[a, b, c])


def test_layout_produces_no_overlap():
    layout = layout_program(prog)
    boxes = list(layout.nodes.values())
    for i, b1 in enumerate(boxes):
        for b2 in boxes[i + 1 :]:
            assert not (
                b1.x < b2.x + b2.w
                and b1.x + b1.w > b2.x
                and b1.y < b2.y + b2.h
                and b1.y + b1.h > b2.y
            ), f"{b1} overlaps {b2}"


def test_layout_every_node_has_box():
    layout = layout_program(prog)
    assert set(layout.nodes.keys()) == {"a", "b", "c"}


def test_layout_with_edges_assigns_layers():
    """Diamond: a → b → d, a → c → d."""
    @node(name="d", input_schema=In, output_schema=Out, distribution="t")
    def d(ctx, w, p):
        return Out(ok=Ok(y=p.x))

    diamond = Program(
        name="d",
        nodes=[a, b, c, d],
        edges=[edge(a.ok), edge(b.ok), edge(c.ok)],
    )
    layout = layout_program(diamond)
    # Layer 0: a, Layer 1: b, c, Layer 2: d — verify by x position
    a_box = layout.nodes["a"]
    b_box = layout.nodes["b"]
    d_box = layout.nodes["d"]
    assert a_box.x < b_box.x < d_box.x


def test_layout_result_has_dimensions():
    layout = layout_program(prog)
    assert layout.width > 0
    assert layout.height > 0
