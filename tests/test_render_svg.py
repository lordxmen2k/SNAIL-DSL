"""Tests for SVG writer.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal, Program, edge, node
from snail.render import DARK, render_program_svg, render_svg


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


prog = Program(name="p", nodes=[a, b], edges=[edge(a.ok)])


def test_render_program_svg_returns_well_formed_xml():
    svg = render_program_svg(prog, title="P")
    assert svg.startswith("<?xml")
    assert svg.rstrip().endswith("</svg>")
    assert "viewBox=" in svg
    assert ">a<" in svg or " a " in svg or ">a<" in svg
    assert "b" in svg


def test_render_svg_uses_theme_colors():
    layout = prog.__class__  # noqa: F841 — placeholder
    from snail.render.layout import layout_program

    layout = layout_program(prog)
    svg = render_svg(layout, DARK, title="hello")
    assert "0a0e1a" in svg  # dark bg
    assert "3D8BFD" in svg  # blue OK edge / node stroke


def test_render_includes_title():
    svg = render_program_svg(prog, title="My Program")
    assert "My Program" in svg
