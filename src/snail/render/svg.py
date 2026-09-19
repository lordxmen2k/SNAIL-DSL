"""SVG writer — turns a LayoutResult into a string of SVG markup.

Pure-stdlib (uses html.escape). No external dependencies.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import html
from typing import Any

from snail.render.layout import LayoutResult
from snail.render.style import Theme


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def render_svg(layout: LayoutResult, theme: Theme, *, title: str | None = None) -> str:
    """Render a LayoutResult to a string of SVG markup."""
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {layout.width} {layout.height}" '
        f'font-family="{_escape(theme.font_family)}">',
        f'<rect width="{layout.width}" height="{layout.height}" fill="{theme.bg}"/>',
    ]
    if title:
        parts.append(
            f'<text x="40" y="40" fill="{theme.text}" font-size="22" '
            f'font-weight="700">{_escape(title)}</text>'
        )

    # Edges first so nodes draw over them
    for e in layout.edges:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in e.points)
        color = theme.edge_ok if e.variant == "ok" else theme.edge_ood
        dash = "" if e.variant == "ok" else 'stroke-dasharray="6 3"'
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2" {dash}/>'
        )

    # Nodes
    for name, box in layout.nodes.items():
        parts.append(
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" '
            f'width="{box.w:.1f}" height="{box.h:.1f}" rx="8" '
            f'fill="{theme.node_fill}" stroke="{theme.node_stroke}" '
            f'stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{box.x + box.w / 2:.1f}" '
            f'y="{box.y + box.h / 2 + 5:.1f}" text-anchor="middle" '
            f'fill="{theme.text}" font-size="16" font-weight="600">'
            f'{_escape(name)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_program_svg(
    program: Any, *, title: str | None = None, theme: Theme | None = None
) -> str:
    """One-call public API: layout + render a Program to SVG."""
    from snail.render.layout import layout_program

    if theme is None:
        from snail.render.theme_dark import DARK

        theme = DARK

    layout = layout_program(program)
    return render_svg(layout, theme, title=title or program.name)
