"""DAG renderer — public API.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations

from snail.render.layout import Box, EdgeLayout, LayoutResult, layout_program
from snail.render.style import Theme
from snail.render.svg import render_program_svg, render_svg
from snail.render.theme_dark import DARK

__all__ = [
    "Box",
    "EdgeLayout",
    "LayoutResult",
    "layout_program",
    "render_svg",
    "render_program_svg",
    "Theme",
    "DARK",
]
