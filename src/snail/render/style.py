"""Theme — visual vocabulary for the DAG renderer.

Themes bundle colors + fonts into a single dataclass so callers can
swap visual styles without touching the layout or SVG writer.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    bg: str
    node_fill: str
    node_stroke: str
    edge_ok: str
    edge_ood: str
    text: str
    text_dim: str
    font_family: str
