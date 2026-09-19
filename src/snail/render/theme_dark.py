"""Dark theme — navy / blue / amber. Matches the SNAIL book cover.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from snail.render.style import Theme

DARK = Theme(
    bg="#0a0e1a",
    node_fill="#1a2540",
    node_stroke="#3D8BFD",
    edge_ok="#3D8BFD",
    edge_ood="#E8A020",
    text="#e8ecf3",
    text_dim="#8fa2c0",
    font_family="Inter, Helvetica, Arial, sans-serif",
)
