"""snail.types — internal type definitions.

This subpackage holds NodeResult, OODSignal, and related types.
It deliberately does NOT re-export the @node decorator (that's in
snail.node and would create a circular import).
"""

from snail.types.result import NodeResult
from snail.types.ood import OODSignal

__all__ = ["NodeResult", "OODSignal"]
