"""OOD (Out-of-Distribution) signal type.

Every SNAIL node output is one of two shapes:
- `ok` with the actual result
- `ood` with an OODSignal explaining why the node couldn't trust itself

This is the "first-class type" requirement — OOD is not a string,
not a sentinel value, not an exception. It's a variant of the
output type, type-checked at build time.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class OODSignal(BaseModel):
    """Marker that an input was outside the trained distribution.

    The signal carries enough information for downstream routing:
    - `reason`: why this was flagged (low_confidence / schema_violation / explicit)
    - `confidence`: the confidence score that triggered the flip (if any)
    - `threshold`: the per-node threshold that was crossed
    - `distribution`: the declared distribution the input was supposed to match
    """

    reason: Literal["low_confidence", "schema_violation", "explicit"] = "low_confidence"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    distribution: str = ""

    model_config = {"frozen": True}
