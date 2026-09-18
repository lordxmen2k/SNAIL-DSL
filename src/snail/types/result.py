"""NodeResult — the discriminated union every SNAIL node output conforms to.

A user-defined output schema extends NodeResult by declaring the shape of
its `ok` variant. The `ood` variant is always an OODSignal.

Example:
    class ExtractInvoiceTotalOk(BaseModel):
        total: float
        currency: str
        confidence: float

    class ExtractInvoiceTotal(NodeResult):
        ok: ExtractInvoiceTotalOk | None = None
        ood: OODSignal | None = None

The decorator enforces that exactly one of `ok` / `ood` is populated.
"""

from __future__ import annotations
from pydantic import BaseModel, model_validator
from typing import Any

from snail.types.ood import OODSignal


class NodeResult(BaseModel):
    """Base class for all SNAIL node outputs.

    Subclasses MUST declare typed `ok` and `ood` fields.
    The invariant: exactly one of `ok` / `ood` is non-None.
    """

    ok: Any | None = None
    ood: OODSignal | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _exactly_one_variant(self) -> "NodeResult":
        has_ok = self.ok is not None
        has_ood = self.ood is not None
        if has_ok and has_ood:
            raise ValueError(
                f"NodeResult {type(self).__name__}: both 'ok' and 'ood' populated. "
                "Exactly one must be non-None."
            )
        if not has_ok and not has_ood:
            raise ValueError(
                f"NodeResult {type(self).__name__}: neither 'ok' nor 'ood' populated. "
                "Exactly one must be non-None."
            )
        return self

    @property
    def is_ood(self) -> bool:
        return self.ood is not None

    @property
    def is_ok(self) -> bool:
        return self.ok is not None
