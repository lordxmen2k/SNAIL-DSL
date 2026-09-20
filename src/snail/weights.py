"""Weight pinning — SHA-256 verification of frozen-weights files.

The `weight_pin` format used by ExternalLocalNode and `@node` is:

    <model_name>@sha256:<64-hex-chars>

This module enforces that format AND verifies, at Program construction
time, that any `frozen_weights` path supplied to a node has the
SHA-256 declared in the pin.

This is what turns "frozen and hash-verified" from a marketing claim
into a verifiable discipline: a program that constructs with a
mismatched weight_pin raises immediately, before .run() is called
and before any inference cost is incurred.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import re
from pathlib import Path


class WeightPinMismatch(ValueError):
    """Raised when a frozen-weights file's SHA-256 doesn't match the pin."""


_PIN_RE = re.compile(r"^([\w\.\-]+)@sha256:([0-9a-f]{64})$")


def parse_weight_pin(pin: str) -> tuple[str, str]:
    """Parse `<model>@sha256:<hex>` into (model_name, expected_sha256).

    Raises ValueError on malformed input.
    """
    if not isinstance(pin, str):
        raise ValueError(
            f"weight_pin must be a string, got {type(pin).__name__}"
        )
    m = _PIN_RE.match(pin)
    if not m:
        raise ValueError(
            f"weight_pin must be of the form '<model>@sha256:<64-hex>', "
            f"got {pin!r}"
        )
    return m.group(1), m.group(2)


def verify_weight_pin(pin: str, actual_path: str | Path) -> str:
    """Verify that the file at `actual_path` matches the SHA-256 in `pin`.

    Returns the computed SHA-256 on success.
    Raises WeightPinMismatch on any failure (bad pin, missing file,
    hash mismatch).
    """
    model, expected = parse_weight_pin(pin)
    p = Path(actual_path)
    if not p.exists():
        raise WeightPinMismatch(
            f"weight_pin {model!r} declared but file not found at {p}"
        )
    # Lazy import to avoid a circular dep at module load.
    from snail.training.freezer import compute_sha256

    actual = compute_sha256(p)
    if actual != expected:
        raise WeightPinMismatch(
            f"weight_pin SHA-256 mismatch for {model!r}: "
            f"pin says {expected[:16]}..., file is {actual[:16]}..."
        )
    return actual
