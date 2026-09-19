"""freeze_weights — serialize a SNAIL node's weights + metadata.

The frozen file format is JSON with:
  - format: "snail-json-v1"
  - node name, distribution, schemas
  - weights payload (whatever the user put there)
  - sha256 of the file itself (computed after writing)

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_weights(node: Any, output_path: str | Path) -> Path:
    """Serialize a SNAIL node's metadata to a frozen file.

    The node must expose `_snail_name`, `_snail_distribution`,
    `_snail_input_schema`, `_snail_output_schema`,
    `_snail_confidence_threshold`, and `_snail_weights`.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    in_schema = getattr(node, "_snail_input_schema", None)
    out_schema = getattr(node, "_snail_output_schema", None)
    in_schema_name = in_schema.__name__ if in_schema else None
    out_schema_name = out_schema.__name__ if out_schema else None

    weights = getattr(node, "_snail_weights", None)
    weights_raw = getattr(weights, "raw", None) if weights else None

    body: dict[str, Any] = {
        "format": "snail-json-v1",
        "node": getattr(node, "_snail_name", "unknown"),
        "distribution": getattr(node, "_snail_distribution", ""),
        "input_schema": in_schema_name,
        "output_schema": out_schema_name,
        "confidence_threshold": getattr(node, "_snail_confidence_threshold", 0.5),
        "weights": weights_raw,
    }
    out.write_text(json.dumps(body, indent=2))
    sha = compute_sha256(out)
    final = json.loads(out.read_text())
    final["sha256"] = sha
    out.write_text(json.dumps(final, indent=2))
    return out
