"""Golden test helpers — write and verify golden test cases.

A golden case is an (input, expected) pair. `write_golden` emits it to
a directory layout. `verify_golden` runs a Program against every golden
case and reports mismatches.

Expected keys support `_min` and `_max` suffixes for numeric bounds:
    expected: {y_min: 5, y_max: 10}     # 5 <= y <= 10
    expected: {intent: "refund"}         # exact match

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoldenFailure:
    case_id: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    reason: str


def write_golden(name: str, output_dir: str | Path, case: dict[str, Any]) -> Path:
    """Write a golden case under output_dir/golden/name/{id}.json."""
    out_dir = Path(output_dir) / "golden" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = (
        f"{int(time.time() * 1000)}_"
        f"{abs(hash(json.dumps(case, sort_keys=True))) % 10**8}"
    )
    p = out_dir / f"{case_id}.json"
    p.write_text(json.dumps(case, indent=2, sort_keys=True))
    return p


def verify_golden(program: Any, golden_dir: str | Path) -> list[GoldenFailure]:
    """Run every golden case under golden_dir/golden/*/*.json against program.

    Returns a list of GoldenFailure (empty list = all passed).
    """
    failures: list[GoldenFailure] = []
    g_dir = Path(golden_dir)
    if not g_dir.exists():
        return failures

    for case_file in sorted(g_dir.glob("golden/*/*.json")):
        try:
            case = json.loads(case_file.read_text())
        except Exception:
            continue
        try:
            # If the top-level nodes expect a BaseModel input, wrap the
            # raw dict input so it passes the @node decorator's schema check.
            result = program.run(_coerce_input(program, case["input"]))
        except Exception as e:
            failures.append(
                GoldenFailure(
                    case_id=case_file.name,
                    expected=case.get("expected", {}),
                    actual={},
                    reason=f"exception: {e}",
                )
            )
            continue

        terminal = result.terminal
        actual_payload: dict[str, Any] = {}
        if terminal is not None and terminal.is_ok:
            ok_payload = terminal.ok
            if hasattr(ok_payload, "model_dump"):
                actual_payload = ok_payload.model_dump()
            elif isinstance(ok_payload, dict):
                actual_payload = ok_payload

        if not _matches(case.get("expected", {}), actual_payload):
            failures.append(
                GoldenFailure(
                    case_id=case_file.name,
                    expected=case.get("expected", {}),
                    actual=actual_payload,
                    reason="payload mismatch",
                )
            )
    return failures


def _matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Check expected against actual with _min / _max numeric bounds."""
    for k, v in expected.items():
        if k.endswith("_min"):
            field = k[:-4]
            if float(actual.get(field, 0)) < float(v):
                return False
        elif k.endswith("_max"):
            field = k[:-4]
            if float(actual.get(field, 0)) > float(v):
                return False
        else:
            if actual.get(k) != v:
                return False
    return True


def _coerce_input(program: Any, raw_input: Any) -> Any:
    """Wrap dict inputs as the top-level node's input_schema BaseModel if applicable."""
    if not isinstance(raw_input, dict):
        return raw_input
    # Find top-level nodes (no incoming edges)
    incoming: set[str] = set()
    for e in program._validated_edges:
        incoming.add(e.target_node)
    for name, proxy in program._node_proxies.items():
        if name in incoming:
            continue
        schema = proxy.input_schema
        if schema is not None and isinstance(schema, type) and hasattr(schema, "model_validate"):
            try:
                return schema.model_validate(raw_input)
            except Exception:
                return raw_input
    return raw_input
