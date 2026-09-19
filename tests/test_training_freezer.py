"""Tests for freeze_weights and compute_sha256.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock

from snail.training.freezer import compute_sha256, freeze_weights


def test_freeze_weights_emits_snail_json(tmp_path: Path):
    mock_node = MagicMock()
    mock_node._snail_name = "n"
    mock_node._snail_distribution = "d"
    mock_node._snail_input_schema = MagicMock()
    mock_node._snail_input_schema.__name__ = "In"
    mock_node._snail_output_schema = MagicMock()
    mock_node._snail_output_schema.__name__ = "Out"
    mock_node._snail_confidence_threshold = 0.5
    mock_node._snail_weights = None

    out = tmp_path / "frozen.snail.json"
    p = freeze_weights(mock_node, out)
    data = json.loads(p.read_text())
    assert data["node"] == "n"
    assert data["distribution"] == "d"
    assert data["format"] == "snail-json-v1"
    assert "sha256" in data
    assert len(data["sha256"]) == 64  # sha256 hex length


def test_compute_sha256(tmp_path: Path):
    f = tmp_path / "x"
    f.write_text("hello")
    assert compute_sha256(f).startswith("2cf24d")  # sha256("hello")[:6]


def test_freeze_weights_records_schemas(tmp_path: Path):
    mock_node = MagicMock()
    mock_node._snail_name = "n"
    mock_node._snail_distribution = "d"
    mock_node._snail_input_schema = MagicMock()
    mock_node._snail_input_schema.__name__ = "InSchema"
    mock_node._snail_output_schema = MagicMock()
    mock_node._snail_output_schema.__name__ = "OutSchema"
    mock_node._snail_confidence_threshold = 0.7
    mock_node._snail_weights = None

    out = tmp_path / "frozen.snail.json"
    p = freeze_weights(mock_node, out)
    data = json.loads(p.read_text())
    assert data["input_schema"] == "InSchema"
    assert data["output_schema"] == "OutSchema"
    assert data["confidence_threshold"] == 0.7
