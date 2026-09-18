"""Shared pytest fixtures and configuration for SNAIL tests."""

import sys
import pytest
from pathlib import Path

# Ensure the src/ layout is importable during testing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def sample_input_dict():
    return {"image_path": "scan.png", "size": (1024, 768)}


@pytest.fixture
def small_weights_file(tmp_path):
    """Write a tiny JSON sidecar weights file for testing."""
    p = tmp_path / "tiny.snail.json"
    p.write_text('{"model": "stub", "weights": [0.1, 0.2, 0.3]}')
    return p
