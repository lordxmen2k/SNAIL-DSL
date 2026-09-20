"""Tests for weight_pin SHA-256 verification (v0.4.0).

Benchmarks:
- test_parse_weight_pin_valid
- test_parse_weight_pin_rejects_missing_hash
- test_parse_weight_pin_rejects_wrong_format
- test_compute_sha256_matches_known_value
- test_verify_weight_pin_match
- test_verify_weight_pin_mismatch
- test_program_rejects_mismatched_pin
"""

from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from snail import (
    Program,
    edge,
    WeightPinMismatch,
    parse_weight_pin,
    verify_weight_pin,
    NodeResult,
)
from snail.training.freezer import compute_sha256, freeze_weights


# ── Schemas ──────────────────────────────────────────────────────────────


class FooOk:
    pass


# ── Tests ─────────────────────────────────────────────────────────────────


def test_parse_weight_pin_valid():
    """Valid `<model>@sha256:<64-hex>` parses into (name, hash)."""
    h = "a" * 64
    name, expected = parse_weight_pin(f"phi-4-mini@sha256:{h}")
    assert name == "phi-4-mini"
    assert expected == h


def test_parse_weight_pin_rejects_missing_hash():
    """Pin without `sha256:` prefix or empty hash raises ValueError."""
    with pytest.raises(ValueError, match="form"):
        parse_weight_pin("phi-4-mini@sha256:")
    with pytest.raises(ValueError, match="form"):
        parse_weight_pin("phi-4-mini")


def test_parse_weight_pin_rejects_wrong_format():
    """Hash shorter than 64 hex chars rejected."""
    with pytest.raises(ValueError, match="form"):
        parse_weight_pin("phi-4-mini@sha256:abc123")  # only 6 hex chars


def test_compute_sha256_matches_known_value():
    """SHA-256 of a known string matches the standard hash."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"hello world")
        path = Path(f.name)
    try:
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_sha256(path) == expected
    finally:
        path.unlink()


def test_verify_weight_pin_match():
    """Pin matching the file's actual hash → returns hash without raising."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"weights content")
        path = Path(f.name)
    try:
        h = hashlib.sha256(b"weights content").hexdigest()
        actual = verify_weight_pin(f"my-model@sha256:{h}", path)
        assert actual == h
    finally:
        path.unlink()


def test_verify_weight_pin_mismatch():
    """Pin not matching the file's actual hash → WeightPinMismatch."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"weights content")
        path = Path(f.name)
    try:
        wrong = "0" * 64
        with pytest.raises(WeightPinMismatch, match="mismatch"):
            verify_weight_pin(f"my-model@sha256:{wrong}", path)
    finally:
        path.unlink()


def test_program_rejects_mismatched_pin():
    """A Program constructed with a node whose weight_pin doesn't match
    its frozen_weights file raises WeightPinMismatch at construction time."""
    from snail.node import node as snail_node

    class Foo(NodeResult):
        pass

    # Write a fake weights file we know the hash of.
    with tempfile.NamedTemporaryFile(suffix=".snail.json", delete=False) as f:
        f.write(json.dumps({"format": "snail-json-v1", "weights": {}}).encode())
        path = Path(f.name)
    try:
        actual_hash = compute_sha256(path)
        wrong_hash = "0" * 64

        # First: pin matches → Program constructs
        @snail_node(
            name="matched",
            input_schema=dict,
            output_schema=Foo,
            distribution="d",
            frozen_weights=str(path),
            weight_pin=f"my-model@sha256:{actual_hash}",
            confidence_threshold=0.5,
        )
        def matched(ctx, weights, x):
            return Foo()

        prog_match = Program(name="matched", nodes=[matched])

        # Second: pin does NOT match → Program raises WeightPinMismatch
        @snail_node(
            name="mismatched",
            input_schema=dict,
            output_schema=Foo,
            distribution="d",
            frozen_weights=str(path),
            weight_pin=f"my-model@sha256:{wrong_hash}",
            confidence_threshold=0.5,
        )
        def mismatched(ctx, weights, x):
            return Foo()

        with pytest.raises((ValueError, WeightPinMismatch), match="mismatch"):
            Program(name="bad", nodes=[mismatched])

        # Verify the matched one still constructed (no regression)
        assert prog_match.name == "matched"
    finally:
        path.unlink()
