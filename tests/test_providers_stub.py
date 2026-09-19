"""Tests for StubProvider.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations

from snail.providers.stub import StubProvider


def test_stub_provider_deterministic():
    p = StubProvider()
    r1 = p.complete("hello world")
    r2 = p.complete("hello world")
    assert r1.text == r2.text
    assert r1.model == "stub-v1"
    assert r1.latency_ms >= 0
    assert 0.0 <= (r1.confidence or 0.0) <= 1.0


def test_stub_provider_different_prompts_different_responses():
    p = StubProvider()
    r1 = p.complete("hello")
    r2 = p.complete("goodbye")
    # Deterministic but distinct hashes → distinct canned responses
    assert r1.text != r2.text or r1.raw != r2.raw


def test_stub_provider_overrides_model():
    p = StubProvider()
    r = p.complete("x", model="custom-model")
    assert r.model == "custom-model"
