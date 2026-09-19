"""Tests for Provider ABC + registry + error hierarchy.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations

from snail.providers import ProviderRegistry
from snail.providers.base import Provider, ProviderResponse
from snail.providers.stub import StubProvider


def test_registry_registers_and_retrieves():
    reg = ProviderRegistry()
    reg.register(StubProvider())
    p = reg.get("stub")
    assert isinstance(p, Provider)


def test_default_provider_is_stub():
    reg = ProviderRegistry()
    reg.register(StubProvider())
    assert reg.default().name == "stub"


def test_provider_response_dataclass():
    r = ProviderResponse(text="hello", confidence=0.9, latency_ms=42.0, raw={}, model="stub-v1")
    assert r.text == "hello"
    assert r.confidence == 0.9
