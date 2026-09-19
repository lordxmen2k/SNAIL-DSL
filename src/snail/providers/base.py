"""Provider protocol — base classes for hosted LLM providers.

A Provider is anything that can take a prompt and return a
ProviderResponse. Built-in providers: stub (default), anthropic,
openai-compat, ollama. Custom providers can be registered at runtime.

The discipline is unchanged: HostedNode still goes through @node,
still produces an output_schema, still emits an OOD signal on
confidence failure. Providers are a transport-layer concern; the
SNAIL contract sits above them.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Result of a single provider completion call."""

    text: str
    confidence: float | None
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""


class Provider(ABC):
    """Abstract base for hosted LLM providers."""

    name: str = ""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a prompt to the provider and return the response.

        Implementations should raise ProviderError subclasses on failure:
        - AuthError if API key missing/invalid
        - RateLimitError on HTTP 429
        - TimeoutError on request timeout
        - SchemaError if the response cannot be parsed
        """


class ProviderRegistry:
    """In-process registry mapping provider names to Provider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> Provider:
        if name not in self._providers:
            raise KeyError(
                f"Unknown provider: {name!r}. "
                f"Available: {sorted(self._providers)}"
            )
        return self._providers[name]

    def default(self) -> Provider:
        if "stub" not in self._providers:
            raise KeyError("Stub provider not registered")
        return self._providers["stub"]
