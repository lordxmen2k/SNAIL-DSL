"""Provider error hierarchy.

All provider implementations (Anthropic, OpenAI, Ollama, ...) raise
ProviderError subclasses on failure. The CLI and HostedNode wrapper
catch ProviderError and convert to OOD when appropriate.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class AuthError(ProviderError):
    """Missing or invalid API key."""


class RateLimitError(ProviderError):
    """Provider throttled the request."""


class SchemaError(ProviderError):
    """Provider returned an unparseable response."""


class TimeoutError(ProviderError):
    """Request exceeded the configured timeout."""
