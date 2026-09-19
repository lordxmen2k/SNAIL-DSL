"""Provider registry — public API for the SNAIL Provider abstraction.

Providers wrap external LLM endpoints (Anthropic, OpenAI-compatible,
Ollama) behind a common interface. HostedNode dispatches through
this registry via the `provider` kwarg.

Built-ins:
- stub: deterministic no-network provider (default)
- anthropic: Anthropic Messages API
- openai: OpenAI-compatible Chat Completions (also Together, Groq, OpenRouter)
- ollama: local Ollama /api/chat

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations

from snail.providers.base import Provider, ProviderRegistry, ProviderResponse
from snail.providers.errors import (
    AuthError,
    ProviderError,
    RateLimitError,
    SchemaError,
    TimeoutError,
)

__all__ = [
    "Provider",
    "ProviderRegistry",
    "ProviderResponse",
    "ProviderError",
    "AuthError",
    "RateLimitError",
    "SchemaError",
    "TimeoutError",
]
