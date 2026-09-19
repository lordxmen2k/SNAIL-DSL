"""StubProvider — deterministic no-network provider for tests and demos.

The minimal stub lives here for Task 1; Task 2 replaces it with the
full deterministic canned-response implementation.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from typing import Any

from snail.providers.base import Provider, ProviderResponse


class StubProvider(Provider):
    """Returns a deterministic empty response.

    Task 2 will replace this with the canned-response implementation
    that hashes the prompt and returns a stable response. For now, just
    enough to satisfy test imports.
    """

    name = "stub"

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> ProviderResponse:
        return ProviderResponse(
            text="",
            confidence=None,
            latency_ms=0.0,
            raw={},
            model=model or "stub-v1",
        )
