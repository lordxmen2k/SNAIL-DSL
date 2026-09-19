"""StubProvider — deterministic no-network provider for tests and demos.

Hashes the prompt and returns a stable canned response. Same prompt
always produces the same output, so golden tests can assert on it.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import hashlib
import time
from typing import Any

from snail.providers.base import Provider, ProviderResponse


class StubProvider(Provider):
    """Deterministic no-network provider.

    Used as the default HostedNode provider in v0.1.0/v0.2.0 — lets
    tests, examples, and demos run without API keys.
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
        t0 = time.time()
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        canned = f"[stub:{h}] {prompt[:80]}"
        return ProviderResponse(
            text=canned,
            confidence=0.5,
            latency_ms=(time.time() - t0) * 1000.0,
            raw={"prompt_hash": h, "stub": True},
            model=model or "stub-v1",
        )
