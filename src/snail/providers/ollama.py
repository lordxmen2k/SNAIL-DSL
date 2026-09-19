"""Ollama provider — local /api/chat endpoint.

Connects to an Ollama instance running on localhost (or another host).
No API key required. Default model: llama4-scout.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import time
from typing import Any
import httpx

from snail.providers.base import Provider, ProviderResponse
from snail.providers.errors import SchemaError, TimeoutError


class OllamaProvider(Provider):
    """Ollama local chat provider."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_s: float = 60.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        model = model or "llama4-scout"
        url = f"{self.base_url}/api/chat"
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": (
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": "user", "content": prompt}],
        }

        t0 = time.time()
        try:
            resp = httpx.post(url, json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Ollama request exceeded {timeout_s}s: {e}"
            ) from e

        latency_ms = (time.time() - t0) * 1000.0
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception as e:
            raise SchemaError(
                f"Ollama returned non-JSON: {resp.text[:200]}"
            ) from e

        text = (data.get("message") or {}).get("content", "")
        return ProviderResponse(
            text=text,
            confidence=None,
            latency_ms=latency_ms,
            raw=data,
            model=data.get("model", model),
        )
