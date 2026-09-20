"""Anthropic provider — Anthropic Messages API.

Supports the 2026 Claude model family:
- claude-sonnet-4-5 (Sep 2025)
- claude-sonnet-5 (Jun 2026)
- claude-opus-5 (Jul 2026)

Reads ANTHROPIC_API_KEY from env by default. Configure via the
`api_key_env` constructor arg to point at a different env var.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import os
import time
from typing import Any, Optional
import httpx

from snail.providers.base import Provider, ProviderResponse
from snail.providers.errors import (
    AuthError,
    RateLimitError,
    SchemaError,
    TimeoutError,
)
from snail.providers.pricing import cost_usd


_DEFAULT_BASE = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    """Anthropic Messages API provider."""

    name = "anthropic"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_s: float = 30.0,
        system: str | None = None,
        max_tokens: int = 1024,
        cost_per_1k_input: Optional[float] = None,
        cost_per_1k_output: Optional[float] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AuthError(f"Missing API key in env var {self.api_key_env!r}")

        model = model or "claude-sonnet-4-5"
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        t0 = time.time()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Anthropic request exceeded {timeout_s}s: {e}"
            ) from e

        latency_ms = (time.time() - t0) * 1000.0
        if resp.status_code == 429:
            raise RateLimitError(f"Anthropic rate-limited: {resp.text[:200]}")
        if resp.status_code in (401, 403):
            raise AuthError(f"Anthropic auth failed: {resp.text[:200]}")
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception as e:
            raise SchemaError(
                f"Anthropic returned non-JSON response: {resp.text[:200]}"
            ) from e

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # v0.3.0 — tokens + cost
        usage = data.get("usage", {}) or {}
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
        actual_model = data.get("model", model)
        cost = cost_usd(
            "anthropic",
            actual_model,
            tokens_in,
            tokens_out,
            override_input=cost_per_1k_input,
            override_output=cost_per_1k_output,
        )

        return ProviderResponse(
            text=text,
            confidence=None,  # Anthropic Messages API does not expose logprobs
            latency_ms=latency_ms,
            raw=data,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
