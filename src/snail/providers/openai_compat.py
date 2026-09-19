"""OpenAI-compat provider — Chat Completions API.

Works with OpenAI, Together, Groq, OpenRouter, and any other
provider implementing the OpenAI Chat Completions schema. Configure
`base_url` and `api_key_env` for the target service.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import os
import time
from typing import Any
import httpx

from snail.providers.base import Provider, ProviderResponse
from snail.providers.errors import (
    AuthError,
    RateLimitError,
    SchemaError,
    TimeoutError,
)


class OpenAICompatProvider(Provider):
    """OpenAI-compatible Chat Completions provider."""

    name = "openai"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
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
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ProviderResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AuthError(f"Missing API key in env var {self.api_key_env!r}")

        model = model or "gpt-5"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        t0 = time.time()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"OpenAI-compat request exceeded {timeout_s}s: {e}"
            ) from e

        latency_ms = (time.time() - t0) * 1000.0
        if resp.status_code == 429:
            raise RateLimitError(f"OpenAI-compat rate-limited: {resp.text[:200]}")
        if resp.status_code in (401, 403):
            raise AuthError(f"OpenAI-compat auth failed: {resp.text[:200]}")
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception as e:
            raise SchemaError(
                f"OpenAI-compat returned non-JSON: {resp.text[:200]}"
            ) from e

        text = ""
        for choice in data.get("choices", []):
            msg = choice.get("message") or {}
            text += msg.get("content", "") or ""

        return ProviderResponse(
            text=text,
            confidence=None,
            latency_ms=latency_ms,
            raw=data,
            model=data.get("model", model),
        )
