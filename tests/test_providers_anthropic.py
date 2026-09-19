"""Tests for AnthropicProvider.

Uses recorded fixture + httpx mock — no live network.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from snail.providers.anthropic import AnthropicProvider
from snail.providers.errors import AuthError


FIXTURE = Path(__file__).parent / "fixtures" / "anthropic_messages_response.json"


def test_anthropic_builds_request_and_parses_response(monkeypatch):
    fixture_data = json.loads(FIXTURE.read_text())
    fake_response = MagicMock()
    fake_response.json.return_value = fixture_data
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = json.dumps(fixture_data)

    with patch("snail.providers.anthropic.httpx.post", return_value=fake_response) as mock_post:
        p = AnthropicProvider()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        r = p.complete("Summarize: hello", model="claude-sonnet-5")

    assert r.text == fixture_data["content"][0]["text"]
    assert r.model == "claude-sonnet-5"
    assert r.latency_ms >= 0
    args, kwargs = mock_post.call_args
    assert "messages" in kwargs["json"]
    assert kwargs["headers"]["x-api-key"] == "test-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_missing_api_key_raises():
    p = AnthropicProvider()
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(AuthError):
        p.complete("hello")


def test_anthropic_uses_custom_api_key_env(monkeypatch):
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "id": "x",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "ok"}],
    }
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = "{}"

    monkeypatch.setenv("MY_COMPANY_KEY", "company-key")
    with patch("snail.providers.anthropic.httpx.post", return_value=fake_response) as mock_post:
        p = AnthropicProvider(api_key_env="MY_COMPANY_KEY")
        r = p.complete("hi")

    assert r.text == "ok"
    assert mock_post.call_args.kwargs["headers"]["x-api-key"] == "company-key"
