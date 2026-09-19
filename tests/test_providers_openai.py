"""Tests for OpenAICompatProvider.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from snail.providers.openai_compat import OpenAICompatProvider
from snail.providers.errors import AuthError


FIXTURE = Path(__file__).parent / "fixtures" / "openai_chat_response.json"


def test_openai_parses_response(monkeypatch):
    fixture_data = json.loads(FIXTURE.read_text())
    fake_response = MagicMock()
    fake_response.json.return_value = fixture_data
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = json.dumps(fixture_data)

    with patch("snail.providers.openai_compat.httpx.post", return_value=fake_response):
        p = OpenAICompatProvider()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        r = p.complete("hi", model="gpt-5")

    assert r.text == "Hi back."
    assert r.model == "gpt-5-2026-07"


def test_openai_missing_key_raises():
    p = OpenAICompatProvider()
    os.environ.pop("OPENAI_API_KEY", None)
    with pytest.raises(AuthError):
        p.complete("hi")


def test_openai_custom_base_url():
    p = OpenAICompatProvider(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    )
    assert p.base_url == "https://api.together.xyz/v1"
    assert p.api_key_env == "TOGETHER_API_KEY"


def test_openai_sends_system_prompt(monkeypatch):
    fixture_data = json.loads(FIXTURE.read_text())
    fake_response = MagicMock()
    fake_response.json.return_value = fixture_data
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = json.dumps(fixture_data)

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with patch(
        "snail.providers.openai_compat.httpx.post", return_value=fake_response
    ) as mock_post:
        p = OpenAICompatProvider()
        p.complete("hi", system="You are a summarizer.")

    body = mock_post.call_args.kwargs["json"]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are a summarizer."
