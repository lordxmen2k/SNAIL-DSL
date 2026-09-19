"""Tests for OllamaProvider.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from snail.providers.ollama import OllamaProvider


FIXTURE = Path(__file__).parent / "fixtures" / "ollama_chat_response.json"


def test_ollama_parses_response():
    fixture_data = json.loads(FIXTURE.read_text())
    fake_response = MagicMock()
    fake_response.json.return_value = fixture_data
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = json.dumps(fixture_data)

    with patch(
        "snail.providers.ollama.httpx.post", return_value=fake_response
    ) as mock_post:
        p = OllamaProvider()
        r = p.complete("hi", model="llama4-scout")

    assert "Hi back" in r.text
    assert r.model == "llama4-scout"

    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "llama4-scout"
    assert body["messages"][-1]["content"] == "hi"
    assert body["stream"] is False


def test_ollama_custom_base_url():
    p = OllamaProvider(base_url="http://gpu-host.lan:11434")
    assert p.base_url == "http://gpu-host.lan:11434"


def test_ollama_sends_system_prompt():
    fixture_data = json.loads(FIXTURE.read_text())
    fake_response = MagicMock()
    fake_response.json.return_value = fixture_data
    fake_response.raise_for_status.return_value = None
    fake_response.status_code = 200
    fake_response.text = json.dumps(fixture_data)

    with patch(
        "snail.providers.ollama.httpx.post", return_value=fake_response
    ) as mock_post:
        p = OllamaProvider()
        p.complete("hi", system="Be terse.")

    body = mock_post.call_args.kwargs["json"]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Be terse."
