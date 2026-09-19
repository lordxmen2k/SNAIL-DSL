"""Tests for HostedNode dispatching through the Provider registry.

HostedNode's output is a dict (endpoint, prompt, response, provider,
model, confidence) so the test schemas mirror that structure.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import os
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal
from snail.wrappers import HostedNode


class In(BaseModel):
    text: str


class HostedOk(BaseModel):
    endpoint: str
    prompt: str
    response: str
    provider: str
    model: str | None = None
    confidence: float | None = None


class Out(NodeResult):
    ok: HostedOk | None = None
    ood: OODSignal | None = None


def test_hosted_node_uses_default_stub():
    """Without explicit provider kwarg, HostedNode uses the stub."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    h = HostedNode(
        name="summarize",
        input_schema=In,
        output_schema=Out,
        distribution="test",
        endpoint="anthropic://claude-sonnet-5",
        prompt_template="Summarize: {text}",
        api_key_env="ANTHROPIC_API_KEY",
    )
    ctx = NodeContext(program_name="t", run_id="r", node_name="summarize", metadata={})
    r = h(ctx, In(text="hello world"))
    assert r.is_ok
    assert r.ok.provider == "stub"


def test_hosted_node_dispatches_to_anthropic():
    """With provider='anthropic', HostedNode hits AnthropicProvider."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    h = HostedNode(
        name="summarize",
        input_schema=In,
        output_schema=Out,
        distribution="test",
        endpoint="anthropic://claude-sonnet-5",
        prompt_template="Summarize: {text}",
        api_key_env="ANTHROPIC_API_KEY",
        provider="anthropic",
    )
    fake_data = {
        "id": "x",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "summary text"}],
        "usage": {},
    }
    with patch("snail.providers.anthropic.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = fake_data
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "{}"
        ctx = NodeContext(program_name="t", run_id="r", node_name="summarize", metadata={})
        r = h(ctx, In(text="hello"))
    assert r.is_ok
    assert r.ok.response == "summary text"
    assert r.ok.provider == "anthropic"


def test_hosted_node_unknown_provider_yields_ood():
    """Unknown provider name → OOD signal (not exception)."""
    h = HostedNode(
        name="summarize",
        input_schema=In,
        output_schema=Out,
        distribution="test",
        endpoint="bogus://whatever",
        prompt_template="Summarize: {text}",
        api_key_env="BOGUS_API_KEY",
        provider="no_such_provider",
    )
    ctx = NodeContext(program_name="t", run_id="r", node_name="summarize", metadata={})
    r = h(ctx, In(text="hello"))
    assert r.is_ood
