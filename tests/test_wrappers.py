"""Tests for the wrappers — ExternalLocalNode, HostedNode, DeterministicNode."""

import os
import pytest
from pydantic import BaseModel
from snail import NodeResult, OODSignal, NodeContext
from snail.wrappers import (
    ExternalLocalNode,
    HostedNode,
    DeterministicNode,
)


# Loose schemas — wrappers return dicts, so the `ok` field accepts Any.
class EmailResult(NodeResult):
    ok: dict | None = None
    ood: OODSignal | None = None


class LayoutResult(NodeResult):
    ok: dict | None = None
    ood: OODSignal | None = None


class SummaryResult(NodeResult):
    ok: dict | None = None
    ood: OODSignal | None = None


def test_deterministic_node_ok():
    """DeterministicNode wraps a pure function as a SNAIL node."""

    validate = DeterministicNode(
        name="validate_email",
        input_schema=dict,
        output_schema=EmailResult,
        fn=lambda x: {"address": x["value"]} if "@" in x.get("value", "") else None,
        ood_on_none=True,
    )
    assert validate._is_snail_node
    assert validate._snail_name == "validate_email"

    ctx = NodeContext(program_name="t", run_id="r", node_name="validate_email", metadata={})
    ok_out = validate(ctx, {"value": "a@b.com"})
    assert ok_out.is_ok
    assert ok_out.ok["address"] == "a@b.com"

    ood_out = validate(ctx, {"value": "no-at-sign"})
    assert ood_out.is_ood
    assert ood_out.ood.reason == "explicit"


def test_deterministic_node_no_ood_when_disabled():
    """When ood_on_none=False, a None return wraps to {"value": None}."""

    validate = DeterministicNode(
        name="validate_email_loose",
        input_schema=dict,
        output_schema=EmailResult,
        fn=lambda x: None,
        ood_on_none=False,
    )

    ctx = NodeContext(program_name="t", run_id="r", node_name="validate_email_loose", metadata={})
    out = validate(ctx, {"value": "anything"})
    assert out.is_ok
    assert out.ok == {"value": None}


def test_external_local_node_metadata():
    """ExternalLocalNode stamps weight_pin for traceability."""

    classify = ExternalLocalNode(
        name="classify_layout",
        input_schema=dict,
        output_schema=LayoutResult,
        distribution="rvl_cdip_subset",
        call=lambda img: {"layout": "tabular", "confidence": 0.85},
        weight_pin="yolov8n@sha256:abc123def456",
    )
    assert classify._is_snail_node
    assert classify._snail_extra_metadata["wrapped_kind"] == "external_local"
    assert classify._snail_extra_metadata["weight_pin"] == "yolov8n@sha256:abc123def456"

    ctx = NodeContext(program_name="t", run_id="r", node_name="classify_layout", metadata={})
    out = classify(ctx, {"path": "img.png"})
    assert out.is_ok
    assert out.ok["layout"] == "tabular"


def test_external_local_node_bad_pin_rejected():
    """weight_pin without @sha256: should be rejected."""

    with pytest.raises(ValueError, match="weight_pin"):
        ExternalLocalNode(
            name="bad_pin",
            input_schema=dict,
            output_schema=LayoutResult,
            distribution="t",
            call=lambda x: x,
            weight_pin="yolov8n",
        )


def test_hosted_node_metadata():
    """HostedNode stamps endpoint + api_key_env for traceability."""

    summarize = HostedNode(
        name="summarize",
        input_schema=dict,
        output_schema=SummaryResult,
        distribution="english_v1",
        endpoint="anthropic://claude-sonnet-4-5",
        prompt_template="Summarize: {text}",
        api_key_env="FAKE_API_KEY_FOR_TEST",
    )
    assert summarize._is_snail_node
    assert summarize._snail_extra_metadata["wrapped_kind"] == "hosted"
    assert summarize._snail_extra_metadata["endpoint"] == "anthropic://claude-sonnet-4-5"

    os.environ["FAKE_API_KEY_FOR_TEST"] = "sk-test-xxx"
    try:
        ctx = NodeContext(program_name="t", run_id="r", node_name="summarize", metadata={})
        out = summarize(ctx, {"text": "hello world"})
        assert out.is_ok
        assert "endpoint" in out.ok
        assert "Summarize:" in out.ok["prompt"]
    finally:
        del os.environ["FAKE_API_KEY_FOR_TEST"]


def test_hosted_node_missing_api_key_returns_ood():
    """If the API key env var is unset, HostedNode returns OOD."""

    os.environ.pop("FAKE_KEY_FOR_MISSING_TEST", None)
    summarize = HostedNode(
        name="summarize_no_key",
        input_schema=dict,
        output_schema=SummaryResult,
        distribution="t",
        endpoint="anthropic://x",
        prompt_template="",
        api_key_env="FAKE_KEY_FOR_MISSING_TEST",
    )
    ctx = NodeContext(program_name="t", run_id="r", node_name="summarize_no_key", metadata={})
    out = summarize(ctx, {"text": "x"})
    assert out.is_ood
    assert out.ood.reason == "explicit"


def test_hosted_node_missing_endpoint_rejected():
    with pytest.raises(ValueError, match="endpoint"):
        HostedNode(
            name="bad",
            input_schema=dict,
            output_schema=SummaryResult,
            distribution="t",
            endpoint="",
            prompt_template="",
            api_key_env="X",
        )


def test_hosted_node_missing_api_key_env_rejected():
    with pytest.raises(ValueError, match="api_key_env"):
        HostedNode(
            name="bad",
            input_schema=dict,
            output_schema=SummaryResult,
            distribution="t",
            endpoint="anthropic://x",
            prompt_template="",
            api_key_env="",
        )
