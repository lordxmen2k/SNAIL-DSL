"""Tests for the @node decorator — single-pass, frozen, OOD-aware."""

import pytest
from pydantic import BaseModel
from snail import node, NodeContext, NodeResult, OODSignal


class GreetingOk(BaseModel):
    text: str
    confidence: float


class Greeting(NodeResult):
    ok: GreetingOk | None = None
    ood: OODSignal | None = None


def test_node_decorator_basic_ok(sample_input_dict, small_weights_file):
    """A node that returns OK with high confidence should pass through."""

    @node(
        name="make_greeting",
        input_schema=dict,
        output_schema=Greeting,
        distribution="test_v1",
        frozen_weights=small_weights_file,
        confidence_threshold=0.5,
    )
    def make_greeting(ctx, weights, payload):
        return Greeting(ok=GreetingOk(text="hello", confidence=0.9))

    ctx = NodeContext(program_name="t", run_id="r1", node_name="make_greeting", metadata={})
    result = make_greeting(ctx, sample_input_dict)

    assert result.is_ok
    assert result.ok.text == "hello"
    assert result.ok.confidence == 0.9
    assert result.is_ood is False


def test_node_low_confidence_becomes_ood(sample_input_dict, small_weights_file):
    """Confidence below threshold should flip to OOD automatically."""

    @node(
        name="low_conf_node",
        input_schema=dict,
        output_schema=Greeting,
        distribution="test_v1",
        frozen_weights=small_weights_file,
        confidence_threshold=0.7,
    )
    def low_conf_node(ctx, weights, payload):
        return Greeting(ok=GreetingOk(text="hello", confidence=0.3))

    ctx = NodeContext(program_name="t", run_id="r1", node_name="low_conf_node", metadata={})
    result = low_conf_node(ctx, sample_input_dict)

    assert result.is_ood
    assert result.ok is None
    assert result.ood.reason == "low_confidence"
    assert result.ood.confidence == 0.3
    assert result.ood.threshold == 0.7


def test_node_explicit_ood(sample_input_dict, small_weights_file):
    """User can explicitly return OOD; decorator passes it through."""

    @node(
        name="explicit_ood",
        input_schema=dict,
        output_schema=Greeting,
        distribution="test_v1",
        frozen_weights=small_weights_file,
    )
    def explicit_ood(ctx, weights, payload):
        return Greeting(ood=OODSignal(reason="explicit", distribution="test_v1"))

    ctx = NodeContext(program_name="t", run_id="r1", node_name="explicit_ood", metadata={})
    result = explicit_ood(ctx, sample_input_dict)

    assert result.is_ood
    assert result.ood.reason == "explicit"


def test_node_invalid_return_type_raises(sample_input_dict, small_weights_file):
    """Returning a non-BaseModel should raise TypeError."""

    @node(
        name="bad_return",
        input_schema=dict,
        output_schema=Greeting,
        distribution="test_v1",
        frozen_weights=small_weights_file,
    )
    def bad_return(ctx, weights, payload):
        return "this is not a Greeting"

    ctx = NodeContext(program_name="t", run_id="r1", node_name="bad_return", metadata={})
    with pytest.raises(TypeError):
        bad_return(ctx, sample_input_dict)


def test_node_marks_itself():
    """Decorated function must have _is_snail_node flag set."""

    @node(
        name="marked",
        input_schema=dict,
        output_schema=Greeting,
        distribution="t",
        frozen_weights="",
    )
    def marked(ctx, weights, payload):
        return Greeting(ok=GreetingOk(text="x", confidence=1.0))

    assert marked._is_snail_node is True
    assert marked._snail_name == "marked"
    assert marked._snail_distribution == "t"


def test_node_empty_distribution_rejected():
    """Distribution must be non-empty — declared at decoration time."""

    with pytest.raises(ValueError, match="distribution"):
        @node(
            name="bad",
            input_schema=dict,
            output_schema=Greeting,
            distribution="",
            frozen_weights="",
        )
        def bad(ctx, weights, payload):
            return Greeting()


def test_weights_sha256_mismatch_rejects(tmp_path):
    """Pinned weights with wrong hash should refuse to load."""
    from pathlib import Path

    p = tmp_path / "wrong.snail.json"
    p.write_text('{"x": 1}')

    with pytest.raises(RuntimeError, match="SHA-256"):
        @node(
            name="hashed",
            input_schema=dict,
            output_schema=Greeting,
            distribution="t",
            frozen_weights=p,
            weights_sha256="0" * 64,
        )
        def hashed(ctx, weights, payload):
            return Greeting()


def test_weights_sha256_match_loads(tmp_path):
    """Correct hash should load cleanly."""
    import hashlib

    p = tmp_path / "right.snail.json"
    content = b'{"x": 1}'
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    @node(
        name="hashed_ok",
        input_schema=dict,
        output_schema=Greeting,
        distribution="t",
        frozen_weights=p,
        weights_sha256=expected,
    )
    def hashed_ok(ctx, weights, payload):
        # weights.raw should be the parsed JSON.
        assert weights.raw == {"x": 1}
        return Greeting(ok=GreetingOk(text="ok", confidence=1.0))

    ctx = NodeContext(program_name="t", run_id="r", node_name="hashed_ok", metadata={})
    out = hashed_ok(ctx, {"k": "v"})
    assert out.is_ok


def test_node_result_invariant_enforced():
    """NodeResult must have exactly one of ok/ood populated."""
    with pytest.raises(ValueError, match="both"):
        Greeting(ok=GreetingOk(text="x", confidence=1.0), ood=OODSignal(reason="explicit"))

    with pytest.raises(ValueError, match="neither"):
        Greeting()


def test_node_output_is_immutable(sample_input_dict, small_weights_file):
    """Returned result should be frozen — mutation attempts raise."""

    @node(
        name="frozen_out",
        input_schema=dict,
        output_schema=Greeting,
        distribution="t",
        frozen_weights=small_weights_file,
    )
    def frozen_out(ctx, weights, payload):
        return Greeting(ok=GreetingOk(text="hi", confidence=1.0))

    ctx = NodeContext(program_name="t", run_id="r", node_name="frozen_out", metadata={})
    result = frozen_out(ctx, sample_input_dict)
    with pytest.raises(Exception):  # pydantic frozen raises ValidationError
        result.ok = GreetingOk(text="mutated", confidence=0.0)
