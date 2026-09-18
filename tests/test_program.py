"""Tests for the Program / edge / DAG machinery."""

import pytest
from pydantic import BaseModel
from snail import node, NodeContext, NodeResult, OODSignal, Program, edge


# Two simple node types for testing.

class StepOk(BaseModel):
    value: int
    confidence: float


class Step(NodeResult):
    ok: StepOk | None = None
    ood: OODSignal | None = None


def make_step_node(name, output_value, confidence):
    """Factory to create a small @node for tests."""

    @node(
        name=name,
        input_schema=dict,
        output_schema=Step,
        distribution="test",
        frozen_weights="",  # empty → no file required for testing
        confidence_threshold=0.5,
    )
    def fn(ctx, weights, payload):
        return Step(ok=StepOk(value=output_value, confidence=confidence))

    return fn


def make_ood_node(name):
    """Factory to create a node that always returns OOD."""

    @node(
        name=name,
        input_schema=dict,
        output_schema=Step,
        distribution="test",
        frozen_weights="",
    )
    def fn(ctx, weights, payload):
        return Step(ood=OODSignal(reason="explicit", distribution="test"))

    return fn


def test_program_single_node_runs():
    """Single-node program runs end-to-end."""
    n = make_step_node("only", 42, 0.9)
    prog = Program(name="single", nodes=[n])
    result = prog.run({"input": "data"})
    assert "only" in result.outputs
    assert result.outputs["only"].is_ok
    assert result.outputs["only"].ok.value == 42


def test_program_two_nodes_serial():
    """Two nodes, second consumes first's output via edge()."""
    a = make_step_node("a", 10, 0.9)
    b = make_step_node("b", 20, 0.9)

    # Edge from a.ok to b
    e1 = edge(a.ok)
    prog = Program(name="two", nodes=[a, b], edges=[e1])
    result = prog.run({"input": "data"})

    assert "a" in result.outputs
    assert "b" in result.outputs
    assert result.outputs["a"].ok.value == 10
    assert result.outputs["b"].ok.value == 20


def test_program_ood_routing():
    """When a node returns OOD, downstream can be reached via .ood edge."""
    ooder = make_ood_node("ooder")
    sink = make_step_node("sink", 99, 0.9)

    e1 = edge(ooder.ood)
    prog = Program(name="ood_route", nodes=[ooder, sink], edges=[e1])
    result = prog.run({"input": "data"})
    assert result.outputs["ooder"].is_ood
    assert result.outputs["sink"].is_ok


def test_program_unknown_node_rejected():
    """Edge referencing an unknown node should fail validation."""
    n = make_step_node("only", 1, 0.9)
    e = edge(n.ok)
    # Manually craft an edge to a nonexistent node
    from snail.program import Edge
    bad_e = Edge(source_node="ghost", source_variant="ok", target_node="only")

    with pytest.raises(ValueError, match="static validation"):
        Program(name="bad", nodes=[n], edges=[bad_e])


def test_program_non_node_function_rejected():
    """Functions without @node should be rejected."""
    def plain_fn(x):
        return x

    with pytest.raises(TypeError, match="@node"):
        Program(name="bad", nodes=[plain_fn])


def test_program_duplicate_node_name_rejected():
    """Two nodes with the same name should be rejected."""
    a = make_step_node("dup", 1, 0.9)
    b = make_step_node("dup", 2, 0.9)
    with pytest.raises(ValueError, match="duplicate"):
        Program(name="dup_prog", nodes=[a, b])


def test_program_topological_order():
    """topological_order should return nodes in dependency order."""
    a = make_step_node("a", 1, 0.9)
    b = make_step_node("b", 1, 0.9)
    c = make_step_node("c", 1, 0.9)

    prog = Program(
        name="topo",
        nodes=[a, b, c],
        edges=[edge(a.ok), edge(b.ok)],
    )
    order = prog.topological_order()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_program_manifest_emitted():
    """Every run should emit a manifest with one event per node fired."""
    a = make_step_node("a", 1, 0.9)
    b = make_step_node("b", 1, 0.9)
    prog = Program(name="m", nodes=[a, b], edges=[edge(a.ok)])

    result = prog.run({"input": "data"})
    assert result.manifest is not None
    assert result.manifest.program_name == "m"
    assert len(result.manifest.node_events) == 2
    fired = {e.node_name for e in result.manifest.node_events}
    assert fired == {"a", "b"}


def test_program_manifest_records_ood_variant():
    """When a node returns OOD, the manifest event should record 'ood'."""
    ooder = make_ood_node("ooder")
    sink = make_step_node("sink", 1, 0.9)
    prog = Program(name="m2", nodes=[ooder, sink], edges=[edge(ooder.ood)])

    result = prog.run({"input": "data"})
    ood_events = [e for e in result.manifest.node_events if e.variant == "ood"]
    assert len(ood_events) == 1
    assert ood_events[0].node_name == "ooder"


def test_program_manifest_to_json_serializable():
    """Manifest must serialize to JSON cleanly."""
    a = make_step_node("a", 1, 0.9)
    prog = Program(name="js", nodes=[a])
    result = prog.run({"input": "data"})
    js = result.manifest.to_json()
    assert isinstance(js, str)
    assert "program_name" in js
    assert "node_events" in js


def test_program_empty_pipeline():
    """Program with no nodes should refuse to build."""
    with pytest.raises((ValueError, TypeError)):
        Program(name="empty", nodes=[])
