"""End-to-end integration test for v0.2.0.

Exercises: train_recipe → freeze → write_golden → verify_golden →
render_svg → CLI run. All in a single tmp_path so it's a true
integration smoke test.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from pydantic import BaseModel

from snail import NodeContext, NodeResult, OODSignal, Program, __version__, node
from snail.cli import cli
from snail.render import render_program_svg
from snail.training import load_recipe, train_recipe, verify_golden, write_golden


def test_version_is_v020():
    assert __version__ == "0.2.0"


class In(BaseModel):
    x: int


class Ok(BaseModel):
    y: int


class Out(NodeResult):
    ok: Ok | None = None
    ood: OODSignal | None = None


@node(name="double", input_schema=In, output_schema=Out, distribution="integration_test")
def double(ctx, weights, payload: In):
    return Out(ok=Ok(y=payload.x * 2))


@node(name="noop", input_schema=In, output_schema=Out, distribution="integration_test")
def noop(ctx, weights, payload: In):
    return Out(ok=Ok(y=payload.x))


prog = Program(name="integration_test", nodes=[double, noop])


def test_end_to_end_train_freeze_golden_run(tmp_path: Path):
    # 1. Train a recipe
    recipe_yaml = tmp_path / "recipe.yaml"
    recipe_yaml.write_text(
        "name: integration\n"
        "dataset: t\n"
        "frozen_output: out.snail.json\n"
        "nodes:\n"
        "  - name: n\n"
        "    input_schema: In\n"
        "    output_schema: Out\n"
        "    distribution: integration_test\n"
        "    epochs: 1\n"
        "    learning_rate: 0.001\n"
        "    weight_pin: model@sha256:deadbeef\n"
    )
    recipe = load_recipe(recipe_yaml)
    paths = train_recipe(
        recipe,
        lambda p, n: {"label": "ok", "confidence": 0.9},
        [{"x": i} for i in range(3)],
        output_dir=tmp_path,
    )
    assert any(tmp_path.rglob("*.snail.json"))

    # Use a single-node program for the golden verify step so the
    # terminal node is the one we wrote the golden case for.
    single_prog = Program(name="integration_single", nodes=[double])

    # 2. Write golden
    golden_dir = tmp_path / "goldens"
    write_golden("double", golden_dir, {"input": {"x": 5}, "expected": {"y": 10}})

    # 3. Verify golden
    failures = verify_golden(single_prog, golden_dir)
    assert failures == []

    # 4. Render SVG
    svg = render_program_svg(prog, title="integration_test")
    assert svg.startswith("<?xml")
    assert "</svg>" in svg


def test_cli_run_against_program_file(tmp_path: Path):
    prog_file = tmp_path / "p.py"
    prog_file.write_text(
        "from pydantic import BaseModel\n"
        "from snail import node, NodeContext, NodeResult, OODSignal, Program\n"
        "class In(BaseModel):\n    x: int\n"
        "class Ok(BaseModel):\n    y: int\n"
        "class Out(NodeResult):\n    ok: Ok | None = None\n    ood: OODSignal | None = None\n"
        "@node(name='double', input_schema=In, output_schema=Out, distribution='t')\n"
        "def double(ctx, w, p): return Out(ok=Ok(y=p.x*2))\n"
        "prog = Program(name='cli_test', nodes=[double])\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", str(prog_file), "--input", json.dumps({"x": 7})]
    )
    assert result.exit_code == 0
    assert "double" in result.output


def test_anthropic_provider_dispatch_end_to_end(tmp_path: Path, monkeypatch):
    """End-to-end: HostedNode with provider=anthropic hits the wire."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from snail.wrappers import HostedNode

    class HIn(BaseModel):
        text: str

    class HOk(BaseModel):
        endpoint: str
        prompt: str
        response: str
        provider: str
        model: str | None = None
        confidence: float | None = None

    class HOut(NodeResult):
        ok: HOk | None = None
        ood: OODSignal | None = None

    h = HostedNode(
        name="summarize",
        input_schema=HIn,
        output_schema=HOut,
        distribution="t",
        endpoint="anthropic://claude-sonnet-5",
        prompt_template="Summarize: {text}",
        api_key_env="ANTHROPIC_API_KEY",
        provider="anthropic",
    )
    fake = {
        "id": "x",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "summary text"}],
        "usage": {},
    }
    with patch("snail.providers.anthropic.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = fake
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "{}"
        ctx = NodeContext(
            program_name="t", run_id="r", node_name="summarize", metadata={}
        )
        r = h(ctx, HIn(text="hello"))
    assert r.is_ok
    assert r.ok.response == "summary text"
    assert r.ok.provider == "anthropic"
