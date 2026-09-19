"""Tests for the SNAIL CLI.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path

from click.testing import CliRunner

from snail.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "inspect" in result.output
    assert "render" in result.output
    assert "train" in result.output
    assert "verify" in result.output


def test_cli_inspect_loads_program(tmp_path: Path):
    prog_file = tmp_path / "myprog.py"
    prog_file.write_text(
        "from pydantic import BaseModel\n"
        "from snail import node, NodeContext, NodeResult, OODSignal, Program\n"
        "class In(BaseModel):\n    x: int\n"
        "class Ok(BaseModel):\n    y: int\n"
        "class Out(NodeResult):\n    ok: Ok | None = None\n    ood: OODSignal | None = None\n"
        "@node(name='a', input_schema=In, output_schema=Out, distribution='t')\n"
        "def a(ctx, w, p): return Out(ok=Ok(y=p.x))\n"
        "my_program = Program(name='my_program', nodes=[a])\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(prog_file)])
    assert result.exit_code == 0
    assert "my_program" in result.output
    assert "a" in result.output


def test_cli_run_emits_manifest(tmp_path: Path):
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
    result = runner.invoke(cli, ["run", str(prog_file), "--input", json.dumps({"x": 7})])
    assert result.exit_code == 0
    assert "double" in result.output


def test_cli_render_writes_svg(tmp_path: Path):
    prog_file = tmp_path / "p.py"
    prog_file.write_text(
        "from pydantic import BaseModel\n"
        "from snail import node, NodeContext, NodeResult, OODSignal, Program\n"
        "class In(BaseModel):\n    x: int\n"
        "class Ok(BaseModel):\n    y: int\n"
        "class Out(NodeResult):\n    ok: Ok | None = None\n    ood: OODSignal | None = None\n"
        "@node(name='a', input_schema=In, output_schema=Out, distribution='t')\n"
        "def a(ctx, w, p): return Out(ok=Ok(y=p.x))\n"
        "@node(name='b', input_schema=In, output_schema=Out, distribution='t')\n"
        "def b(ctx, w, p): return Out(ok=Ok(y=p.x))\n"
        "prog = Program(name='cli_render', nodes=[a, b])\n"
    )
    out = tmp_path / "out.svg"
    runner = CliRunner()
    result = runner.invoke(cli, ["render", str(prog_file), "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert out.read_text().startswith("<?xml")


def test_cli_train_emits_weights(tmp_path: Path):
    recipe = tmp_path / "test.recipe.yaml"
    recipe.write_text(
        "name: t\n"
        "dataset: d\n"
        "frozen_output: out.snail.json\n"
        "nodes:\n"
        "  - name: n\n"
        "    input_schema: In\n"
        "    output_schema: Out\n"
        "    distribution: d\n"
        "    epochs: 1\n"
        "    learning_rate: 0.001\n"
        "    weight_pin: model@sha256:x\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(recipe), "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "n" in result.output
    assert any(tmp_path.rglob("*.snail.json"))


def test_cli_verify_passes(tmp_path: Path):
    prog_file = tmp_path / "p.py"
    prog_file.write_text(
        "from pydantic import BaseModel\n"
        "from snail import node, NodeContext, NodeResult, OODSignal, Program\n"
        "class In(BaseModel):\n    x: int\n"
        "class Ok(BaseModel):\n    y: int\n"
        "class Out(NodeResult):\n    ok: Ok | None = None\n    ood: OODSignal | None = None\n"
        "@node(name='double', input_schema=In, output_schema=Out, distribution='t')\n"
        "def double(ctx, w, p): return Out(ok=Ok(y=p.x*2))\n"
        "prog = Program(name='p', nodes=[double])\n"
    )
    golden_dir = tmp_path / "goldens"
    (golden_dir / "golden" / "double").mkdir(parents=True)
    (golden_dir / "golden" / "double" / "case.json").write_text(
        json.dumps({"input": {"x": 5}, "expected": {"y": 10}})
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["verify", str(prog_file), "--golden-dir", str(golden_dir)])
    assert result.exit_code == 0
    assert "passed" in result.output or "0 failures" in result.output


def test_cli_verify_fails_on_mismatch(tmp_path: Path):
    prog_file = tmp_path / "p.py"
    prog_file.write_text(
        "from pydantic import BaseModel\n"
        "from snail import node, NodeContext, NodeResult, OODSignal, Program\n"
        "class In(BaseModel):\n    x: int\n"
        "class Ok(BaseModel):\n    y: int\n"
        "class Out(NodeResult):\n    ok: Ok | None = None\n    ood: OODSignal | None = None\n"
        "@node(name='double', input_schema=In, output_schema=Out, distribution='t')\n"
        "def double(ctx, w, p): return Out(ok=Ok(y=p.x*2))\n"
        "prog = Program(name='p', nodes=[double])\n"
    )
    golden_dir = tmp_path / "goldens"
    (golden_dir / "golden" / "double").mkdir(parents=True)
    (golden_dir / "golden" / "double" / "case.json").write_text(
        json.dumps({"input": {"x": 5}, "expected": {"y": 999}})
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["verify", str(prog_file), "--golden-dir", str(golden_dir)])
    assert result.exit_code == 1
    assert "failure" in (result.output + (result.stderr or ""))
