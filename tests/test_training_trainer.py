"""Tests for train_recipe.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
import json
from pathlib import Path

from snail.training import NodeSpec, Recipe, train_recipe


def test_train_recipe_emits_frozen_weights(tmp_path: Path):
    recipe = Recipe(
        name="t",
        dataset="d",
        frozen_output=str(tmp_path / "out.snail.json"),
        node_specs=[
            NodeSpec("n", "In", "Out", "dist", 1, 0.001, "model@sha256:deadbeef")
        ],
        golden_cases=[],
    )

    def fake_model_forward(payload, node_name):
        return {"label": "ok", "confidence": 0.9}

    dataset = [{"text": f"sample-{i}"} for i in range(3)]
    out = train_recipe(recipe, fake_model_forward, dataset, output_dir=tmp_path)
    assert "n" in out
    assert Path(out["n"]).exists()
    assert Path(out["n"]).suffix == ".json"

    body = json.loads(Path(out["n"]).read_text())
    assert body["format"] == "snail-json-v1"
    assert body["node"] == "n"
    assert body["samples_seen"] == 3


def test_train_recipe_writes_trace(tmp_path: Path):
    recipe = Recipe(
        name="t",
        dataset="d",
        frozen_output="out.snail.json",  # relative path
        node_specs=[NodeSpec("n", "In", "Out", "dist")],
        golden_cases=[],
    )

    def f(payload, node_name):
        return {"confidence": 0.5}

    train_recipe(recipe, f, [{"x": 1}, {"x": 2}], output_dir=tmp_path)

    trace_path = tmp_path / "n_trace.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert len(trace) == 2
    assert trace[0]["output"]["confidence"] == 0.5


def test_train_recipe_loss_reflects_confidence(tmp_path: Path):
    recipe = Recipe(
        name="t",
        dataset="d",
        frozen_output="out.snail.json",
        node_specs=[NodeSpec("n", "In", "Out", "dist")],
        golden_cases=[],
    )

    def f(payload, node_name):
        return {"confidence": 0.8}  # loss = 0.2 per sample

    out = train_recipe(recipe, f, [{"x": 1}, {"x": 2}, {"x": 3}], output_dir=tmp_path)
    body = json.loads(Path(out["n"]).read_text())
    assert abs(body["final_loss"] - 0.2) < 1e-6
