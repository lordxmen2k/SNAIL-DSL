"""Tests for Recipe + YAML parser.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from pathlib import Path

from snail.training import GoldenCase, NodeSpec, Recipe, load_recipe


def test_load_recipe_from_yaml(tmp_path: Path):
    yaml = tmp_path / "test.recipe.yaml"
    yaml.write_text(
        "name: classify_intent\n"
        "dataset: customer_intents_v3\n"
        "frozen_output: weights/classify_intent.snail.json\n"
        "nodes:\n"
        "  - name: classify_intent\n"
        "    input_schema: Message\n"
        "    output_schema: Intent\n"
        "    distribution: customer_intents_v3\n"
        "    epochs: 5\n"
        "    learning_rate: 0.001\n"
        "    weight_pin: phi-4-mini-3.8b@sha256:abc123\n"
        "golden_cases:\n"
        '  - input: {"text": "I want a refund"}\n'
        '    expected: {"intent": "refund", "confidence_min": 0.7}\n'
    )
    r = load_recipe(yaml)
    assert r.name == "classify_intent"
    assert len(r.node_specs) == 1
    assert r.node_specs[0].name == "classify_intent"
    assert r.node_specs[0].epochs == 5
    assert len(r.golden_cases) == 1


def test_recipe_to_yaml_round_trip(tmp_path: Path):
    r = Recipe(
        name="t",
        dataset="d",
        frozen_output="out.json",
        node_specs=[
            NodeSpec("n", "In", "Out", "dist", 1, 0.001, "model@sha256:x")
        ],
        golden_cases=[],
    )
    p = tmp_path / "r.recipe.yaml"
    p.write_text(r.to_yaml())
    r2 = load_recipe(p)
    assert r2.name == r.name
    assert r2.node_specs[0].name == "n"
    assert r2.node_specs[0].weight_pin == "model@sha256:x"


def test_recipe_with_multiple_nodes(tmp_path: Path):
    yaml = tmp_path / "two_nodes.recipe.yaml"
    yaml.write_text(
        "name: two\n"
        "dataset: d\n"
        "frozen_output: out.json\n"
        "nodes:\n"
        "  - name: a\n"
        "    input_schema: In\n"
        "    output_schema: Mid\n"
        "    distribution: da\n"
        "  - name: b\n"
        "    input_schema: Mid\n"
        "    output_schema: Out\n"
        "    distribution: db\n"
        "    epochs: 3\n"
        "    learning_rate: 0.01\n"
    )
    r = load_recipe(yaml)
    assert len(r.node_specs) == 2
    assert r.node_specs[0].epochs == 1
    assert r.node_specs[1].epochs == 3
    assert r.node_specs[1].learning_rate == 0.01
