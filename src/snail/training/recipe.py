"""Recipe — declarative spec for a SNAIL training run.

A Recipe says: "this distribution, this many epochs, this weight pin,
these golden cases." The trainer reads the recipe and emits frozen
weights + a manifest.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeSpec:
    """Per-node spec inside a recipe."""

    name: str
    input_schema: str
    output_schema: str
    distribution: str
    epochs: int = 1
    learning_rate: float = 0.001
    weight_pin: str = ""


@dataclass
class GoldenCase:
    """A golden test case — input + expected output."""

    input: dict[str, Any]
    expected: dict[str, Any]


@dataclass
class Recipe:
    """A training recipe — collection of node specs + golden cases."""

    name: str
    dataset: str
    frozen_output: str
    node_specs: list[NodeSpec] = field(default_factory=list)
    golden_cases: list[GoldenCase] = field(default_factory=list)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(
            {
                "name": self.name,
                "dataset": self.dataset,
                "frozen_output": self.frozen_output,
                "nodes": [
                    {
                        "name": n.name,
                        "input_schema": n.input_schema,
                        "output_schema": n.output_schema,
                        "distribution": n.distribution,
                        "epochs": n.epochs,
                        "learning_rate": n.learning_rate,
                        "weight_pin": n.weight_pin,
                    }
                    for n in self.node_specs
                ],
                "golden_cases": [
                    {"input": c.input, "expected": c.expected}
                    for c in self.golden_cases
                ],
            },
            sort_keys=False,
        )
