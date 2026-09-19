"""Recipe parser — load a `.recipe.yaml` file into a Recipe.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from pathlib import Path
import yaml

from snail.training.recipe import GoldenCase, NodeSpec, Recipe


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe from a YAML file."""
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    nodes = [NodeSpec(**n) for n in data.get("nodes", [])]
    golden = [GoldenCase(**c) for c in data.get("golden_cases", [])]
    return Recipe(
        name=data["name"],
        dataset=data["dataset"],
        frozen_output=data["frozen_output"],
        node_specs=nodes,
        golden_cases=golden,
    )
